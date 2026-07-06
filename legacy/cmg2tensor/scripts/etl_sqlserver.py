"""
ETL: data/processed  →  SQL Server
Esquema destino: tblModelo, tblCorrida, tblPropiedadesMalla,
                 tblResultadoVariables_Header, tblResultadoVariables_Detalle,
                 tblTasaInyeccion

Dependencias:
    pip install pyodbc torch numpy

Uso básico (Windows Auth):
    python scripts/etl_sqlserver.py \
        --server MISERVIDOR --database SimulacionesCO2 --trusted-connection

Uso con usuario/contraseña:
    python scripts/etl_sqlserver.py \
        --server MISERVIDOR --database SimulacionesCO2 --user sa --password secret

Cadena de conexión directa:
    python scripts/etl_sqlserver.py \
        --connection-string "DRIVER={ODBC Driver 17 for SQL Server};SERVER=...;DATABASE=...;UID=...;PWD=..."

Opciones útiles:
    --max-simulations 5   cargar solo las primeras 5 corridas (modo prueba)
    --model CO2_951       nombre del modelo (default: CO2_951)
    --base-date 2025-10-01  fecha inicio de la simulación
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pyodbc
import torch

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS Y CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
PROCESSED_ROOT = Path("data/processed")

MODEL_NAME = "CO2_951"
BASE_DATE  = date(2025, 10, 1)
DIM_K, DIM_J, DIM_I = 97, 50, 50

# Carpetas de variables estáticas que varían por corrida
STATIC_FOLDERS: dict[str, str] = {
    "cohesion_layer_cubes": "COHESION",
    "afi_layer_cubes":      "AFI",
}

# Carpetas de propiedades de reservorio (iguales en todas las corridas)
RESERVOIR_FOLDERS: dict[str, str] = {
    "porosity_layer_cubes":      "POROSITY",
    "permeability_layer_cubes":  "PERMEABILITY",
}

# Variables dinámicas (362 timesteps) — conviven en la misma carpeta/archivo
DYNAMIC_FOLDER    = "layer_cubes"
DYNAMIC_VARIABLES = ["SF", "VD"]

VARIABLE_UNITS: dict[str, str] = {
    "AFI":          "deg",
    "COHESION":     "psi",
    "SF":           "adim",
    "VD":           "ft",
    "POROSITY":     "frac",
    "PERMEABILITY": "md",
}

WELL_NAME_MAP: dict[str, str] = {
    "injection_tene_1": "TENE-1",
    "injection_tene_2": "TENE-2",
}

BATCH_SIZE = 500   # filas por executemany (~5 MB por lote con blobs de 10 KB)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("etl_sqlserver.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONEXIÓN
# ─────────────────────────────────────────────────────────────────────────────
def build_connection(args: argparse.Namespace) -> pyodbc.Connection:
    if args.connection_string:
        conn_str = args.connection_string
    else:
        parts = [
            f"DRIVER={{{args.driver}}}",
            f"SERVER={args.server}",
            f"DATABASE={args.database}",
        ]
        if args.trusted_connection:
            parts.append("Trusted_Connection=yes")
        else:
            parts += [f"UID={args.user}", f"PWD={args.password}"]
        conn_str = ";".join(parts)

    conn = pyodbc.connect(conn_str, autocommit=False)
    conn.setdecoding(pyodbc.SQL_CHAR, encoding="utf-8")
    conn.setencoding(encoding="utf-8")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Convierte un slice 2D [J, I] a float32 bytes en orden row-major."""
    return t.numpy().astype(np.float32).tobytes()


def days_to_date(base: date, days: float) -> date:
    return base + timedelta(days=float(days))


def get_split(sim_path: Path) -> str:
    return sim_path.parent.name if sim_path.parent.name in ("train", "test") else "train"


def get_codigo_corrida(sim_name: str) -> int:
    return int(sim_name.split("_")[0])


def get_tipo_muestreo(sim_name: str) -> str:
    parts = sim_name.split("_", 1)
    return parts[1] if len(parts) > 1 else "unknown"


def load_timeline(sim_path: Path, folder_name: str) -> list[dict]:
    """Lee el JSON de timeline para una carpeta de variable."""
    stem = folder_name.replace("_layer_cubes", "").replace("layer_cubes", "sf_vd")
    candidates = [
        sim_path / f"timeline_{stem}.json",
        sim_path / "timeline_sf_vd.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            ids  = data.get("time_ids", [])
            days = data.get("time_ids_days", [0.0] * len(ids))
            return [{"ts": int(i), "days": float(d)} for i, d in zip(ids, days)]
    return []


def iter_simulations(limit: int | None = None) -> list[Path]:
    sims: list[Path] = []
    for split in ("train", "test"):
        d = PROCESSED_ROOT / split
        if d.exists():
            sims.extend(sorted(d.iterdir()))
    if not sims:
        sims = [
            p for p in sorted(PROCESSED_ROOT.iterdir())
            if p.is_dir() and p.name not in ("train", "test", "train_test_norm")
        ]
    sims = [s for s in sims if s.is_dir()]
    return sims[:limit] if limit else sims


# ─────────────────────────────────────────────────────────────────────────────
# INSERT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def insert_modelo(cur: pyodbc.Cursor, modelo: str) -> None:
    cur.execute("""
        IF NOT EXISTS (SELECT 1 FROM tblModelo WHERE Modelo = ?)
            INSERT INTO tblModelo (Modelo, DimK, DimJ, DimI, Descripcion)
            VALUES (?, ?, ?, ?, ?)
    """, modelo, modelo, DIM_K, DIM_J, DIM_I, "Modelo geomecánico CO2 CCS")
    log.info("tblModelo: %s  (K=%d J=%d I=%d)", modelo, DIM_K, DIM_J, DIM_I)


def insert_corrida(cur: pyodbc.Cursor, sim_path: Path, modelo: str,
                   base_date: date) -> int:
    cur.execute("""
        INSERT INTO tblCorrida
            (Modelo, CodigoCorrida, TipoMuestreo, Particion, FechaModelo, FechaCarga)
        VALUES (?, ?, ?, ?, ?, ?)
    """, modelo,
        get_codigo_corrida(sim_path.name),
        get_tipo_muestreo(sim_path.name),
        get_split(sim_path),
        base_date,
        datetime.now())
    cur.execute("SELECT SCOPE_IDENTITY()")
    return int(cur.fetchone()[0])


def insert_header(cur: pyodbc.Cursor, id_corrida: int,
                  variable: str, timestep: int, fecha: date) -> int:
    cur.execute("""
        INSERT INTO tblResultadoVariables_Header
            (ID_Corrida, Variable, TimeStep, FechaResultado, Unidad)
        VALUES (?, ?, ?, ?, ?)
    """, id_corrida, variable, timestep, fecha, VARIABLE_UNITS.get(variable))
    cur.execute("SELECT SCOPE_IDENTITY()")
    return int(cur.fetchone()[0])


def flush_detalle(cur: pyodbc.Cursor, batch: list) -> None:
    if not batch:
        return
    cur.executemany("""
        INSERT INTO tblResultadoVariables_Detalle (ID_Header, K, CapaDatos)
        VALUES (?, ?, ?)
    """, batch)
    batch.clear()


# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO POR TIPO DE VARIABLE
# ─────────────────────────────────────────────────────────────────────────────
def process_static_variable(cur: pyodbc.Cursor, sim_path: Path,
                             id_corrida: int, folder_name: str,
                             variable: str, base_date: date) -> None:
    """Variables estáticas (1 timestep): AFI, COHESION."""
    folder = sim_path / folder_name
    if not folder.exists():
        return

    id_header = insert_header(cur, id_corrida, variable, 0, base_date)

    batch: list = []
    pt_files = sorted(folder.glob("*.pt"))
    for pt_path in pt_files:
        try:
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            log.warning("    No se pudo leer %s: %s", pt_path.name, e)
            continue

        cube    = blob.get("cube")       # [1, 1, J, I]
        layer_k = int(blob.get("layer_k", 1))
        if cube is None:
            continue

        batch.append((id_header, layer_k, pyodbc.Binary(tensor_to_bytes(cube[0, 0]))))
        if len(batch) >= BATCH_SIZE:
            flush_detalle(cur, batch)

    flush_detalle(cur, batch)
    log.info("    %s  header=%d  capas=%d", variable, id_header, len(pt_files))


def process_reservoir_property(cur: pyodbc.Cursor, sim_path: Path,
                                folder_name: str, variable: str,
                                modelo: str) -> None:
    """Propiedades de reservorio: insertar solo una vez para todo el modelo."""
    cur.execute(
        "SELECT COUNT(1) FROM tblPropiedadesMalla WHERE Modelo = ? AND Variable = ?",
        modelo, variable,
    )
    if cur.fetchone()[0] > 0:
        log.info("    %s ya existe en tblPropiedadesMalla — omitido.", variable)
        return

    folder = sim_path / folder_name
    if not folder.exists():
        log.warning("    Carpeta no encontrada: %s", folder)
        return

    batch: list = []
    pt_files = sorted(folder.glob("*.pt"))
    for pt_path in pt_files:
        try:
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            log.warning("    No se pudo leer %s: %s", pt_path.name, e)
            continue

        cube    = blob.get("cube")
        layer_k = int(blob.get("layer_k", 1))
        if cube is None:
            continue

        batch.append((modelo, variable, layer_k, pyodbc.Binary(tensor_to_bytes(cube[0, 0]))))
        if len(batch) >= BATCH_SIZE:
            cur.executemany("""
                INSERT INTO tblPropiedadesMalla (Modelo, Variable, K, CapaDatos)
                VALUES (?, ?, ?, ?)
            """, batch)
            batch.clear()

    if batch:
        cur.executemany("""
            INSERT INTO tblPropiedadesMalla (Modelo, Variable, K, CapaDatos)
            VALUES (?, ?, ?, ?)
        """, batch)

    log.info("    %s → tblPropiedadesMalla  capas=%d", variable, len(pt_files))


def process_dynamic_variables(cur: pyodbc.Cursor, sim_path: Path,
                               id_corrida: int, base_date: date) -> None:
    """Variables dinámicas (362 timesteps): SF y VD desde layer_cubes/."""
    folder = sim_path / DYNAMIC_FOLDER
    if not folder.exists():
        return

    timeline = load_timeline(sim_path, DYNAMIC_FOLDER)
    if not timeline:
        log.warning("    Sin timeline para layer_cubes en %s", sim_path.name)
        return

    # Insertar todos los headers (SF × 362 ts + VD × 362 ts = 724 headers)
    # y construir mapa {(variable, timestep_index) → id_header}
    header_map: dict[tuple[str, int], int] = {}
    for entry in timeline:
        ts    = entry["ts"]
        fecha = days_to_date(base_date, entry["days"])
        for var in DYNAMIC_VARIABLES:
            header_map[(var, ts)] = insert_header(cur, id_corrida, var, ts, fecha)

    log.info("    Headers dinámicos: %d", len(header_map))

    # Procesar archivos .pt — cada uno es una capa K con shape [2, T, J, I]
    batch: list = []
    pt_files = sorted(folder.glob("*.pt"))
    for pt_path in pt_files:
        try:
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            log.warning("    No se pudo leer %s: %s", pt_path.name, e)
            continue

        cube     = blob.get("cube")       # [2, T, J, I]  — índice 0=SF, 1=VD
        time_ids = blob.get("time_ids", [])
        layer_k  = int(blob.get("layer_k", 1))
        if cube is None:
            continue

        n_t = cube.shape[1]
        for t in range(n_t):
            ts = int(time_ids[t]) if t < len(time_ids) else t
            for v_idx, var in enumerate(DYNAMIC_VARIABLES):
                hid = header_map.get((var, ts))
                if hid is None:
                    continue
                batch.append((hid, layer_k, pyodbc.Binary(tensor_to_bytes(cube[v_idx, t]))))
                if len(batch) >= BATCH_SIZE:
                    flush_detalle(cur, batch)

    flush_detalle(cur, batch)
    log.info("    layer_cubes: %d archivos procesados", len(pt_files))


def process_injection(cur: pyodbc.Cursor, sim_path: Path,
                      id_corrida: int, modelo: str, base_date: date) -> None:
    """Tasas de inyección por pozo y timestep."""
    inj_folder = sim_path / "injection_name_tensors"
    inj_report = sim_path / "injection_name_tensors_report.json"
    if not inj_folder.exists():
        return

    parametro = "Gas Rate SC - Monthly (ft3/day)"
    if inj_report.exists():
        try:
            rep = json.loads(inj_report.read_text(encoding="utf-8"))
            parametro = rep.get("parameter", parametro)
        except Exception:
            pass

    file_modelo = f"{modelo}_{get_codigo_corrida(sim_path.name):03d}"
    codigo      = get_codigo_corrida(sim_path.name)

    rows: list = []
    for pt_path in sorted(inj_folder.glob("*.pt")):
        try:
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            log.warning("    No se pudo leer inyección %s: %s", pt_path.name, e)
            continue

        pozo      = WELL_NAME_MAP.get(pt_path.stem, pt_path.stem)
        series    = blob.get("series")      # tensor [T]
        time_days = blob.get("time_days", [])
        if series is None:
            continue

        for ts, (val, td) in enumerate(zip(series.tolist(), time_days)):
            rows.append((
                id_corrida,
                pozo,
                ts,
                float(td),
                days_to_date(base_date, td),
                parametro,
                float(val),
                file_modelo,
            ))

    if rows:
        cur.executemany("""
            INSERT INTO tblTasaInyeccion
                (ID_Corrida, NombrePozo, TimeStep, TimeDay, Fecha,
                 Parametro, Valor, FileModelo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        log.info("    Inyección: %d filas  (%d pozos)", len(rows),
                 len({r[1] for r in rows}))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETL data/processed → SQL Server (esquema VARBINARY)"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--connection-string",
                     help="Cadena de conexión ODBC completa")
    grp.add_argument("--server", default="localhost",
                     help="Servidor SQL Server (default: localhost)")

    parser.add_argument("--database",           default="SimulacionesCO2")
    parser.add_argument("--driver",             default="ODBC Driver 17 for SQL Server")
    parser.add_argument("--user",               default="")
    parser.add_argument("--password",           default="")
    parser.add_argument("--trusted-connection", action="store_true",
                        help="Autenticación Windows")
    parser.add_argument("--model",              default=MODEL_NAME,
                        help="Nombre del modelo (default: CO2_951)")
    parser.add_argument("--base-date",          default="2025-10-01",
                        help="Fecha inicio simulación YYYY-MM-DD (default: 2025-10-01)")
    parser.add_argument("--max-simulations",    type=int, default=None,
                        help="Limitar número de corridas (útil para pruebas)")
    args = parser.parse_args()

    modelo    = args.model
    base_date = date.fromisoformat(args.base_date)

    log.info("Conectando a SQL Server...")
    conn = build_connection(args)
    cur  = conn.cursor()
    log.info("Conexión OK.")

    # Modelo base (idempotente)
    insert_modelo(cur, modelo)
    conn.commit()

    simulations = iter_simulations(args.max_simulations)
    total       = len(simulations)
    log.info("Corridas a procesar: %d", total)

    for i, sim_path in enumerate(simulations, 1):
        log.info("─" * 60)
        log.info("[%d/%d] %s", i, total, sim_path.name)

        try:
            id_corrida = insert_corrida(cur, sim_path, modelo, base_date)
            log.info("  tblCorrida  ID_Corrida=%d", id_corrida)

            # Propiedades de reservorio — solo se insertan la primera vez
            for folder_name, variable in RESERVOIR_FOLDERS.items():
                process_reservoir_property(cur, sim_path, folder_name,
                                           variable, modelo)

            # Variables estáticas por corrida (AFI, Cohesion)
            for folder_name, variable in STATIC_FOLDERS.items():
                process_static_variable(cur, sim_path, id_corrida,
                                        folder_name, variable, base_date)

            # Variables dinámicas (SF, VD — 362 timesteps)
            process_dynamic_variables(cur, sim_path, id_corrida, base_date)

            # Tasas de inyección
            process_injection(cur, sim_path, id_corrida, modelo, base_date)

            conn.commit()
            log.info("  COMMIT OK")

        except Exception:
            conn.rollback()
            log.error("  ERROR en %s — rollback", sim_path.name, exc_info=True)

    cur.close()
    conn.close()
    log.info("ETL finalizado. %d corridas procesadas.", total)


if __name__ == "__main__":
    main()
