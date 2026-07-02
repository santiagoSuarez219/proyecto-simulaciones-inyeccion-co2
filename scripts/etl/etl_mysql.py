"""
ETL: data/processed  →  MySQL 8
Esquema destino: tblModelo, tblCorrida, tblPropiedadesMalla,
                 tblResultadoVariables_Header, tblResultadoVariables_Detalle,
                 tblTasaInyeccion

Dependencias:
    pip install mysql-connector-python torch numpy

Uso:
    python scripts/etl_mysql.py --user root --password TUPASS
    python scripts/etl_mysql.py --user root --password TUPASS --max-simulations 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import mysql.connector
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS Y CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
PROCESSED_ROOT = Path("data/processed")

MODEL_NAME = "CO2_951"
BASE_DATE  = date(2025, 10, 1)
DIM_K, DIM_J, DIM_I = 97, 50, 50

STATIC_FOLDERS: dict[str, str] = {
    "cohesion_layer_cubes": "COHESION",
    "afi_layer_cubes":      "AFI",
}
RESERVOIR_FOLDERS: dict[str, str] = {
    "porosity_layer_cubes":      "POROSITY",
    "permeability_layer_cubes":  "PERMEABILITY",
}
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

BATCH_SIZE = 1000

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("etl_mysql.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONEXIÓN Y ESQUEMA
# ─────────────────────────────────────────────────────────────────────────────
def build_connection(args: argparse.Namespace,
                     database: str | None = None) -> mysql.connector.MySQLConnection:
    kwargs: dict = dict(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        charset="utf8mb4",
        autocommit=False,
    )
    if database:
        kwargs["database"] = database
    conn = mysql.connector.connect(**kwargs)
    cur = conn.cursor()
    cur.execute("SET SESSION innodb_lock_wait_timeout = 600")
    cur.execute("SET SESSION foreign_key_checks = 0")
    cur.execute("SET SESSION unique_checks = 0")
    cur.close()
    return conn


def apply_schema(conn: mysql.connector.MySQLConnection,
                 ddl_path: Path) -> None:
    ddl = ddl_path.read_text(encoding="utf-8")
    cur = conn.cursor()
    for statement in ddl.split(";"):
        stmt = statement.strip()
        if stmt:
            cur.execute(stmt)
    conn.commit()
    cur.close()
    log.info("Esquema aplicado desde %s", ddl_path)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def tensor_to_bytes(t: torch.Tensor) -> bytes:
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
    stem = folder_name.replace("_layer_cubes", "").replace("layer_cubes", "sf_vd")
    for path in [sim_path / f"timeline_{stem}.json", sim_path / "timeline_sf_vd.json"]:
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
def insert_modelo(cur, modelo: str) -> None:
    cur.execute("""
        INSERT IGNORE INTO tblModelo (Modelo, DimK, DimJ, DimI, Descripcion)
        VALUES (%s, %s, %s, %s, %s)
    """, (modelo, DIM_K, DIM_J, DIM_I, "Modelo geomecánico CO2 CCS"))
    log.info("tblModelo: %s  (K=%d J=%d I=%d)", modelo, DIM_K, DIM_J, DIM_I)


def insert_corrida(cur, sim_path: Path, modelo: str, base_date: date) -> int:
    cur.execute("""
        INSERT INTO tblCorrida
            (Modelo, NombreCorrida, CodigoCorrida, TipoMuestreo, Particion, FechaModelo, FechaCarga)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        modelo,
        sim_path.name,
        get_codigo_corrida(sim_path.name),
        get_tipo_muestreo(sim_path.name),
        get_split(sim_path),
        base_date,
        datetime.now(),
    ))
    return cur.lastrowid


def insert_header(cur, id_corrida: int,
                  variable: str, timestep: int, fecha: date) -> int:
    cur.execute("""
        INSERT INTO tblResultadoVariables_Header
            (ID_Corrida, Variable, TimeStep, FechaResultado, Unidad)
        VALUES (%s, %s, %s, %s, %s)
    """, (id_corrida, variable, timestep, fecha, VARIABLE_UNITS.get(variable)))
    return cur.lastrowid


def flush_detalle(cur, batch: list) -> None:
    if not batch:
        return
    cur.executemany("""
        INSERT INTO tblResultadoVariables_Detalle (ID_Header, K, CapaDatos)
        VALUES (%s, %s, %s)
    """, batch)
    batch.clear()


# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO POR TIPO DE VARIABLE
# ─────────────────────────────────────────────────────────────────────────────
def process_static_variable(cur, sim_path: Path, id_corrida: int,
                             folder_name: str, variable: str,
                             base_date: date) -> None:
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

        cube    = blob.get("cube")
        layer_k = int(blob.get("layer_k", 1))
        if cube is None:
            continue

        batch.append((id_header, layer_k, tensor_to_bytes(cube[0, 0])))
        if len(batch) >= BATCH_SIZE:
            flush_detalle(cur, batch)

    flush_detalle(cur, batch)
    log.info("    %s  header=%d  capas=%d", variable, id_header, len(pt_files))


def process_reservoir_property(cur, sim_path: Path, folder_name: str,
                                variable: str, modelo: str) -> None:
    cur.execute(
        "SELECT COUNT(1) FROM tblPropiedadesMalla WHERE Modelo = %s AND Variable = %s",
        (modelo, variable),
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

        batch.append((modelo, variable, layer_k, tensor_to_bytes(cube[0, 0])))
        if len(batch) >= BATCH_SIZE:
            cur.executemany("""
                INSERT INTO tblPropiedadesMalla (Modelo, Variable, K, CapaDatos)
                VALUES (%s, %s, %s, %s)
            """, batch)
            batch.clear()

    if batch:
        cur.executemany("""
            INSERT INTO tblPropiedadesMalla (Modelo, Variable, K, CapaDatos)
            VALUES (%s, %s, %s, %s)
        """, batch)

    log.info("    %s -> tblPropiedadesMalla  capas=%d", variable, len(pt_files))


def process_dynamic_variables(cur, sim_path: Path, id_corrida: int,
                               base_date: date) -> None:
    folder = sim_path / DYNAMIC_FOLDER
    if not folder.exists():
        return

    timeline = load_timeline(sim_path, DYNAMIC_FOLDER)
    if not timeline:
        log.warning("    Sin timeline para layer_cubes en %s", sim_path.name)
        return

    # Insertar los 724 headers (SF+VD × 362 ts) y construir el mapa de IDs
    header_map: dict[tuple[str, int], int] = {}
    for entry in timeline:
        ts    = entry["ts"]
        fecha = days_to_date(base_date, entry["days"])
        for var in DYNAMIC_VARIABLES:
            header_map[(var, ts)] = insert_header(cur, id_corrida, var, ts, fecha)

    log.info("    Headers dinámicos: %d", len(header_map))

    batch: list = []
    pt_files = sorted(folder.glob("*.pt"))
    for pt_path in pt_files:
        try:
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            log.warning("    No se pudo leer %s: %s", pt_path.name, e)
            continue

        cube     = blob.get("cube")      # [2, T, J, I]
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
                batch.append((hid, layer_k, tensor_to_bytes(cube[v_idx, t])))
                if len(batch) >= BATCH_SIZE:
                    flush_detalle(cur, batch)

    flush_detalle(cur, batch)
    log.info("    layer_cubes: %d archivos procesados", len(pt_files))


def process_injection(cur, sim_path: Path, id_corrida: int,
                      modelo: str, base_date: date) -> None:
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

    rows: list = []
    for pt_path in sorted(inj_folder.glob("*.pt")):
        try:
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            log.warning("    No se pudo leer inyección %s: %s", pt_path.name, e)
            continue

        pozo      = blob.get("name") or WELL_NAME_MAP.get(pt_path.stem, pt_path.stem)
        series    = blob.get("tensor")
        time_days = blob.get("time_ids_days", [])
        if series is None:
            continue

        for ts, (val, td) in enumerate(zip(series.tolist(), time_days)):
            rows.append((
                id_corrida, pozo, ts,
                float(td), days_to_date(base_date, td),
                parametro, float(val), file_modelo,
            ))

    if rows:
        cur.executemany("""
            INSERT INTO tblTasaInyeccion
                (ID_Corrida, NombrePozo, TimeStep, TimeDay, Fecha,
                 Parametro, Valor, FileModelo)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, rows)
        log.info("    Inyección: %d filas  (%d pozos)", len(rows),
                 len({r[1] for r in rows}))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETL data/processed → MySQL (esquema LONGBLOB)"
    )
    parser.add_argument("--host",             default="localhost")
    parser.add_argument("--port",             type=int, default=3306)
    parser.add_argument("--database",         default="SimulacionesCO2")
    parser.add_argument("--user",             default="root")
    parser.add_argument("--password",         default="")
    parser.add_argument("--model",            default=MODEL_NAME)
    parser.add_argument("--base-date",        default="2025-10-01")
    parser.add_argument("--max-simulations",  type=int, default=None,
                        help="Limitar corridas (útil para pruebas)")
    parser.add_argument("--apply-schema",     action="store_true",
                        help="Crear tablas antes de cargar datos")
    parser.add_argument("--rebuild-indexes",  action="store_true",
                        help="Solo reconstruir índices secundarios al final (no cargar datos)")
    args = parser.parse_args()

    modelo    = args.model
    base_date = date.fromisoformat(args.base_date)

    log.info("Conectando a MySQL %s:%d ...", args.host, args.port)
    if args.apply_schema:
        # Conectar sin BD para poder crearla
        conn = build_connection(args)
        ddl_path = Path("sql/ddl_mysql.sql")
        apply_schema(conn, ddl_path)
        conn.close()

    log.info("Conectando a %s ...", args.database)
    conn = build_connection(args, database=args.database)
    log.info("Conexión OK.")

    cur = conn.cursor()

    # Modo solo reconstruir índices
    if args.rebuild_indexes:
        _rebuild_indexes(cur, conn)
        cur.close()
        conn.close()
        return

    # Soltar índices secundarios antes de la carga masiva
    _drop_indexes(cur, conn)

    # Modelo base (idempotente vía INSERT IGNORE)
    insert_modelo(cur, modelo)
    conn.commit()

    # Corridas ya cargadas — para saltar en modo resume
    cur.execute("SELECT NombreCorrida FROM tblCorrida WHERE Modelo = %s", (modelo,))
    ya_cargadas = {r[0] for r in cur.fetchall()}
    if ya_cargadas:
        log.info("Corridas ya en BD: %d — se omitiran.", len(ya_cargadas))

    simulations = iter_simulations(args.max_simulations)
    total       = len(simulations)
    log.info("Corridas a procesar: %d", total)

    for i, sim_path in enumerate(simulations, 1):
        if sim_path.name in ya_cargadas:
            log.info("[%d/%d] %s — ya cargada, omitida.", i, total, sim_path.name)
            continue

        log.info("-" * 60)
        log.info("[%d/%d] %s", i, total, sim_path.name)

        id_corrida: int | None = None
        try:
            id_corrida = insert_corrida(cur, sim_path, modelo, base_date)
            conn.commit()  # libera el lock en tblCorrida inmediatamente
            log.info("  tblCorrida  ID_Corrida=%d", id_corrida)

            for folder_name, variable in RESERVOIR_FOLDERS.items():
                process_reservoir_property(cur, sim_path, folder_name,
                                           variable, modelo)
            conn.commit()

            for folder_name, variable in STATIC_FOLDERS.items():
                process_static_variable(cur, sim_path, id_corrida,
                                        folder_name, variable, base_date)
            conn.commit()

            process_dynamic_variables(cur, sim_path, id_corrida, base_date)
            conn.commit()

            process_injection(cur, sim_path, id_corrida, modelo, base_date)
            conn.commit()
            log.info("  COMMIT OK")

        except Exception:
            conn.rollback()
            if id_corrida is not None:
                # la corrida fue commiteada — borrarla para que el próximo run la reintente
                cur.execute("DELETE FROM tblCorrida WHERE ID_Corrida = %s", (id_corrida,))
                conn.commit()
            log.error("  ERROR en %s — rollback", sim_path.name, exc_info=True)

    # Reconstruir índices secundarios al terminar la carga
    _rebuild_indexes(cur, conn)

    cur.close()
    conn.close()
    log.info("ETL finalizado. %d corridas procesadas.", total)


# ─────────────────────────────────────────────────────────────────────────────
# GESTIÓN DE ÍNDICES SECUNDARIOS
# ─────────────────────────────────────────────────────────────────────────────
_SECONDARY_INDEXES = [
    ("tblResultadoVariables_Detalle",  "IX_tblRVD_Header_K",
     "CREATE INDEX IX_tblRVD_Header_K ON tblResultadoVariables_Detalle (ID_Header, K)"),
    ("tblResultadoVariables_Header",   "IX_tblRVH_Corrida_Variable",
     "CREATE INDEX IX_tblRVH_Corrida_Variable ON tblResultadoVariables_Header (ID_Corrida, Variable)"),
    ("tblResultadoVariables_Header",   "IX_tblRVH_Corrida_Variable_TS",
     "CREATE INDEX IX_tblRVH_Corrida_Variable_TS ON tblResultadoVariables_Header (ID_Corrida, Variable, TimeStep)"),
    ("tblCorrida",                     "IX_tblCorrida_Modelo_Tipo",
     "CREATE INDEX IX_tblCorrida_Modelo_Tipo ON tblCorrida (Modelo, TipoMuestreo)"),
    ("tblTasaInyeccion",               "IX_tblTI_Corrida_Pozo",
     "CREATE INDEX IX_tblTI_Corrida_Pozo ON tblTasaInyeccion (ID_Corrida, NombrePozo)"),
]


def _drop_indexes(cur, conn) -> None:
    log.info("Soltando índices secundarios para carga masiva ...")
    for table, idx, _ in _SECONDARY_INDEXES:
        cur.execute(
            "SELECT COUNT(1) FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND INDEX_NAME = %s",
            (table, idx),
        )
        if cur.fetchone()[0]:
            cur.execute(f"DROP INDEX `{idx}` ON `{table}`")
            log.info("  DROP INDEX %s.%s", table, idx)
    conn.commit()
    log.info("Índices secundarios eliminados — carga sin overhead de B-tree.")


def _rebuild_indexes(cur, conn) -> None:
    log.info("Reconstruyendo índices secundarios ...")
    for table, idx, ddl in _SECONDARY_INDEXES:
        cur.execute(
            "SELECT COUNT(1) FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND INDEX_NAME = %s",
            (table, idx),
        )
        if cur.fetchone()[0] == 0:
            log.info("  CREATE INDEX %s.%s ...", table, idx)
            cur.execute(ddl)
            conn.commit()
            log.info("  OK")
        else:
            log.info("  %s.%s ya existe — omitido.", table, idx)
    log.info("Índices reconstruidos.")


if __name__ == "__main__":
    main()
