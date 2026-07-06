# cmg2tensor

Convierte salidas CMG (`.txt`) a tensores por capa para entrenamiento/análisis de modelos de inyección de CO₂.

## Qué hace

- Procesa `SF` + `VD` en un tensor por capa con shape `(V, T, NJ, NI)`, `V=[SF, VD]`.
- Procesa variables individuales opcionales (`Permeability`, `Porosity`, `Cohesion`, `AFI`, `Pressure`, `Gas Saturation`) con `V=1`.
- Procesa inyección desde Excel (`Time (day)`, `Name`, `Value`) para `TENE-1` y `TENE-2`.
- Convierte tiempo de días a meses en `time_ids` y conserva `time_ids_days`.
- Guarda en `.pt` por defecto (`.npz` opcional).
- Incluye reporte JSON por pipeline y tiempos de ejecución.
- Pipeline paralelo de dos fases con `ProcessPoolExecutor` (`--parallel`).
- Split estratificado train/test: simulaciones `train` se normalizan, `test` no.
- Histogramas globales por capas sin cargar todos los cubos a RAM.
- ETL a MySQL / SQL Server: carga `data/processed/` a base de datos relacional.

---

## Requisitos

- Python 3.10+
- `numpy`
- `torch` (si usas salida `.pt`, es el default)
- `pandas` + `openpyxl` (solo para `--injection-path`)
- `scikit-learn` (solo para `scripts/make_split.py`)
- `mysql-connector-python` (solo para ETL a MySQL)
- `pyodbc` (solo para ETL a SQL Server)

Instalar el paquete en modo editable (requerido para todos los scripts):

```bash
pip install -e .
```

---

## Estructura del proyecto

```
cmg2tensor/
├── src/
│   └── cmg2tensor/
│       ├── config.py             Constantes compartidas
│       ├── parse_txt.py          Parser CMG .txt → ParseResult (streaming)
│       ├── build_tensors.py      ParseResult → .pt / .npz
│       ├── discovery.py          Heurística de nombres de archivo
│       ├── normalize.py          Funciones min-max
│       ├── stats.py              Fase 1: scan global min/max
│       ├── histograms.py         Histogramas globales por capas
│       ├── orchestrator.py       Utilidades de batch serial
│       ├── cli.py                Parser de argumentos y main()
│       ├── pipeline/
│       │   ├── serial.py         Pipeline por simulación (SF/VD, variables, inyección)
│       │   └── parallel.py       Orquestador paralelo dos fases (ProcessPoolExecutor)
│       └── utils/
│           ├── raw_standardization.py   Renombra/estandariza data/raw
│           ├── simulations_dataset.py   Generación de split CSV
│           ├── apply_train_test_split.py Mueve carpetas según split
│           ├── check_formats.py         Verifica estructura de archivos
│           └── standardize_formats.py  CLI para estandarización
├── scripts/
│   ├── make_split.py             Split estratificado 90/10 y organización de carpetas
│   ├── normalize_train_test.py   Normalización post-procesado (train stats → train + test)
│   ├── check_missing_processed.py Reporte de simulaciones faltantes en processed/
│   ├── fix_perm_por_reports.py   Repara reportes de permeabilidad/porosidad existentes
│   ├── etl_mysql.py              ETL: data/processed → MySQL 8
│   ├── etl_sqlserver.py          ETL: data/processed → SQL Server
│   ├── populate_db.py            Pobla la BD SQLite local (demo académica)
│   └── query_db.py               Queries de ejemplo sobre la BD SQLite
├── sql/
│   ├── ddl_mysql.sql             DDL completo para MySQL 8
│   └── ddl_sqlserver.sql         DDL equivalente para SQL Server
├── docs/
│   └── database_schema.md        Documentación del esquema de BD y decisiones de diseño
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── test/
│   └── processed/
│       ├── train/
│       └── test/
├── tests/
├── notebooks/
└── reports/
```

---

## Estructura de datos

```
data/
├── raw/
│   ├── train/
│   │   ├── 001_normal/          ← inputs CMG (.txt, .xlsx)
│   │   ├── 002_high_to_low/
│   │   └── ...  (270 sims)
│   └── test/
│       ├── 076_inverse_normal/
│       └── ...  (30 sims)
└── processed/
    ├── train/
    │   ├── 001_normal/          ← tensores .pt + reportes JSON
    │   │   ├── layer_cubes/
    │   │   ├── layer_cubes_report.json
    │   │   └── timeline.json
    │   └── ...
    └── test/
        └── ...

reports/
├── batch_simulations_report.json        ← modo serial
├── batch_parallel_report.json           ← modo --parallel
├── train_test_split_80_20.csv           ← split estratificado (referenciado por --split-csv)
└── global_normalization/
    └── train.json                       ← stats globales (min/max por variable)
```

---

## Flujo de trabajo completo

### Paso 0 — Preparar el split (una sola vez)

Genera y organiza el split estratificado 90/10:

```bash
# Organizar data/raw en subcarpetas train/ y test/
python scripts/make_split.py --dir data/raw

# Aplicar el mismo split a data/processed (si ya hay simulaciones procesadas)
python scripts/make_split.py --dir data/processed
```

---

### Paso 1 — Calcular estadísticas globales (solo train)

Escanea todas las simulaciones `train` para obtener `min/max` por variable. Se guarda en `reports/global_normalization/train.json`.

```powershell
$env:PYTHONPATH="src"; python -m cmg2tensor `
  --all-simulations --raw-root data/raw `
  --nz 20 --nj 100 --ni 100 `
  --compute-global-normalization-only
```

> Solo es necesario ejecutar este paso una vez. Si el archivo `train.json` ya existe, el Paso 2 lo reutiliza automáticamente.

---

### Paso 2 — Transformar todas las simulaciones (paralelo recomendado)

```powershell
$env:PYTHONPATH="src"; python -m cmg2tensor `
  --all-simulations --raw-root data/raw --output-dir data/processed `
  --nz 20 --nj 100 --ni 100 `
  --global-normalization-stats reports/global_normalization/train.json `
  --parallel --n-workers 8
```

**Comportamiento:**
- `train/` → normalizado con stats globales de `train.json`
- `test/`  → guardado sin normalizar (stats de entrenamiento no se filtran al test)
- Cada simulación fallida se registra en el reporte sin abortar el batch

---

### Paso 2 alternativo — Batch serial (más lento, sin dependencias de spawn)

```powershell
$env:PYTHONPATH="src"; python -m cmg2tensor `
  --all-simulations --raw-root data/raw --output-dir data/processed `
  --nz 20 --nj 100 --ni 100 `
  --global-normalization-stats reports/global_normalization/train.json
```

---

### Reintentar simulaciones fallidas

```powershell
$env:PYTHONPATH="src"; python -m cmg2tensor `
  --all-simulations --raw-root data/raw --output-dir data/processed `
  --nz 20 --nj 100 --ni 100 `
  --parallel --n-workers 8 `
  --retry-failed reports/batch_parallel_report.json
```

---

### Una simulación (modo single)

```powershell
$env:PYTHONPATH="src"; python -m cmg2tensor `
  --sf-path data/raw/train/001_normal/SF.txt `
  --vd-path data/raw/train/001_normal/VD.txt `
  --permeability-path data/raw/train/001_normal/Permeability_CO2IA.txt `
  --porosity-path data/raw/train/001_normal/Porosity_CO2IA.txt `
  --cohesion-path data/raw/train/001_normal/Cohesion_CO2IA.txt `
  --afi-path data/raw/train/001_normal/AFI_CO2IA.txt `
  --pressure-path data/raw/train/001_normal/Pressure_CO2IA.txt `
  --nz 20 --nj 100 --ni 100 `
  --output-dir data/processed/train/001_normal
```

---

## Referencia de flags CLI

### Entrada — modo batch

| Flag | Descripción |
|---|---|
| `--all-simulations` | Procesar todas las subcarpetas de `--raw-root` |
| `--raw-root` | Raíz de simulaciones (default: `data/raw`). Soporta subdirs `train/`/`test/` automáticamente |
| `--split-csv` | CSV `[simulation_name, split]` para enrutar train/test explícitamente |

### Entrada — modo single

| Flag | Descripción |
|---|---|
| `--sf-path` / `--vd-path` | Archivos SF y VD |
| `--permeability-path` | Archivo de permeabilidad |
| `--porosity-path` | Archivo de porosidad |
| `--cohesion-path` | Archivo de cohesión |
| `--afi-path` | Archivo de AFI / friction angle |
| `--pressure-path` | Archivo de presión |
| `--gas-saturation-path` | Archivo de saturación de gas |
| `--injection-path` | Excel de inyección (`.xlsx`) |
| `--injection-sheet` | Hoja del Excel (default: `Well Summary`) |

### Grilla y salida

| Flag | Descripción |
|---|---|
| `--nz`, `--nj`, `--ni` | Dimensiones de la grilla (defaults: 20, 100, 100) |
| `--processed-dir` / `--output-dir` | Carpeta raíz de salida (default: `data/processed`) |
| `--normalize` (default) / `--no-normalize` | Normalización min-max |
| `--torch-output` (default) / `--npz-output` | Formato de salida |
| `--full-tensor-output` | Tensor completo `(V, T, Z, J, I)` en lugar de archivos por capa |

### Normalización global

| Flag | Descripción |
|---|---|
| `--normalization-scope` | `split` (default) o `simulation` |
| `--global-normalization-stats` | JSON con stats previos; omite el Paso 1 |
| `--compute-global-normalization-only` | Solo calcula stats y guarda `train.json`, luego sale |

### Paralelismo

| Flag | Descripción |
|---|---|
| `--parallel` | Activar `ProcessPoolExecutor` (requiere `--all-simulations`) |
| `--n-workers N` | Número de workers (default: `min(8, cpu_count)`) |
| `--retry-failed REPORT` | Reprocesar simulaciones fallidas de un reporte anterior |

### Control de flujo

| Flag | Descripción |
|---|---|
| `--skip-existing-outputs` | Saltar simulaciones con `layer_cubes_report.json` existente |
| `--skip-existing-pipelines` | Saltar pipelines individuales con reporte existente |
| `--skip-missing-required` | No abortar si falta SF/VD en batch |
| `--shared-permeability-path` | Fallback permeabilidad para simulaciones que no tienen la suya |
| `--shared-porosity-path` | Fallback porosidad |

---

## Arquitectura interna

```
src/cmg2tensor/
├── config.py             Constantes compartidas
│                           • DAYS_PER_MONTH, DEFAULT_INJECTION_PARAMETER, DEFAULT_INCLUDE_INJECTION_NAMES
│                           • DEFAULT_SPLIT_CSV, DEFAULT_BATCH_REPORT_PATH
│                           • DEFAULT_GLOBAL_NORMALIZATION_DIR, DEFAULT_GLOBAL_STATS_FILE
├── discovery.py          Descubrimiento de inputs por heurística de nombres de archivo
│                           • discover_simulation_inputs() — usado por el pipeline (lanza ValueError si falta SF/VD)
│                           • discover_raw_roles()         — usado por estandarización (sin raise)
├── parse_txt.py          Parser CMG → tensor 4D (T, NZ, NJ, NI)
│                           • Streaming 2-pass (sin cargar el archivo completo)
│                           • float32 directo, np.fromstring fast path
├── normalize.py          Funciones de normalización min-max
├── stats.py              Fase 1: scan O(1) via RESULTS PROP headers
│                           • Fallback a parse completo + del inmediato
├── histograms.py         Histogramas globales por capas
│                           • construir_histogramas_globales_por_capas() — acumula sin cargar todos los cubos
│                           • graficar_histograma_global() — visualización por variable
├── orchestrator.py       Utilidades de batch serial
│                           • _load_train_test_split_csv()
│                           • _compute_global_minmax_for_simulations()
│                           • _write_batch_report_incremental()
├── pipeline/
│   ├── serial.py         Pipelines por simulación
│   │                       • run_layer_cubes_pipeline()
│   │                       • run_single_variable_layer_cubes_pipeline()
│   │                       • run_injection_excel_pipeline()
│   │                       • _run_requested_pipelines() — orquesta todos los pipelines de una sim
│   └── parallel.py       Orquestador paralelo dos fases (ProcessPoolExecutor, spawn context)
│                           • run_batch_pipeline() — entry point público
│                           • WorkerResult, BatchReport — dataclasses de resultado
├── cli.py                Parser de argumentos y main()
└── build_tensors.py      Serialización .pt / .npz
```

### Flujo de datos interno

```
data/raw/train/<sim>/          data/raw/test/<sim>/
        │                               │
  discovery.py                    discovery.py
  (descubre SF, VD, vars)         (descubre SF, VD, vars)
        │                               │
  stats.py [Fase 1]                     │
  (scan min/max por variable)           │
        │                               │
  → reports/global_normalization/train.json
        │                               │
  pipeline/serial.py [Fase 2]    pipeline/serial.py [Fase 2]
  normalize=True                  normalize=False
  global_stats=train.json         global_stats=None
        │                               │
  data/processed/train/<sim>/    data/processed/test/<sim>/
```

---

## Rendimiento esperado (8 workers, grilla 20×100×100)

| Fase | Serial | Paralelo (8 workers) | RAM por worker |
|---|---|---|---|
| Fase 1 — stats scan | ~4 h | ≤ 30 min | ~0 MB (solo headers) |
| Fase 2 — transform | ~6 h | ≤ 45 min | ≤ 8 GB |

> El parser anterior cargaba **~3 GB por archivo a RAM**. El parser streaming actual usa **~120 MB por worker** para una grilla típica (20×100×100, T=15, float32).

---

## Salidas

### Modo por capas (default)

- `layer_cubes/layer_cube_kXXX.pt` para SF/VD.
- `<variable>_layer_cubes/layer_cube_kXXX.pt` para variables individuales.
- `injection_name_tensors/injection_tene_1.pt` y `injection_tene_2.pt` si hay Excel.

Cada archivo `.pt` contiene un dict:

```python
{
    "cube":          torch.Tensor,  # shape (V, T, NJ, NI), float32
    "time_ids":      list[int],     # meses desde t0
    "time_ids_days": list[int],     # días absolutos
    "time_unit":     "months",
    "variables":     list[str],
    "layer_k":       int,
    "normalization": dict,          # {"method": "global_minmax", "min": ..., "max": ...} o {"method": "none"}
}
```

### Modo full tensor (`--full-tensor-output`)

- `full_tensors/sf_vd_tensor.pt` — shape `(V, T, NZ, NJ, NI)`
- `full_tensors/permeability_tensor.pt`, etc.

---

## Reportes JSON

| Archivo | Contenido |
|---|---|
| `layer_cubes_report.json` | SF/VD: shape, time_ids, normalization, execution_seconds |
| `<var>_layer_cubes_report.json` | Variable individual: ídem |
| `injection_name_tensors_report.json` | Series de inyección: nombres, timesteps |
| `reports/batch_simulations_report.json` | Resumen batch serial |
| `reports/batch_parallel_report.json` | Resumen batch paralelo: phase1_seconds, phase2_seconds, por worker |
| `reports/global_normalization/train.json` | `{variable: {min, max, span}}` para todo el split train |

---

## Utilidades de preparación de datos

```bash
# Estandarizar nombres en data/raw (renombra a SF.txt, VD.txt, cohesion.txt, ...)
python src/cmg2tensor/utils/standardize_formats.py --raw-root data/raw/test --apply

# Verificar estructura de archivos en data/raw
python src/cmg2tensor/utils/check_formats.py --raw-root data/raw/test --out-file reports/raw_structure_report.txt

# Organizar/reorganizar split train/test
python scripts/make_split.py --dir data/raw
python scripts/make_split.py --dir data/processed

# Verificar simulaciones faltantes en processed/
python scripts/check_missing_processed.py --processed-dir data/processed

# Normalización post-procesado (train stats → train + test)
python scripts/normalize_train_test.py
python scripts/normalize_train_test.py --recompute-stats   # ignorar caché
python scripts/normalize_train_test.py --dry-run           # solo mostrar plan

# Reparar reportes de permeabilidad/porosidad
python scripts/fix_perm_por_reports.py
```

---

## Base de datos relacional

Los tensores procesados pueden cargarse a una base de datos relacional para consultas y análisis. Ver `docs/database_schema.md` para el esquema completo y decisiones de diseño.

### Esquema: 6 tablas

| Tabla | Descripción | Volumen |
|---|---|---|
| `tblModelo` | Catálogo de modelos geomecánicos | 1 fila |
| `tblCorrida` | Simulaciones con su tipo de muestreo y partición | 260 filas |
| `tblPropiedadesMalla` | Propiedades estáticas (Permeabilidad, Porosidad) por capa | 194 filas |
| `tblResultadoVariables_Header` | Encabezado de resultados por variable y timestep | ~188 K filas |
| `tblResultadoVariables_Detalle` | Grid 50×50 serializado como blob por capa y timestep | ~18.3 M filas |
| `tblTasaInyeccion` | Series de tasa de inyección por pozo | ~188 K filas |

### ETL a MySQL

```bash
# Aplicar esquema y cargar datos completos
python scripts/etl_mysql.py --user root --password TUPASS --apply-schema

# Solo prueba (2 corridas)
python scripts/etl_mysql.py --user root --password TUPASS --max-simulations 2
```

### ETL a SQL Server

```bash
# Windows Auth
python scripts/etl_sqlserver.py --server MISERVIDOR --database SimulacionesCO2 --trusted-connection

# Usuario/contraseña
python scripts/etl_sqlserver.py --server MISERVIDOR --database SimulacionesCO2 --user sa --password secret
```

### Archivos SQL

| Archivo | Descripción |
|---|---|
| `sql/ddl_mysql.sql` | DDL completo para MySQL 8 |
| `sql/ddl_sqlserver.sql` | DDL equivalente para SQL Server |

---

## Histogramas globales por capas

Para construir histogramas globales livianos sin recargar todos los cubos cada vez:

```python
from cmg2tensor import (
    construir_histogramas_globales_por_capas,
    graficar_histograma_global,
)

stats = construir_histogramas_globales_por_capas(
    dataset="data/processed/train",
    bins=128,
    output_path="reports/global_histograms/train_histograms.json",
)

graficar_histograma_global(
    "reports/global_histograms/train_histograms.json",
    variable="SF",
    log_y=True,
)
```

### Comportamiento

- Recorre artefactos `layer_cube_k*.pt/.npz` por capas y no almacena valores crudos.
- Acumula `counts`, `min`, `max`, `sum`, `sum_sq` y `n` por variable.
- Escribe un JSON liviano por variable con bins, frecuencias y estadísticos agregados.
- Si los reportes del pipeline contienen `min/max` por variable, el tensor se recorre una sola vez.
- Si el dataset fue generado sin metadata suficiente de rangos, hace un fallback a dos pasadas streaming para mantener histogramas exactos con bins consistentes.

### JSON de salida

```json
{
  "SF": {
    "bins": [0.0, 0.01, 0.02],
    "counts": [123, 456],
    "min": 0.0,
    "max": 1.0,
    "mean": 0.42,
    "std": 0.11,
    "n": 579,
    "p5": 0.08,
    "p50": 0.41,
    "p95": 0.63
  }
}
```

---

## Tests

```bash
# Tests rápidos (sin datos reales, ~1s)
$env:PYTHONPATH="src"; python -m pytest tests/ -m "not slow" -v

# Benchmark con archivo real
$env:PYTHONPATH="src"; python tests/benchmark_parse.py --path data/raw/train/001_normal/SF.txt --nz 20 --nj 100 --ni 100
```

Los tests `@pytest.mark.slow` requieren el pipeline paralelo completo (spawna procesos).

---

## Validaciones

- `SF` y `VD` deben tener el mismo timeline.
- Cada `TIME` debe tener exactamente `NZ * NJ` bloques `(K, J)`.
- Cada bloque `(K, J)` debe tener exactamente `NI` valores.
- No permite bloques duplicados ni índices fuera de rango.
- Inyección filtra nombres de forma exacta (`TENE-1`, `TENE-2`).
- Si el Excel trae múltiples métricas en la columna `Parameter`, el pipeline filtra automáticamente `"Gas Rate SC - Monthly (ft3/day)"`.
- Simulaciones `test` **nunca** se normalizan, incluso si `--normalize` está activo.

---

## Notebook de visualización

`notebooks/quick_view_processed_data.ipynb` — visualización por capas.
Opción `SOURCE_MODE='full'` para visualizar tensores completos.
