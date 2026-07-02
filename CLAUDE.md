# CLAUDE.md — Modelo-ITM (FNO CO₂ Geomecánico)

> Este archivo es la fuente de verdad para Claude Code en este proyecto.
> Léelo completo antes de ejecutar cualquier acción.

---

## Inicialización de sesión

Antes de cualquier tarea, Claude debe ejecutar estos pasos en orden:

1. Leer este archivo completo.
2. Listar los specs activos (`[IN PROGRESS]` o `[TESTING]`) en `specs/`.
3. Confirmar el repositorio activo y la rama actual con `git status`.
4. Si hay contexto previo relevante (spec en curso, decisión de arquitectura,
   deuda técnica pendiente), pedirlo al usuario antes de proceder.

---

## Reglas generales

- Toda la comunicación con el usuario debe ser en español.
- Antes de editar cualquier archivo, leer las secciones relevantes de su contenido.
  Para archivos de más de 300 líneas, navegar por secciones antes de editar;
  no asumir estructura sin haberla leído.
- No adivines rutas, imports ni nombres de variables: confírmalos leyendo el código.
- Si tienes dudas bloqueantes, usa `AskUserQuestion` antes de proceder.
- Nunca interrumpas una tarea a mitad para pedir confirmación, salvo que el
  riesgo de continuar sea alto (borrado de datos, corrupción de checkpoints, etc.).
- Prefiere cambios quirúrgicos sobre refactors amplios no solicitados.

---

## Agentes especializados

En `/.agents/` viven instrucciones para subagentes. Leer el archivo del agente
antes de invocarlo. No improvisar su comportamiento.

| Agente        | Cuándo invocarlo                                                              |
|---------------|-------------------------------------------------------------------------------|
| `@architect`  | Diseño de specs: fases, archivos impactados, sin código                       |
| `@reviewer`   | Revisión de código antes de marcar un spec como `[DONE]`                     |
| `@tester`     | Generación y ejecución de casos de prueba                                     |

> Si en `/.agents/` existen agentes adicionales específicos del proyecto,
> tienen precedencia sobre la tabla anterior.

---

## Contexto del proyecto

Modelo de deep learning para **predicción espacio-temporal del Factor de Seguridad (SF)
y la Deformación Volumétrica (VD)** en reservorios depletados bajo inyección de CO₂
(almacenamiento geológico de carbono, CCS).

El modelo aprende a mapear propiedades estáticas del reservorio (permeabilidad, porosidad,
cohesión, AFI, profundidad de capa) + tasas de inyección temporales (pozos TENE-1 y
TENE-2) → evolución temporal de SF y VD en una grilla 2D por capa.

**Estado actual:** en desarrollo activo. Arquitectura base implementada y entrenando.
Paper en redacción: *"A U-Net Approach for Safety Factor Prediction in Depleted
Reservoirs Using Synthetic Data"* (el título menciona U-Net; la implementación actual
usa FNO con condicionamiento FiLM).

**Usuarios objetivo:** investigadores de geomecánica y CCS. No hay UI; los resultados
se consumen como tensores y gráficas PNG de diagnóstico.

---

## Estructura del ecosistema

```
09-Proyecto-Deep-Learning/
├── 01-Modelo-ITM/                        # ← Repositorio activo (este CLAUDE.md)
│   └── (ver "Estructura de 01-Modelo-ITM" abajo — incluye el ETL en src/fno_co2/etl/)
├── cmg2tensor/                           # ⚠️ Referencia histórica — YA MIGRADO a
│                                          #    01-Modelo-ITM/src/fno_co2/etl/. No editar
│                                          #    ni ejecutar; se conserva solo como registro.
├── Paper Base/
│   └── spe-220850-pa_DL...pdf            # Paper de referencia (SPE)
├── Paper actualmente en desarrollo/
│   └── A U-Net Approach...docx           # Paper propio en redacción
├── Modelo Inicial IA.pptx                # Presentación de arquitectura inicial
└── Modelo de IA refinado.docx            # Documentación del modelo refinado
```

### Estructura de `01-Modelo-ITM/`

Paquete Python único en `src/fno_co2/` que contiene **todo el pipeline**: ETL (CMG
`.txt` → tensores `.pt`) → modelo → entrenamiento → inferencia → visualización, con
configs, tests, specs y outputs separados del código.

```
01-Modelo-ITM/
├── CLAUDE.md                     # Este archivo — fuente de verdad del proyecto
├── README.md                     # Resumen y estado del proyecto
├── pyproject.toml                # Paquete instalable en modo editable (pip install -e ".[dev,db]")
├── pytest.ini                    # Config de pytest (marker `slow`)
├── .gitignore
├── configs/
│   └── default.yaml              # Espejo del dataclass Config (referencia, aún sin loader)
├── src/fno_co2/                # Paquete Python único (nombre de import: fno_co2)
│   ├── etl/                      # Pipeline ETL: CMG .txt → tensores .pt (ex-cmg2tensor)
│   │   ├── pipeline/             # Orquestación serial/paralela por simulación
│   │   └── utils/                # Discovery, estandarización de nombres, split train/test
│   ├── data/                     # Dataset, carga de tensores .pt, series de inyección
│   ├── models/                   # PhysicalFNOArchitecture, ResBlock, FiLMSpectralBlock
│   ├── training/                 # Loop de entrenamiento, losses, métricas, checkpoints
│   ├── inference/                # Carga de checkpoints y predicción fuera del loop de training
│   ├── visualization/            # Curvas de entrenamiento, PNGs de diagnóstico por época
│   └── utils/                    # Helpers transversales (device, seeds, IO)
├── scripts/
│   ├── train.py                  # Entry point de entrenamiento
│   └── etl/                      # Scripts ETL (make_split, etl_mysql, normalize_train_test, …)
├── sql/                           # DDL MySQL y SQL Server (ex-cmg2tensor/sql/)
├── notebooks/                     # Notebooks exploratorios — no productivos
├── tests/
│   ├── unit/                     # Tests rápidos: dataset, losses, forward con tensores dummy
│   ├── etl/                      # Tests del ETL (regresión numérica, robustez, stress)
│   └── integration/              # Tests con datos reales o corridas cortas (marcar @pytest.mark.slow)
├── outputs/                      # Generado en runtime — ignorado por git salvo estructura
│   ├── checkpoints/              # latest.pt, best.pt
│   ├── logs/                     # metrics_history.json, config.json
│   └── figures/                  # training_curves.png, uncertainty_curves.png, visuals/
├── docs/                         # Documentación técnica (database_schema.md; arquitectura-y-correcciones-spec-000.md)
└── specs/                        # Specs de funcionalidades (ver "Specs de funcionalidades")
```

> **Estado:** Fase 2 (código de entrenamiento) y Fase A2 (ETL `cmg2tensor` → `src/fno_co2/etl/`)
> completadas. Todo el pipeline vive en el paquete único `fno_co2`. Listo para Parte B
> (correcciones científicas y del ETL).

---

## Stack tecnológico

- **Lenguaje:** Python 3.12 (fijado en `pyproject.toml`, `requires-python = ">=3.12,<3.13"`)
- **Framework DL:** PyTorch (tensores `.pt`, DataLoader, nn.Module)
- **Arquitectura:** `PhysicalFNOArchitecture` — FNO (Fourier Neural Operator) con
  bloques `FiLMSpectralBlock` (modulación espectral + condicionamiento FiLM) y
  codificador/decodificador convolucional con `ResBlock`
- **Pipeline de datos:** `fno_co2.etl` (integrado; ex-`cmg2tensor`) — parser streaming
  de archivos CMG `.txt` → tensores 4D `(V, T, NJ, NI)` por capa, normalizados globalmente
- **Librerías:** `numpy`, `pandas`, `openpyxl`, `scikit-learn`, `tqdm`,
  `matplotlib`, `mpl_toolkits`
- **Base de datos (opcional):** MySQL 8 / SQL Server vía `mysql-connector-python`
  / `pyodbc` (extra `[db]` del paquete)

### Comandos

```bash
# Instalar 01-Modelo-ITM en modo editable (paquete fno_co2, entorno venv con Python 3.12)
cd 01-Modelo-ITM
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"        # + ".[dev,db]" si vas a usar el ETL a MySQL/SQL Server

# Ejecutar entrenamiento (desde 01-Modelo-ITM con venv activado)
cd 01-Modelo-ITM
source .venv/bin/activate
python scripts/train.py \
  --data-root data/processed \
  --output-dir outputs/ \
  --device cuda \
  --epochs 100 \
  --batch-size 4

# Ejecutar el ETL (equivalente al antiguo "python -m cmg2tensor")
python -m fno_co2.etl --help

# Tests del proyecto completo (modelo + ETL, sin datos reales, ~2 s)
python -m pytest tests/ -m "not slow" -v

# Benchmark del parser con datos reales
python tests/etl/benchmark_parse.py \
  --path data/raw/train/001_normal/SF.txt --nz 20 --nj 100 --ni 100
```

---

## Dependencias

- Package manager: `pip` — no mezclar con conda en el mismo entorno.
- Antes de instalar cualquier dependencia nueva:
  1. Verificar si ya existe algo equivalente en el entorno activo.
  2. Mencionarlo al usuario con justificación clara.
  3. Esperar confirmación explícita.
- Preferir dependencias con mantenimiento activo y bajo footprint.
- Para entornos CUDA: asegurar que la versión de PyTorch coincida con el
  driver instalado (`torch.cuda.is_available()` debe retornar `True`).

---

## Variables de entorno

No hay archivo `.env` formal. Las rutas críticas se pasan como argumentos CLI
o se configuran en el dataclass `Config` en `src/fno_co2/config.py`.

Variables relevantes del entorno de ejecución:

| Variable / parámetro       | Descripción                                                  |
|----------------------------|--------------------------------------------------------------|
| `--data-root`              | Raíz de `data/processed/` con subdirs `train/` y `test/`   |
| `--output-dir`             | Carpeta de salida para checkpoints, JSONs y PNGs            |
| `--checkpoint-dir`         | Override de carpeta de checkpoints (default: `output/`)     |
| `--device`                 | `cuda`, `cpu` o `auto`                                       |

> ⚠️ Nunca escribas credenciales de base de datos (MySQL/SQL Server) en
> archivos rastreados por git. Pasarlas siempre por CLI (`--user`, `--password`).

---

## Base de datos (opcional — ETL relacional)

El ETL `fno_co2.etl` puede cargar tensores procesados a una BD relacional.
No es necesario para entrenar el modelo; es para análisis y consultas. Requiere el
extra `[db]` (`pip install -e ".[dev,db]"`).

- **Motor principal:** MySQL 8.0 (`SimulacionesCO2`)
- **Alternativa:** SQL Server (DDL equivalente en `sql/ddl_sqlserver.sql`)
- **6 tablas:** `tblModelo`, `tblCorrida`, `tblPropiedadesMalla`,
  `tblResultadoVariables_Header`, `tblResultadoVariables_Detalle`, `tblTasaInyeccion`
- **Volumen aprox:** 260 corridas × 362 timesteps × 97 capas → ~18 M filas en detalle

Nunca ejecutar ETL en producción sin confirmación explícita del usuario.

```bash
# ETL a MySQL (solo local o staging), desde 01-Modelo-ITM con venv activado
python scripts/etl/etl_mysql.py --user root --password TUPASS --apply-schema

# Prueba con 2 corridas
python scripts/etl/etl_mysql.py --user root --password TUPASS --max-simulations 2
```

---

## Arquitectura del modelo

### `PhysicalFNOArchitecture` (`src/fno_co2/models/fno.py`)

```
Entrada por paso temporal:
  x   : (B, 5, H, W)   — [SF_t, VD_t, Permeabilidad, Porosidad, Cohesión / AFI / Presión]
  d   : (B, 1)          — profundidad normalizada de la capa k
  inj : (B, T, 2)       — series de inyección TENE-1 y TENE-2 (T timesteps)

Encoder:
  Conv2d(5→h_dim) + GELU + ResBlock(h_dim, dropout_p)

Condicionamiento temporal (FiLM):
  t_embed : Embedding(T, cond_dim)        — embedding de paso temporal
  cond_mlp: Linear(3→cond_dim) × 2       — proyecta [inj_t1, inj_t2, depth] → cond_dim

FNO Blocks (× 4):
  FiLMSpectralBlock:
    FFT2 → multiplicación espectral (modos truncados) → iFFT2
    + Conv2d 1×1 local → GELU
    + modulación FiLM: γ(cond)·x + β(cond)

Decoder:
  ResBlock(h_dim, dropout_p) → Conv2d(h_dim→h_dim//2) → GELU → Conv2d(h_dim//2→2)

Salida:
  (B, T, 2, H, W)  — predicciones de SF y VD para todos los T pasos temporales
```

`ResBlock` inserta `nn.Dropout2d(dropout_p)` entre sus dos convoluciones (encoder y
decoder). Con `dropout_p > 0` (default `0.1`) esto habilita **MC Dropout real**: en
inferencia, `predict_with_uncertainty` fuerza esas capas a modo train durante N pasadas
(`cfg.uncertainty_passes`) y estima la desviación estándar de las predicciones como señal
de incertidumbre — antes del hallazgo C2 el modelo no tenía ninguna capa `Dropout`, por lo
que toda esta maquinaria devolvía 0.0 de incertidumbre / 1.0 de confianza siempre.

**Hiperparámetros default (`Config`):**

| Parámetro         | Valor   | Descripción                        |
|-------------------|---------|------------------------------------|
| `time_steps`      | 61      | Timesteps de predicción            |
| `hidden_dim`      | 128     | Canales ocultos                    |
| `spectral_modes`  | 16      | Modos de Fourier truncados         |
| `dropout_p`       | 0.1     | Probabilidad de `Dropout2d` en `ResBlock` (habilita MC Dropout real) |
| `use_group_norm`  | `False` | **EXPERIMENTAL** (M3): `GroupNorm` en `ResBlock`; cambia la arquitectura, no verificado en entrenamiento real completo |
| `deterministic`   | `False` | `True` prioriza reproducibilidad sobre rendimiento en CUDA (`cudnn.deterministic=True`) |
| `batch_size`      | 4       | Tamaño de batch                    |
| `lr`              | 8e-4    | Learning rate inicial (AdamW)      |
| `lr_scheduler`    | `"cosine"` | Scheduler de LR (`CosineAnnealingLR`, `T_max=epochs`); `None` para LR constante |
| `lr_min`          | 1e-6    | LR mínimo al final del ciclo coseno (`eta_min`) |
| `weight_decay`    | 1e-4    | Regularización AdamW — no se aplica a bias/embeddings/gamma-beta de FiLM (param groups) |
| `use_amp`         | `False` | Mixed precision (AMP); solo tiene efecto con `device="cuda"` |
| `grad_clip`       | 1.0     | Clip de gradiente (norma L2)       |
| `sf_weight`       | 2.5     | Peso de SF en la loss total        |
| `vd_weight`       | 1.0     | Peso de VD en la loss total        |
| `grad_weight`     | 0.8     | Peso de pérdida de gradiente espacial |
| `uncertainty_passes` | 30   | Pasadas MC Dropout para incertidumbre |

### Función de pérdida

```
loss = sf_weight × L_SF + vd_weight × L_VD

L_SF = seg_t0_weight   × SmoothL1(SF_t=0)   + grad_loss(SF_t=0)
     + seg_t1_20_weight × SmoothL1(SF_t1:20) + grad_loss(SF_t1:20)
     + seg_t21_60_weight × SmoothL1(SF_t21:60) + grad_loss(SF_t21:60)

L_VD = SmoothL1(VD) + grad_weight × grad_loss(VD)
```

### Dataset (`DatasetLayers`)

- Lee tensores `.pt` por capa (formato: `layer_cube_kXXX.pt`)
- Cada ítem: `(x, depth, injection, y)` donde `x` son propiedades estáticas
  y `y` es la evolución temporal de SF+VD
- `train/` **y** `test/` se normalizan con las **mismas estadísticas globales de train**
  (min-max en `[0,1]`), de modo que train y validación quedan en la misma escala. No hay
  fuga de datos: las stats se calculan solo con `train/` (corrección C1 del `spec-000`;
  antes de C1 `test/` se guardaba sin normalizar — ver `docs/arquitectura-y-correcciones-spec-000.md`)

---

## Pipeline de datos (`fno_co2.etl`)

### Flujo completo

```
data/raw/{train,test}/<sim>/
  SF.txt, VD.txt, Permeability_CO2IA.txt, Porosity_CO2IA.txt,
  Cohesion_CO2IA.txt, AFI_CO2IA.txt, Pressure_CO2IA.txt, inyeccion.xlsx
        │
  [Paso 0] make_split.py   → organiza en train/ y test/ (90/10 estratificado)
        │
  [Paso 1] --compute-global-normalization-only
           → reports/global_normalization/train.json
        │
  [Paso 2] --parallel --n-workers 8
           train/ → normalizado con train.json
           test/  → normalizado con train.json (misma escala que train; C1)
        │
data/processed/{train,test}/<sim>/
  layer_cubes/layer_cube_kXXX.pt   ← consumido por DatasetLayers
  layer_cubes_report.json
  timeline.json
```

### Dimensiones de la grilla

| Dimensión | Valor | Descripción           |
|-----------|-------|-----------------------|
| NZ        | 20    | Capas verticales      |
| NJ        | 100   | Filas de malla        |
| NI        | 100   | Columnas de malla     |
| T         | variable | Timesteps (hasta 61) |

### Comandos ETL

Desde `01-Modelo-ITM/` con el venv activado (no requiere `PYTHONPATH`; el paquete está
instalado en modo editable):

```bash
# Paso 1 — Estadísticas globales (una sola vez)
python -m fno_co2.etl \
  --all-simulations --raw-root data/raw \
  --nz 20 --nj 100 --ni 100 \
  --compute-global-normalization-only

# Paso 2 — Transformar todo (paralelo)
python -m fno_co2.etl \
  --all-simulations --raw-root data/raw --output-dir data/processed \
  --nz 20 --nj 100 --ni 100 \
  --global-normalization-stats reports/global_normalization/train.json \
  --parallel --n-workers 8

# Reintentar fallidas
python -m fno_co2.etl \
  --all-simulations --raw-root data/raw --output-dir data/processed \
  --nz 20 --nj 100 --ni 100 \
  --parallel --n-workers 8 \
  --retry-failed reports/batch_parallel_report.json

# Split train/test estratificado
python scripts/etl/make_split.py --dir data/raw
```

---

## Convenciones de código

- Lenguaje: **Python 3.12** con type hints donde sea práctico.
- Nombres de archivos: `snake_case`.
- Clases: `PascalCase`. Funciones y variables: `snake_case`.
- Configuración del modelo: centralizada en el dataclass `Config`
  (`src/fno_co2/config.py`); no hardcodear hiperparámetros en funciones internas.
- Paquete Python de este repo: `fno_co2` (import name), instalado en modo
  editable desde `01-Modelo-ITM/` vía `pyproject.toml`.
- Comentarios de sección con `# ==========================================`.
- Mensajes de log y errores en español (consistente con el resto del proyecto).
- No usar `print` para diagnóstico permanente; usar `fno_co2.utils.get_logger(__name__)`
  (`logging` estándar, con un handler que enruta a `tqdm.write()` para no romper las
  barras de progreso activas). `training/loop.py` e `inference/uncertainty.py` ya siguen
  este patrón; replicarlo en módulos nuevos.
- Los tensores siempre en `float32`; no mezclar dtypes sin justificación.

---

## Testing

- Framework: `pytest`
- Ubicación: `01-Modelo-ITM/tests/`, con subcarpetas `unit/` (modelo), `etl/` (pipeline
  ETL, ex-`cmg2tensor/tests/`) e `integration/`
- Convención: `test_*.py`
- Tests lentos (requieren pipeline completo, datos reales o una corrida de
  entrenamiento): marcados con `@pytest.mark.slow`; excluir con `-m "not slow"` en el CI
- Antes de cerrar una tarea con lógica crítica de parsing, normalización, dataset,
  loss o forward del modelo, verificar que existe al menos un test que cubra el caso feliz

```bash
# Tests de todo el proyecto (modelo + ETL, sin datos reales)
cd 01-Modelo-ITM && python -m pytest tests/ -m "not slow" -v

# Solo tests del modelo
cd 01-Modelo-ITM && python -m pytest tests/unit -v

# Solo tests del ETL
cd 01-Modelo-ITM && python -m pytest tests/etl -m "not slow" -v

# Benchmark real del parser
cd 01-Modelo-ITM && python tests/etl/benchmark_parse.py \
  --path data/raw/train/001_normal/SF.txt --nz 20 --nj 100 --ni 100
```

---

## Specs de funcionalidades

### Ubicación y nomenclatura

- Carpeta: `specs/` en `09-Proyecto-Deep-Learning/` (raíz del ecosistema).
- Nomenclatura: `spec-NNN-slug-descriptivo.md`

### Estados válidos

| Estado          | Significado                                              |
|-----------------|----------------------------------------------------------|
| `[IN PROGRESS]` | Implementación iniciada                                  |
| `[TESTING]`     | Implementación completa, pendiente de pruebas            |
| `[DONE]`        | Pruebas superadas                                        |

Los specs completados **no se borran**; se marcan con `[DONE]` en el título.

---

## Nuevas funcionalidades

### Antes de implementar

1. Analizar el impacto en el pipeline de datos, la arquitectura del modelo
   y los scripts de entrenamiento.
2. Usar el subagente `@architect` para crear el plan de implementación
   (solo fases y archivos; sin código).
3. Guardar el plan en `specs/` y esperar aprobación antes de escribir código.
4. Crear una rama nueva desde `development`.

### Durante la implementación

- Trabajar fase por fase según el spec; no saltarse pasos.
- Al iniciar la Fase 1, cambiar el estado del spec a `[IN PROGRESS]`.
- Si el scope debe cambiar, proponer la modificación al usuario antes de
  proceder. No editar el spec unilateralmente.
- Deuda técnica fuera del scope: documentar con `# DEBT:` en el código y
  registrar en `specs/backlog.md`.

---

## Despliegue / Ejecución en producción

> ⚠️ No existe despliegue web. El "despliegue" es la ejecución del
> entrenamiento en un servidor con GPU. Ningún paso debe ejecutarse sin
> confirmación explícita del usuario.

### Checklist pre-ejecución en servidor

- [ ] Checkpoints anteriores respaldados en ubicación segura.
- [ ] `train.json` de normalización global presente y validado.
- [ ] Datos procesados en `data/processed/{train,test}/` completos
      (verificar con `check_missing_processed.py`).
- [ ] Versión de PyTorch y CUDA compatibles (`torch.cuda.is_available() == True`).
- [ ] `Config` revisada: `epochs`, `batch_size`, `lr`, `output_dir`.
- [ ] `--auto-resume` activo si se retoma un entrenamiento interrumpido.

### Acciones prohibidas en ejecución

- Borrar checkpoints sin confirmación explícita.
- Sobrescribir `train.json` de normalización global sin regenerarlo completo.
- Modificar `Config` durante un entrenamiento en curso sin reiniciar.
- Ejecutar ETL a BD en producción sin confirmación.

---

## Acciones prohibidas

> Claude nunca debe realizar las siguientes acciones sin confirmación
> explícita del usuario en la misma sesión:

- Borrar archivos o carpetas (salvo temporales de la propia tarea).
- Ejecutar scripts ETL de base de datos en entornos distintos al local.
- Hacer push a `main` o `development` directamente.
- Instalar dependencias nuevas sin mencionarlo y esperar confirmación.
- Eliminar o sobrescribir checkpoints de entrenamiento.
- Reescribir `reports/global_normalization/train.json` sin regenerarlo
  completamente desde los datos raw.
- Editar el spec activo para ampliar su scope sin aprobación del usuario.

---

## Git — Branching & Commits

### Estructura de ramas

| Propósito                         | Prefijo     | Ejemplo                          |
|-----------------------------------|-------------|----------------------------------|
| Nueva funcionalidad o spec        | `feature/`  | `feature/fno-uncertainty`        |
| Corrección de bug                 | `bug/`      | `bug/parser-bloque-duplicado`    |
| Experimento de arquitectura       | `exp/`      | `exp/unet-skip-connections`      |

- `main` — estado estable; solo recibe merges revisados.
- `development` — integración; todas las ramas `feature/`, `bug/` y `exp/`
  se desprenden de aquí.
- Al mergear una rama a `development`, eliminarla inmediatamente.

### Commits

Mensajes **completamente en inglés**, siguiendo
[Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>
```

Tipos válidos: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `perf`.

Ejemplos:
```
feat(model): add MC Dropout uncertainty estimation
fix(parser): handle duplicate TIME blocks in CMG output
perf(pipeline): switch to streaming parser to reduce peak RAM
docs(schema): document tblResultadoVariables_Detalle blob format
chore(deps): pin torch to 2.3.0 for CUDA 12.1 compatibility
```
