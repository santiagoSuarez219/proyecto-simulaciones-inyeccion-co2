# spec-000 — Migración a 01-Modelo-ITM + correcciones de entrenamiento y pipeline [TESTING]

> **Estado (2026-07-03):** todas las fases (A, A2, 6–10) completadas en código, con 108/108
> tests pasando (incluidos `slow`) y smoke tests end-to-end reales. Se marca `[TESTING]`
> en vez de `[DONE]` porque 3 verificaciones requieren hardware/datos que no estuvieron
> disponibles en esta sesión: (1) **C1** — regenerar `data/processed/test/` con datos
> reales y confirmar `val_loss` en la misma escala que `train_loss` en una corrida real;
> (2) **M2** — el path real de `float16`/`GradScaler` en CUDA nunca se ejerció (solo
> verificado que es un no-op seguro en CPU); (3) **M3** — `use_group_norm` no se probó en
> un entrenamiento real completo (experimental, desactivado por defecto). Pasar a `[DONE]`
> requiere ejecutar estas 3 verificaciones con GPU y datos reales.

> **Autor:** revisión de código (rol `@architect`)
> **Fecha:** 2026-07-02
> **Resultado de:** unificación de los antiguos `spec-000` (migración/entorno/repo) y
> `spec-001` (correcciones DL) en un solo documento.
> **Objetivo global:** llevar **todo el pipeline** (código de entrenamiento **y** ETL
> `cmg2tensor`) a la estructura de paquete de `01-Modelo-ITM/`, con Python 3.12, entorno
> virtual y repositorio git/GitHub propio, y **luego** corregir sobre esa estructura los
> errores de correctitud científica, los bugs de acoplamiento y las omisiones de buenas
> prácticas de Deep Learning detectados en la revisión — sin cambiar la arquitectura del
> modelo (`PhysicalFNOArchitecture`) salvo donde se indique explícitamente.

> **✏️ Enmienda de alcance (2026-07-02, confirmada con el usuario):** el ETL `cmg2tensor`
> **ya no queda como repositorio independiente**. Se migra dentro de `01-Modelo-ITM/` como
> subpaquete unificado `src/modelo_itm/etl/` (imports `cmg2tensor.*` → `modelo_itm.etl.*`),
> mediante **copia limpia sin historial** (el repo de Reinaldo-06 se conserva como
> referencia). Con esto el pipeline completo (ETL → modelo → entrenamiento → inferencia →
> visualización) vive en un solo paquete. Ver **PARTE A2**.

**El spec tiene tres partes secuenciales:**
- **Parte A — Migración e infraestructura** (Fases 1–5): mueve el código de entrenamiento
  sin cambiar su comportamiento y monta entorno + repo.
- **Parte A2 — Migración de `cmg2tensor`** (Fases A2.1–A2.4): pliega el ETL en
  `src/modelo_itm/etl/` sin cambiar su comportamiento.
- **Parte B — Correcciones sobre la nueva estructura** (Fases 6–10): aplica los arreglos,
  incluidas las correcciones del ETL (C1, B3) sobre su **nueva** ubicación.

⚠️ **Las Partes A y A2 deben completarse antes de la Parte B.** Las correcciones referencian
los módulos de `src/modelo_itm/` que la migración crea. Las referencias `train_dataset.py:NNN`
y `cmg2tensor/src/cmg2tensor/…` que aparecen en los hallazgos son los anclajes originales
(traducidos al módulo destino según el "Mapa de módulos" de la Fase 2 y de la Fase A2.1).

---

## 0. Contexto y decisiones ya tomadas

El modelo predice la evolución espacio-temporal del **Factor de Seguridad (SF)** y la
**Deformación Volumétrica (VD)** por capa. El pipeline `cmg2tensor` transforma salidas
CMG `.txt` → tensores `.pt` y el script de entrenamiento consume esos tensores.

Decisiones confirmadas con el usuario en sesiones previas:

| Decisión | Valor |
|---|---|
| Alcance de la migración | El **código de entrenamiento** (`Codigo Entrenamiento/train_dataset.py`) → `01-Modelo-ITM/src/modelo_itm/` **y** el ETL `cmg2tensor` → `01-Modelo-ITM/src/modelo_itm/etl/`. Papers, `.pptx` y `.docx` **no** se mueven. |
| Relación con `cmg2tensor` (**enmendada 2026-07-02**) | **Se integra dentro de `01-Modelo-ITM`** como subpaquete unificado `src/modelo_itm/etl/`. Antes era "repo independiente referenciado por ruta"; ahora el pipeline completo vive en un solo paquete. El repo original de Reinaldo-06 se conserva **solo como referencia** (no se borra, no se sincroniza). |
| Estructura del ETL migrado | **Paquete unificado** (no monorepo de 2 paquetes): `cmg2tensor` se pliega como `modelo_itm.etl`; todos los imports `cmg2tensor.*` pasan a `modelo_itm.etl.*` y `python -m cmg2tensor` pasa a `python -m modelo_itm.etl`. |
| Historial git del ETL | **Copia limpia sin historial** (ni `git subtree` ni `filter-repo`). Los commits quedan en el repo original de Reinaldo-06. |
| Versión de Python | **3.12** (`requires-python = ">=3.12,<3.13"`) — soportado por PyTorch 2.12.x; ecosistema (`numpy`, `pandas`, `scikit-learn`, `mysql-connector-python`, `pyodbc`) maduro; se evita 3.13/3.14 (demasiado nuevas para dependencias de BD) y 3.9/3.10 (innecesariamente antiguas). |
| Gestor de entornos | `pip` + `venv` — **no conda**, ya establecido en `CLAUDE.md`. |
| Estructura destino | Ya creada como scaffold vacío: `src/modelo_itm/{data,models,training,inference,visualization,utils}`, `scripts/`, `tests/{unit,integration}`, `configs/`, `outputs/{checkpoints,logs,figures}`, `docs/`, `specs/`, `pyproject.toml`, `pytest.ini`, `.gitignore`, `README.md`. La Parte A2 añade `src/modelo_itm/etl/`, `scripts/etl/`, `sql/` y tests del ETL. |

**Estado verificado del ecosistema (previo a este spec):**
- `09-Proyecto-Deep-Learning/` (raíz) **no** es repositorio git.
- `01-Modelo-ITM/` **no** es repositorio git (aún sin `git init`).
- `cmg2tensor/` **sí** es repositorio git independiente, remoto
  `https://github.com/Reinaldo-06/cmg2tensor.git`. Contiene **solo código** (`src/cmg2tensor/`
  con 20 módulos, `scripts/` con 9 scripts ETL, `sql/`, `tests/`, `docs/`, `notebooks/`).
  Los **datos NO están versionados** (`data/raw/*`, `data/processed/*` y `*.pt` en su
  `.gitignore`; `data/` ni existe localmente). Migrar = mover **código**, no datos.
- `Codigo Entrenamiento/` contiene únicamente `train_dataset.py` (sin checkpoints ni
  outputs que respaldar — bajo riesgo de pérdida de datos al migrar).
- Python en el sistema: 3.9.6 (`/usr/bin/python3`), 3.13.11 (conda `base`), 3.14.6
  (Homebrew). **Ninguno es 3.12** — debe instalarse.
- `gh` (GitHub CLI) **no está instalado**.
- Inconsistencia previa del `README.md` de `cmg2tensor` (`pip install -e .` sin
  `pyproject.toml`/`setup.py`): **se resuelve** con la Parte A2, al quedar el ETL dentro del
  paquete instalable `modelo_itm` (que sí tiene `pyproject.toml`).

---

# PARTE A — Migración e infraestructura

> Migración **quirúrgica**: mover código y ajustar imports **sin** aplicar todavía las
> correcciones de la Parte B. El comportamiento tras la Parte A debe ser *idéntico* al
> actual, solo reorganizado. Los pasos marcados **⚠️ requiere confirmación explícita** no
> se ejecutan solos aunque el spec esté aprobado (borrado, push, creación de recursos
> externos), según §Acciones prohibidas de `CLAUDE.md`.

### Fase 1 — Entorno: Python 3.12 + venv

1. Instalar Python 3.12 vía Homebrew:
   ```bash
   brew install python@3.12
   ```
2. Crear el entorno virtual dentro de `01-Modelo-ITM/`:
   ```bash
   cd 01-Modelo-ITM
   /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
   source .venv/bin/activate
   python --version   # debe imprimir 3.12.x
   ```
3. Fijar la versión en `pyproject.toml`: `requires-python = ">=3.12,<3.13"`.
4. Actualizar `pip` e instalar en modo editable con extras de desarrollo:
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```
5. Añadir `.venv/` a `.gitignore` (verificar que no esté ya cubierto).
6. Actualizar `CLAUDE.md`: §Stack tecnológico → "Python 3.12 (fijado en `pyproject.toml`)";
   §Dependencias → documentar el flujo `venv` + `pip install -e ".[dev]"` como reemplazo
   del `pip install torch numpy ...` suelto.

**Verificación:** `python --version` en el venv reporta 3.12.x; `pip show modelo-itm`
confirma instalación editable; `pip list` no mezcla paquetes de conda `base`.

---

### Fase 2 — Migración de código: `train_dataset.py` → `src/modelo_itm/`

**Mapa de módulos** (origen: `Codigo Entrenamiento/train_dataset.py`; se usa como
traductor de rutas para los hallazgos de la Parte B):

| Módulo destino | Contenido movido |
|---|---|
| `src/modelo_itm/config.py` | `Config` (dataclass), `CFG`, `DEFAULT_DEVICE` |
| `src/modelo_itm/utils/device.py` | `seed_everything`, `resolve_device`, `describe_device`, `assert_model_on_device` |
| `src/modelo_itm/utils/io.py` | `ensure_dir`, `save_json`, `load_json` |
| `src/modelo_itm/utils/time.py` | `get_next_pause_datetime` |
| `src/modelo_itm/data/dataset.py` | `_LAYER_RE`, `_k`, `load_pt`, `load_injection_series`, `DatasetLayers` |
| `src/modelo_itm/data/loaders.py` | `resolve_num_workers`, `build_loader`, `resolve_dir`, `build_datasets` |
| `src/modelo_itm/models/blocks.py` | `ResBlock`, `FiLMSpectralBlock` |
| `src/modelo_itm/models/fno.py` | `PhysicalFNOArchitecture` |
| `src/modelo_itm/training/losses.py` | `spatial_gradient_loss`, `compute_loss_terms` |
| `src/modelo_itm/training/metrics.py` | `torch_r2_score`, `compute_rmse`, `compute_all_metrics`, `init_running_stats`, `update_running_stats`, `finalize_running_stats`, `count_parameters` |
| `src/modelo_itm/training/checkpoint.py` | `build_run_signature`, `check_resume_compatibility`, `save_training_checkpoint`, `try_resume_training` |
| `src/modelo_itm/training/loop.py` | `run_one_epoch`, `evaluate_epoch`, `main()` |
| `src/modelo_itm/inference/uncertainty.py` | `model_has_dropout`, `default_uncertainty_calibration`, `predict_with_uncertainty`, `calibrate_uncertainty`, `build_uncertainty_map`, `summarize_uncertainty`, `load_or_create_uncertainty_calibration` |
| `src/modelo_itm/visualization/plots.py` | `save_history_plots`, `save_epoch_visuals` |
| `scripts/train.py` | `str_to_bool`, `build_parser`, bloque `if __name__ == "__main__":` — entrypoint delgado que importa de `modelo_itm` y llama a `training.loop.main()` |

**Pasos:**
1. Crear los archivos según la tabla dentro del scaffold existente; agregar re-exports en
   los `__init__.py` de cada subpaquete.
2. Mover el código función por función, ajustando imports relativos
   (`from modelo_itm.config import Config`, etc.) y quitando imports monolíticos donde no
   se usen.
3. Reubicar el estado global mutable de "emitir una vez"
   (`_MC_DROPOUT_WARNING_EMITTED`, `_CUDA_BATCH_REPORT_EMITTED`) a variables de módulo en
   `inference/uncertainty.py` y `training/loop.py` respectivamente (mismo comportamiento;
   la limpieza real es el hallazgo B1 de la Parte B).
4. `scripts/train.py` queda como único entrypoint CLI:
   ```bash
   python scripts/train.py --data-root ../cmg2tensor/data/processed --output-dir outputs/ ...
   ```
5. Verificación de equivalencia (mínima, sin GPU en esta sesión):
   - `python -c "import modelo_itm"` no falla.
   - `python scripts/train.py --help` expone los mismos flags que el script original.
   - Si hay muestra de datos, correr `--overfit-sample-idx 0 --epochs 1` y comparar
     métricas contra el `train_dataset.py` original (antes de eliminarlo en la Fase 5).

**Fase 2b — Tests mínimos de la migración** (blindan la migración; **no** son la cobertura
completa del hallazgo M8):
- `tests/unit/test_dataset.py`: `DatasetLayers` con fixture sintético verifica shapes de
  `(x, depth, inj, y)`.
- `tests/unit/test_losses.py`: `compute_loss_terms` con tensores dummy devuelve escalares
  finitos.
- `tests/unit/test_model_forward.py`: `PhysicalFNOArchitecture` con `time_steps` chico
  (p. ej. 4) hace un forward y produce el shape `(B, T, 2, H, W)`.

**Verificación:** `pytest tests/unit -v` pasa; `scripts/train.py --help` igual al original;
ningún módulo bajo `src/modelo_itm/` supera ~300 líneas sin necesidad.

---

### Fase 3 — Repositorio git local de `01-Modelo-ITM`

1. Confirmar que no hay `.git` ascendente que anide repos:
   `git rev-parse --is-inside-work-tree 2>&1` debe fallar.
2. Revisar `.gitignore`: `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`,
   `outputs/{checkpoints,logs,figures}/*` (con excepciones `.gitkeep`).
3. **⚠️ requiere confirmación explícita** (crear historial git es una acción de estado que
   el usuario debe autorizar en la sesión):
   ```bash
   git init -b main
   git add <archivos específicos, no -A>
   git commit -m "chore: scaffold del paquete modelo_itm y migración de train_dataset.py"
   git branch development
   ```
4. Estructura de ramas según `CLAUDE.md` §Git: `main` (estable) y `development`
   (integración), con `feature/`, `bug/`, `exp/` desde `development`.

**Verificación:** `git log --oneline` muestra el commit inicial; `git status` limpio;
`git branch` lista `main` y `development`.

---

### Fase 4 — Repositorio remoto en GitHub

> **⚠️ Toda esta fase requiere confirmación explícita del usuario antes de ejecutar
> cualquier comando.** Crear un repo y hacer push son acciones visibles para terceros y
> difíciles de revertir (§Executing actions with care).

1. Confirmar en el momento de ejecutar (no asumir ahora): cuenta/organización destino
   (¿`Reinaldo-06`, la personal del usuario, otra?), visibilidad (privado/público) y
   nombre exacto (sugerido `modelo-itm`; GitHub no admite espacios).
2. Instalar `gh` si se crea desde terminal (no está instalado): `brew install gh` +
   `gh auth login`. Alternativa: crear el repo vacío desde la web (sin README/licencia/
   gitignore automáticos, para no chocar con los archivos locales).
3. Conectar y publicar:
   ```bash
   git remote add origin <URL confirmada por el usuario>
   git push -u origin main
   git push -u origin development
   ```

**Verificación:** `git remote -v` correcto; ambas ramas publicadas; el repo en GitHub
refleja la estructura local.

---

### Fase 5 — Limpieza y documentación de la migración

1. **⚠️ requiere confirmación explícita** — borrar el original ya migrado:
   ```bash
   rm "../Codigo Entrenamiento/train_dataset.py"
   rmdir "../Codigo Entrenamiento"     # solo si queda vacía
   ```
2. Actualizar `CLAUDE.md`: §Estructura → quitar `Codigo Entrenamiento/`; §Comandos →
   reemplazar la invocación por
   `cd 01-Modelo-ITM && source .venv/bin/activate && python scripts/train.py --data-root ../cmg2tensor/data/processed --output-dir outputs/ ...`;
   §Git → agregar la URL real del remoto; quitar el marcador "Estado de la migración".
3. Actualizar `README.md`: quitar la nota "estructura en migración"; documentar activación
   del entorno y ejecución del entrenamiento.

**Verificación:** no quedan referencias rotas a `Codigo Entrenamiento/` en `CLAUDE.md`,
`README.md` ni `specs/`; `git status` limpio tras comitear (commit confirmado por el
usuario).

---

# PARTE A2 — Migración de `cmg2tensor` a `src/modelo_itm/etl/`

> Migración **quirúrgica** del ETL, con la misma filosofía que la Parte A: mover código y
> reescribir imports **sin** cambiar comportamiento. Las correcciones del ETL (C1, B3) se
> aplican **después**, en la Parte B, sobre la nueva ubicación. Cada fase de esta parte va
> en su propia rama `feature/etl-migration` (o similar) desde `development`.
>
> **Copia limpia, sin historial:** se copian los archivos del repo de Reinaldo-06 a
> `01-Modelo-ITM/`; **no** se borra ni modifica el repo original (queda como referencia,
> igual que `Codigo Entrenamiento/`). Solo se migra **código** (los datos no están
> versionados y no se tocan).

### Fase A2.1 — Mapa de módulos del ETL y reescritura de imports

**Mapa de módulos** (origen: `cmg2tensor/src/cmg2tensor/`; destino: `src/modelo_itm/etl/`).
El plegado es 1:1 conservando la estructura interna:

| Origen (`cmg2tensor/src/cmg2tensor/`) | Destino (`src/modelo_itm/etl/`) |
|---|---|
| `__init__.py`, `__main__.py` | `etl/__init__.py`, `etl/__main__.py` (habilita `python -m modelo_itm.etl`) |
| `cli.py`, `config.py`, `orchestrator.py` | `etl/cli.py`, `etl/config.py`, `etl/orchestrator.py` |
| `discovery.py`, `parse_txt.py`, `build_tensors.py` | `etl/discovery.py`, `etl/parse_txt.py`, `etl/build_tensors.py` |
| `normalize.py`, `stats.py`, `histograms.py` | `etl/normalize.py`, `etl/stats.py`, `etl/histograms.py` |
| `pipeline/{__init__,parallel,serial}.py` | `etl/pipeline/{__init__,parallel,serial}.py` |
| `utils/{__init__,apply_train_test_split,check_formats,raw_standardization,simulations_dataset,standardize_formats}.py` | `etl/utils/…` (misma estructura) |

**Pasos:**
1. Copiar los 20 módulos a `src/modelo_itm/etl/` respetando la jerarquía `pipeline/` y
   `utils/`.
2. **Reescritura mecánica de imports** en todo el código copiado:
   - `import cmg2tensor` → `import modelo_itm.etl`
   - `from cmg2tensor.X import Y` → `from modelo_itm.etl.X import Y`
   - `from cmg2tensor.pipeline.Z import …` → `from modelo_itm.etl.pipeline.Z import …`
   - Imports relativos internos (`from .parse_txt import …`) se conservan tal cual.
3. Verificar que no queda ninguna referencia literal a `cmg2tensor` en el código migrado
   (`grep -rn "cmg2tensor" src/modelo_itm/etl/` debe salir vacío salvo comentarios/paths de
   datos deliberados).
4. `etl/config.py` y `etl/utils/` **no** colisionan con `modelo_itm/config.py` ni
   `modelo_itm/utils/` (viven bajo el namespace `etl`). No fusionar ambos `config`.

> ⚠️ El estado interno del ETL (rutas de datos por defecto, nombres de reportes) se conserva
> **idéntico**; cualquier ajuste de rutas de datos es un detalle de runtime (`--raw-root`,
> `--output-dir`), no de la migración.

### Fase A2.2 — Scripts, SQL, tests, docs y notebooks del ETL

| Origen | Destino | Nota |
|---|---|---|
| `cmg2tensor/scripts/*.py` (9 scripts) | `scripts/etl/` | Reescribir imports a `modelo_itm.etl.*`; mantener nombres (`make_split.py`, `etl_mysql.py`, `check_missing_processed.py`, …) |
| `cmg2tensor/scripts/*.ps1` | `scripts/etl/` | Actualizar invocaciones `python -m cmg2tensor` → `python -m modelo_itm.etl` |
| `cmg2tensor/sql/` (DDL MySQL/SQL Server) | `sql/` (raíz de `01-Modelo-ITM`) | Sin cambios de contenido |
| `cmg2tensor/tests/` | `tests/etl/` (con `unit/` e `integration/` según corresponda) | Reescribir imports; marcar los lentos con `@pytest.mark.slow` |
| `cmg2tensor/docs/database_schema.md` | `docs/` | Sin cambios de contenido |
| `cmg2tensor/notebooks/` | `notebooks/` | Copiar; no productivos |

`data_listing.txt` (2.4 MB, artefacto de listado) **no** se migra. La carpeta `data/` no
existe/está ignorada; el ETL sigue leyendo/escribiendo por rutas CLI.

### Fase A2.3 — Packaging: dependencias del ETL en `pyproject.toml`

El ETL añade dependencias que hoy no están en `pyproject.toml` (que solo tiene `torch`,
`numpy`, `tqdm`, `matplotlib`):

1. **Core** (necesarias para `import modelo_itm.etl`): añadir `pandas`, `openpyxl`,
   `scikit-learn`.
2. **Extra opcional `[db]`** (solo ETL relacional, no requerido para transformar tensores ni
   entrenar): `mysql-connector-python`, `pyodbc`.
3. No hace falta declarar el subpaquete: `tool.setuptools.packages.find` con `where=["src"]`
   ya recoge `modelo_itm.etl` automáticamente al ser subpaquete de `modelo_itm`.
4. Reinstalar en editable: `pip install -e ".[dev]"` (y `".[dev,db]"` si se usa la BD).

### Fase A2.4 — Verificación de equivalencia del ETL

- `python -c "import modelo_itm.etl"` no falla.
- `python -m modelo_itm.etl --help` expone los mismos flags que `python -m cmg2tensor --help`
  del repo original.
- `grep -rn "import cmg2tensor\|from cmg2tensor" src/ scripts/ tests/` sale vacío.
- `pytest tests/etl -m "not slow" -v` pasa (equivalente a los tests originales de
  `cmg2tensor/tests/`).
- (Si hay una simulación de muestra) una corrida corta del pipeline produce los mismos
  `.pt`/reportes que el `cmg2tensor` original **antes** de aplicar C1/B3.

**Documentación (parte de A2):** actualizar `CLAUDE.md` y `README.md`:
- §Estructura del ecosistema → el ETL ya vive en `src/modelo_itm/etl/`; el bloque
  `cmg2tensor/` externo pasa a "referencia histórica".
- §Comandos ETL → reemplazar `$env:PYTHONPATH="src"; python -m cmg2tensor …` por
  `python -m modelo_itm.etl …`; scripts en `scripts/etl/`.
- Rutas `--data-root`/`--raw-root` → apuntar a donde vivan los datos (ya no a
  `../cmg2tensor/`); documentar que los datos **no** se versionan.

**Verificación global de la Parte A2:** el pipeline completo (ETL → modelo → entrenamiento)
se ejecuta desde el paquete único `modelo_itm`; ya no hay dependencia de la ruta
`../cmg2tensor/`. `--data-root` puede apuntar a donde vivan los datos procesados (p. ej.
`data/processed/` local o una ruta externa).

---

# PARTE B — Correcciones sobre la nueva estructura

> Cada fase va en su propia rama `bug/`, `feature/` o `exp/` desde `development` y no se
> cierra hasta pasar revisión (`@reviewer`) y pruebas (`@tester`). Las rutas apuntan a los
> módulos de `src/modelo_itm/` creados en la Parte A y a `src/modelo_itm/etl/` creado en la
> Parte A2; entre paréntesis, el anclaje original en `train_dataset.py` o
> `cmg2tensor/src/cmg2tensor/…` para trazabilidad.

## B.0 — Hallazgos de la revisión

### 🔴 Críticos — correctitud / validez científica

#### C1. ✅ **Corregido en Fase 6 (código).** `test/` se guarda **sin normalizar** y se usa como set de validación
> Reproceso de `data/processed/test/` con datos reales **pendiente** — no hay datos ni GPU
> en esta sesión; requiere confirmación explícita antes de ejecutarse (ver §2 Riesgos).
- **Dónde (tras Parte A2):** `src/modelo_itm/etl/pipeline/parallel.py`
  (`normalize_this = normalize and split != "test"` ← `cmg2tensor/…/pipeline/parallel.py:194`),
  `src/modelo_itm/etl/cli.py`
  (`if use_split_routing and split == "test": normalize_this_sim = False` ← `…/cli.py:402`).
  El entrenamiento usa `val_dir="test"` (`src/modelo_itm/config.py` ← `train_dataset.py:31`).
- **Problema:** `train/` se escala a `[0,1]` con min-max global de train, pero `test/` se
  escribe en **unidades físicas crudas**. El modelo entrena en `[0,1]` y valida contra otra
  escala. Consecuencia: `val_loss`, `val_sf_rmse`, `val_vd_rmse`, `val_*_r2` no son
  comparables con train; **early stopping** y selección de **`best.pt`** se guían por una
  señal inválida.
- **Corrección esperada:** normalizar `test/` con las **mismas estadísticas globales de
  train** (práctica estándar; **no** hay fuga porque los `min/max` vienen solo de train), o
  derivar validación de un split de `train/` ya normalizado. Conservar la capacidad de
  reconstruir unidades físicas para inferencia (guardar `min/max` por variable o reutilizar
  `train.json`).
- **Corrección aplicada:** removido el forzado `normalize=False` para `split == "test"` en
  ambos pipelines (`parallel.py:194`, `cli.py:401-403`); la ruta de `global_stats`
  (calculado **solo** de `train_dirs` en Phase 1, sin fuga) ya se pasaba correctamente a
  todos los splits en Phase 2 — solo faltaba dejar de bloquearla para test.
- **Verificación:** `tests/etl/test_c1_normalize_test_split.py` (nuevo) — corre
  `run_batch_pipeline` con `split_map` train+test y confirma (a) los valores de `test/`
  quedan en `[0,1]` igual que `train/` (antes: unidades físicas crudas fuera de rango), y
  (b) `layer_cubes_report.json` de una simulación de `test/` registra
  `normalization.applied == True`. 2/2 passed. Expuso además **A4** (import roto que
  bloqueaba todo el pipeline paralelo), corregido en el mismo commit.

#### C2. ✅ **Corregido en Fase 8 (Opción A, confirmada con el usuario vía `AskUserQuestion`).** La incertidumbre MC Dropout era **código muerto** (siempre cero)
- **Dónde:** `PhysicalFNOArchitecture` (`src/modelo_itm/models/fno.py` +
  `models/blocks.py` ← `train_dataset.py:625-673`) no contiene ninguna capa `nn.Dropout*`.
  `model_has_dropout()` siempre devuelve `False`, así que todo
  `src/modelo_itm/inference/uncertainty.py` (← `:206-355`) hace short-circuit y devuelve
  **0.0 de incertidumbre y 1.0 de confianza** en todas las épocas.
- **Problema:** contradice `CLAUDE.md` (feature "MC Dropout uncertainty") y el paper. Se
  paga complejidad y cómputo sin obtener señal.
- **Corrección esperada — decisión de diseño requerida:**
  - **Opción A (recomendada):** insertar `nn.Dropout2d(p)` en `ResBlock`/decoder y/o
    `FiLMSpectralBlock`, exponer `dropout_p` en `Config`, verificar que MC Dropout produce
    desviación estándar no trivial.
  - **Opción B:** aislar toda la maquinaria detrás de un flag `enable_uncertainty=False` y
    no anunciarla en `CLAUDE.md`/paper hasta activarla.
- **Corrección aplicada (Opción A):** `ResBlock` (`models/blocks.py`) inserta
  `nn.Dropout2d(dropout_p)` entre sus dos convoluciones; se usa tanto en el encoder como
  en el decoder de `PhysicalFNOArchitecture` (no se tocó `FiLMSpectralBlock` — el spec
  permitía "y/o" y `ResBlock` ya cubre ambos extremos del pipeline sin arriesgar la lógica
  de FFT/parámetros complejos). Nuevo campo `Config.dropout_p: float = 0.1`, propagado a
  través de `PhysicalFNOArchitecture(..., dropout_p=...)`, `training/loop.py::main()` y
  expuesto como `--dropout-p` en `scripts/train.py`. `checkpoint.py::build_run_signature`
  incluye `dropout_p` (cambia la arquitectura, invalida `--auto-resume` con checkpoints
  previos — ver §2 Riesgos, ya documentado para C2).
- **Verificación:** `tests/unit/test_uncertainty.py` (nuevo, 5 tests) —
  `model_has_dropout()` es `True` con el default; **criterio de aceptación exacto del
  spec**, MC Dropout produce `pred_std.max() > 1e-4` (no trivial) con `dropout_p=0.2`;
  con `dropout_p=0.0` la capa existe pero es no-op (`std≈0`, caso límite documentado, no
  el default); `predict_with_uncertainty(passes=1)` sigue devolviendo `std=0` sin overhead;
  `calibrate_uncertainty` ya no hace short-circuit trivial — produce `alpha`/`error_q95`
  reales con dropout activo. 5/5 passed. Suite completa: 56 passed (antes 51), mismo bug
  preexistente B5 sin relación.

#### C3. ✅ **Corregido en Fase 6.** Métricas R² y RMSE promediadas por batch (estadísticamente incorrectas)
- **Dónde:** `evaluate_epoch` (`src/modelo_itm/training/loop.py` ←
  `train_dataset.py:798-847`) acumula `sf_r2`, `vd_r2`, `sf_rmse`, `vd_rmse` por batch y
  luego promedia (`finalize_running_stats` en `training/metrics.py`).
- **Problema:** el R² **no es aditivo**; promediar R² por batch ≠ R² global. El RMSE
  promediado (media de `sqrt(mse_batch)`) tampoco equivale al RMSE global, y con el último
  batch de tamaño distinto el sesgo crece. Las métricas del paper serían incorrectas.
- **Corrección esperada:** acumular globalmente `ss_res`/`ss_tot` para R², y suma de errores
  cuadrados + conteo para RMSE (`sqrt(sum_se / n)`), calculando la métrica una sola vez al
  final de la época. Las *loss* sí pueden promediarse por batch.
- **Corrección aplicada:** `training/metrics.py` gana
  `init_global_regression_accumulators` / `update_global_regression_accumulators` /
  `finalize_global_regression_metrics` — acumulan `sum_sq_error`, `sum_gt`, `sum_gt_sq` y
  `count` por variable (SF/VD) a través de todos los batches; R² usa la identidad
  `SS_tot = sum(gt²) - n·mean(gt)²` (evita una segunda pasada) y RMSE usa
  `sqrt(sum_sq_error / n)`, calculados **una sola vez** al final de `evaluate_epoch`. Las
  *loss* siguen promediándose por batch vía `running_stats` (Welford), sin cambios.
- **Hallazgo adicional descubierto al aplicar C3:** la migración de la Fase 2 había
  omitido la llamada a `evaluate_epoch(train_dl)` que el `train_dataset.py` original hacía
  cada época (`main()` original, líneas 1344-1349) para obtener métricas reales de train;
  en su lugar, `train_sf_r2`/`train_vd_r2` copiaban `val_metrics` y
  `train_sf_rmse`/`train_vd_rmse` usaban `sf_loss`/`vd_loss` (valores de loss, no RMSE) —
  historial con datos falsos. Restaurado en `main()`: `evaluate_epoch(train_loader)` real
  por época, recalibración de incertidumbre por época, auto-pause por hora
  (`cfg.pause_hour`) y resume con candidatos `[latest.pt, best.pt]`, replicando el
  comportamiento original que se había perdido silenciosamente en la Fase 2 (decisión
  confirmada con el usuario: restaurar las 4 desviaciones antes de aplicar C3).
- **Verificación:** `tests/unit/test_metrics.py` (nuevo, 4 tests) — compara la acumulación
  global contra `sklearn.metrics.r2_score` + RMSE de referencia sobre el dataset completo
  (batch único y batches de tamaño desigual con último batch parcial), y confirma
  explícitamente que el resultado correcto **difiere** del promedio ingenuo por batch (el
  bug que se corrige). 4/4 passed.

#### C4. ✅ **Corregido en Fase 9.** `in_c` mal interpretado — bloqueaba **todo** entrenamiento real desde la Fase 2
- **Dónde:** `PhysicalFNOArchitecture.__init__` (`src/modelo_itm/models/fno.py`).
- **Problema:** en el `train_dataset.py` **original**, `in_c=5` ya representa el total de
  canales que entran al encoder **después** de concatenar el canal de profundidad
  (`forward`: `torch.cat([x, depth_map], dim=1)`) — el encoder original usaba
  `nn.Conv2d(in_c, h_dim, ...)` sin sumar nada. El dataset real (`DatasetLayers`) apila
  exactamente 4 propiedades estáticas (AFI/COH/PERM/PORO), así que `x` trae 4 canales y
  `4 + 1 (profundidad) = 5 = in_c` (default), cuadrando exactamente. **Durante la
  migración de la Fase 2, "corregí" esto incorrectamente**: los tests sintéticos que
  escribí en ese momento (`test_model_forward.py`) construían `x` con **5** canales
  (asumiendo que `in_c` significaba "canales de `x` antes de profundidad"), lo cual
  disparaba un `RuntimeError` de shape contra el encoder original. Sin verificar contra
  el dataset real, "arreglé" el síntoma cambiando el encoder a
  `nn.Conv2d(in_c + 1, h_dim, ...)` — lo cual hacía pasar mis tests sintéticos (mal
  construidos) pero **rompía la migración con datos reales**: `DatasetLayers` produce 4
  canales, `4 + 1 = 5`, pero el encoder "corregido" esperaba `in_c + 1 + 1 = 6`.
  **Este bug estuvo presente desde el primer commit de la Fase 2 y bloqueaba
  completamente cualquier intento de entrenar con datos reales** — nunca se detectó
  porque ningún test end-to-end con el dataset real se había ejecutado hasta ahora (sin
  datos/GPU en las sesiones previas, según §2 Riesgos).
- **Cómo se descubrió:** al escribir un smoke test end-to-end de `main()` con datos
  sintéticos generados en el **formato real** de `DatasetLayers` (no un tensor `x`
  sintético suelto) para verificar M5 (logging), el forward falló con el mismatch de
  canales.
- **Corrección aplicada:** revertido el encoder a `nn.Conv2d(in_c, h_dim, ...)` (como el
  original); docstring añadido en `__init__` aclarando que `in_c` es el total
  post-concatenación, no los canales de `x`. Corregidos los 6 archivos de test que
  asumían la interpretación incorrecta (`test_model_forward.py`, `test_amp.py`,
  `test_training_loop.py`, `test_optim.py`, `test_uncertainty.py` — todos construían `x`
  con 5 canales; ahora usan 4).
- **Verificación:** nuevo `test_model_in_c_matches_concatenated_depth_channel` en
  `test_model_forward.py` (regresión explícita); **smoke test end-to-end completo**
  (`main()` con `DatasetLayers` real sobre datos sintéticos en disco, 1 época, CPU) pasa
  sin errores — primera vez que se ejecuta el pipeline de entrenamiento completo con el
  formato real de datos desde el inicio de este spec. `sf_uncertainty_mean`/
  `vd_uncertainty_mean` no triviales en el output real (confirma C2 en un run real, no
  solo en tests aislados). Suite completa: 76 passed, mismo bug preexistente B5.

### 🟠 Altos — bugs / acoplamiento

#### A1. ✅ **Corregido en Fase 7.** Índices temporales hardcodeados en la loss (acoplados a `time_steps=61`)
- **Dónde:** `compute_loss_terms` (`src/modelo_itm/training/losses.py` ←
  `train_dataset.py:685-689`): `pred[:, 0:1, 0]`, `pred[:, 1:21, 0]`, `pred[:, 21:61, 0]`.
- **Problema:** los segmentos `t0 / t1:20 / t21:60` y sus pesos asumen exactamente 61
  timesteps. Con otro `cfg.time_steps` los slices dejan de cubrir el tensor (p. ej.
  `time_steps=30` → `21:61` parcial), produciendo una loss silenciosamente incorrecta.
- **Corrección esperada:** derivar los límites de `cfg.time_steps` (o parametrizar los
  cortes en `Config` como fracciones/índices validados) y añadir una aserción de que los
  segmentos cubren exactamente `[0, time_steps)` sin solapamientos.
- **Corrección aplicada:** nueva `_segment_boundaries(time_steps)` en `losses.py` deriva
  `(b1, b2)` escalando proporcionalmente los límites originales (1, 21) sobre 61 —
  `b1 = max(1, round(time_steps/61))`, `b2 = round(21·time_steps/61)` clamped a
  `(b1+1, time_steps-1)` — de modo que `time_steps=61` reproduce exactamente `(1, 21)`
  (comportamiento default sin cambios). `compute_loss_terms` usa `pred.shape[1]` (el
  tensor real, no `cfg.time_steps`, evitando otra fuente de desincronización) y afirma
  `0 < b1 < b2 < time_steps` explícitamente.
- **Verificación:** `tests/unit/test_losses.py` (+9 tests) — cobertura exacta sin
  solapamiento parametrizada sobre 7 valores de `time_steps` (3 a 200), reproducción
  exacta del default 61, rechazo de `time_steps<3`, y loss finita para 3 valores no
  default (4/30/97), cumpliendo el criterio de aceptación ("al menos dos valores de
  `time_steps` distintos"). 17/17 passed.

#### A2. ✅ **Corregido en Fase 7 (decisión de diseño confirmada con el usuario).** Normalización de inyección doble / inconsistente
- **Dónde:** el ETL normaliza con stats globales
  (`normalize_series_minmax_with_global_stats`), pero `load_injection_series`
  (`src/modelo_itm/data/dataset.py` ← `train_dataset.py:499-521`) vuelve a aplicar
  `log1p` + normalización por `max` local por muestra.
- **Problema:** doble transformación no documentada; la escala efectiva depende del ETL.
  Dificulta reproducibilidad e interpretación física.
- **Corrección esperada:** un único punto de normalización de la inyección (ETL **o**
  dataset, no ambos), documentado. Si se mantiene en el dataset, consumir la serie cruda del
  ETL de forma consistente.
- **Decisión:** normalización **solo en el ETL** (`run_injection_excel_pipeline`, que ya usa
  la misma infraestructura `global_stats` de train que C1 estableció — consistente con el
  resto de variables del pipeline). Se descartó "solo en el dataset" por requerir forzar
  `normalize=False` específicamente para inyección en el ETL, rompiendo la consistencia con
  SF/VD/PERM/etc.
- **Corrección aplicada:** `load_injection_series` ya no aplica `log1p` ni reescalado por
  máximo local — solo alinea longitud a `time_steps` (pad/truncate) y sanitiza NaN/Inf como
  red de seguridad. Consume el tensor `.pt` del ETL ya normalizado en `[0,1]` tal cual.
- **Verificación:** `tests/unit/test_dataset.py` (+4 tests) — confirma pass-through exacto
  de valores ya normalizados por el ETL (sin transformación), padding/truncado correcto, y
  sanitización de NaN/Inf. 7/7 passed.

#### A3. ✅ **Corregido en Fase 7.** Evaluación redundante y costosa por época
- **Dónde:** el loop principal (`src/modelo_itm/training/loop.py:main` ←
  `train_dataset.py:1344-1349`) llama cada época: `run_one_epoch(train)` +
  `calibrate_uncertainty(val)` + `evaluate_epoch(train_dl)` + `evaluate_epoch(val_dl)`.
- **Problema:** `evaluate_epoch(train_dl)` hace un forward completo extra sobre **todo
  train** cada época (≈50% de cómputo adicional) para métricas que podrían acumularse en
  `run_one_epoch`. `calibrate_uncertainty` se recomputa cada época aunque sin dropout no
  aporta nada.
- **Corrección esperada:** acumular métricas de train durante el paso de entrenamiento (o
  evaluar train cada N épocas), y calibrar incertidumbre solo cuando la feature esté activa
  (ligado a C2), no incondicionalmente.
- **Corrección aplicada:** `run_one_epoch` ahora acumula `sf_r2`/`vd_r2`/`sf_rmse`/`vd_rmse`
  globalmente (misma acumulación de C3) durante el propio paso de entrenamiento,
  reutilizando el `pred`/`y` que ya se calculan para la loss — sin forward extra.
  `main()` elimina la llamada `evaluate_epoch(train_loader)` y usa directamente el
  resultado enriquecido de `run_one_epoch`. La recalibración de incertidumbre queda tras
  un guard `if model_has_dropout(model):` — solo se recalibra y reescribe el JSON cuando
  la feature está realmente activa (ligado a C2; hoy siempre `False`, sin dropout).
- **Nota de diseño:** las métricas de train ahora se observan con el modelo en modo
  `.train()` (como ocurre durante el paso real de optimización), no en una pasada `.eval()`
  aislada — es el patrón estándar en frameworks de DL (Lightning, Keras) y hoy no cambia
  nada numéricamente porque el modelo no tiene capas con comportamiento train/eval
  distinto (sin dropout/batchnorm). Cuando C2 añada dropout real, las métricas de train
  reflejarán el efecto del dropout, tal como se espera convencionalmente.
- **Verificación:** `tests/unit/test_training_loop.py` (nuevo, 2 tests) — confirma que
  `run_one_epoch` devuelve `sf_r2`/`vd_r2`/`sf_rmse`/`vd_rmse` finitos, y que coinciden
  exactamente con una acumulación manual independiente sobre los mismos batches. 2/2
  passed.

#### A4. ✅ **Corregido en Fase 6.** Import roto bloqueaba el pipeline paralelo completo
- **Dónde:** `src/modelo_itm/etl/pipeline/parallel.py` (línea 190, dentro del worker de
  Phase 2) ← preexistente en `cmg2tensor/src/cmg2tensor/pipeline/parallel.py:190`
  (confirmado con `diff` contra el repo original antes de la Parte A2 — no introducido por
  la migración).
- **Problema:** `from ..pipelines import _run_requested_pipelines` (plural) apuntaba a un
  módulo inexistente (`etl/pipelines.py`); el módulo real es `etl/pipeline/serial.py`
  (singular, hermano de `parallel.py`). Al ser un import diferido dentro del worker
  (`ProcessPoolExecutor`, ejecutado en un proceso spawneado), **nunca se ejecutaba durante
  la recolección de tests** — solo al correr `run_batch_pipeline` de verdad. Como los tests
  que la ejercitan están marcados `@pytest.mark.slow` (excluidos del CI rápido con
  `-m "not slow"`), el fallo pasó inadvertido: **el pipeline paralelo nunca produjo un solo
  archivo de salida exitoso**, en el original ni en la migración, hasta este hallazgo.
  Descubierto al escribir los tests de verificación de C1 (`tests/etl/test_c1_normalize_test_split.py`),
  que fallaban con `ModuleNotFoundError` en todos los workers de Phase 2.
- **Corrección aplicada:** `from ..pipelines import …` → `from .serial import …`.
- **Verificación:** `pytest tests/etl -v` (incluyendo los marcados `slow`, antes nunca
  ejercitados en CI) — 40 passed, solo los 2 fallos preexistentes de **B5** persisten.

### 🟡 Medios — buenas prácticas de Deep Learning

- **M1. ✅ Corregido en Fase 9. Sin scheduler de LR.** `lr` constante `8e-4` con `AdamW`. Añadir
  `CosineAnnealingLR`/`OneCycleLR`/warmup+decay; guardar el estado del scheduler en el
  checkpoint (`training/checkpoint.py`) para que `--auto-resume` sea consistente.
  **Aplicado:** `training/optim.py::build_scheduler` (`CosineAnnealingLR`, `T_max=cfg.epochs`,
  `eta_min=cfg.lr_min`); `Config.lr_scheduler`/`lr_min` nuevos (`lr_scheduler=None` desactiva);
  `checkpoint.py` guarda/carga `scheduler_state_dict` (permisivo: si el checkpoint no lo
  tiene, o el scheduler actual es distinto, no aborta el resume — solo el scheduler arranca
  desde cero). `scheduler.step()` se llama al final de cada época, **después** de guardar
  los checkpoints (para que `optimizer_state_dict`/`scheduler_state_dict` queden en
  sincronía). Expuesto como `--lr-scheduler`/`--lr-min` en `scripts/train.py`.
  **Verificación:** `tests/unit/test_scheduler.py` (5 tests) — el LR decae de `lr` a
  `lr_min` en un ciclo completo (no permanece constante), round-trip de checkpoint
  preserva el punto exacto del ciclo (no lo reinicia), resume tolera checkpoints sin
  estado de scheduler. 5/5 passed.
- **M2. ✅ Corregido en Fase 9. Sin *mixed precision* (AMP).** El forward expande `z` a `(b·T, h_dim, H, W)`
  (`models/fno.py` ← `train_dataset.py:667`), muy pesado en memoria. Integrar
  `torch.autocast` + `GradScaler`; cuidar las FFT (`FiLMSpectralBlock`) que suelen requerir
  `float32`. **Aplicado:** `Config.use_amp: bool = False` (opt-in, solo efectivo con
  `device.type=="cuda"`); `run_one_epoch`/`evaluate_epoch` envuelven forward+loss en
  `torch.autocast`; `run_one_epoch` usa `torch.amp.GradScaler` para
  `backward`/`unscale_`/`step` (con `enabled=False` es un no-op transparente — mismo código
  path con o sin AMP). `FiLMSpectralBlock.forward` fuerza la FFT/multiplicación espectral a
  float32 con `torch.autocast(enabled=False)` anidado, independientemente del contexto
  externo. **Verificación:** `tests/unit/test_amp.py` (3 tests) — confirma que la FFT se
  mantiene finita bajo un contexto autocast activo (simulado con bfloat16 en CPU) y que
  `use_amp=True` en CPU es un no-op seguro. **Limitación:** sin GPU en esta sesión, el path
  real de float16/GradScaler en CUDA no se ejerció; verificación completa queda pendiente
  de hardware.
- **M3. ✅ Corregido en Fase 10 (EXPERIMENTAL, rama `exp/` sugerida por el spec).**
  **Sin normalización interna en la red.** No hay `LayerNorm`/`GroupNorm` en encoder,
  bloques FNO ni decoder. Evaluar añadirla para estabilidad. **Aplicado:** `ResBlock`
  (`models/blocks.py`) acepta `use_group_norm` — inserta `nn.GroupNorm` tras cada `Conv2d`
  (encoder y decoder). Nueva `_group_norm_num_groups(c)` calcula un `num_groups` seguro
  (prueba 8/4/2/1 hasta encontrar un divisor exacto de `c`, evitando el error de PyTorch
  cuando `c % num_groups != 0`). Nuevo `Config.use_group_norm: bool = False` — **desactivado
  por defecto**, no se cambia la arquitectura del modelo default (tal como pide el spec: no
  tocar la arquitectura salvo donde se indique explícitamente). Incluido en
  `build_run_signature` (cambia arquitectura → invalida `--auto-resume` si difiere). Expuesto
  como `--use-group-norm` en `scripts/train.py`. **No verificado en un entrenamiento real
  completo** (sin datos/GPU en esta sesión) — queda como corrección disponible para evaluar,
  no como recomendación de activarla sin más pruebas.
  **Verificación:** `tests/unit/test_model_forward.py` (+3 tests) — forward finito con
  `h_dim` no múltiplo de 8 (12, ejercita el fallback de `_group_norm_num_groups`), capas
  `GroupNorm` presentes solo cuando `use_group_norm=True`. Smoke test end-to-end con
  `use_group_norm=True` pasa sin errores.
- **M4. ✅ Corregido en Fase 10. Reproducibilidad incompleta.** `cudnn.benchmark=True` (no determinista), sin
  `worker_init_fn` para semillar workers del `DataLoader`, `random.randrange` en
  `save_epoch_visuals`. Añadir `worker_init_fn`, `generator` semillado y documentar el
  trade-off determinismo/rendimiento (o exponer flag `deterministic`). **Aplicado:**
  (1) `resolve_device(requested, deterministic=False)` — nuevo parámetro; con
  `deterministic=True` fija `cudnn.benchmark=False` + `cudnn.deterministic=True` (más lento
  pero reproducible en CUDA); default `False` preserva el comportamiento previo
  (`cudnn.benchmark=True`). Nuevo `Config.deterministic` propagado en `main()` y expuesto
  como `--deterministic`. (2) `data/loaders.py::build_loader` — **siempre** (bajo costo, sin
  trade-off real) pasa `generator=torch.Generator().manual_seed(cfg.seed)` y
  `worker_init_fn=_seed_worker` (semilla `numpy`/`random` en cada proceso worker vía
  `torch.initial_seed()`, patrón oficial de PyTorch). (3) `visualization/plots.py::
  save_epoch_visuals` — reemplazado el módulo global `random.randrange` por un
  `random.Random(cfg.seed + epoch)` local, reproducible por época sin depender de ni
  afectar otro estado de aleatoriedad global del proceso.
  **Verificación:** `tests/unit/test_reproducibility.py` (nuevo, 5 tests) — dos
  `DataLoader` con la misma `cfg.seed` producen el mismo orden de shuffle exacto; seeds
  distintas producen órdenes distintos; la selección de muestra en `save_epoch_visuals` es
  reproducible con la misma seed+epoch (verificado perturbando el estado global de `random`
  antes de cada llamada) y varía entre épocas distintas.
- **M5. ✅ Corregido en Fase 9. Logging por `print`.** `CLAUDE.md` §Convenciones pide evitar `print` para
  diagnóstico permanente. Migrar a `logging`/`tqdm.write` y, dado el paper, considerar
  TensorBoard o Weights & Biases. **Aplicado:** nuevo `utils/logging.py` con
  `get_logger(name)`/`configure_logging()` — instala un `_TqdmLoggingHandler` que enruta
  todos los logs a través de `tqdm.write()` (no rompe las barras de progreso activas).
  Reemplazados los 23 `print()` de `training/loop.py` (21) e `inference/uncertainty.py` (2)
  por `logger.info`/`.warning`/`.debug` según severidad. TensorBoard/W&B quedan fuera de
  alcance (mencionados como "considerar", no como corrección obligatoria).
  **Verificación:** `tests/unit/test_logging.py` (4 tests) — idempotencia de
  `configure_logging()`, handler instalado correctamente; `grep -rn "print(" src/modelo_itm/`
  (fuera de `etl/`) sale vacío; **smoke test end-to-end** de `main()` confirma el logging
  funciona en un entrenamiento real (formato `HH:MM:SS [INFO] mensaje`).
- **M6. ✅ Corregido en Fase 9. Sin guardas de NaN/Inf** en loss o gradientes (solo hay `nan_to_num` en la
  inyección). Añadir detección temprana de `NaN`/`Inf` en la loss para abortar controlado.
  **Aplicado:** `run_one_epoch` (`training/loop.py`) lanza `RuntimeError` explícito si
  `loss` no es finita (tras `compute_loss_terms`, antes de `backward()`) o si la norma de
  gradiente devuelta por `clip_grad_norm_` no es finita (tras `backward()`, antes de
  `optimizer.step()`) — aborta el entrenamiento con un mensaje claro (batch, valores) en
  vez de propagar `NaN` silenciosamente al modelo. **Verificación:**
  `tests/unit/test_training_loop.py` (+3 tests) — ground truth con `NaN`/`Inf` dispara
  `RuntimeError` con el mensaje esperado; datos normales no generan falsos positivos.
- **M7. ✅ Corregido en Fase 9. `weight_decay` sobre embeddings y bias.** `AdamW` decae todos los parámetros,
  incluidos `t_embed`, `gamma/beta` de FiLM y bias. Separar param groups para no penalizar
  embeddings/bias. **Aplicado:** nuevo `training/optim.py::build_param_groups(model,
  weight_decay)` — separa los parámetros en dos grupos: `decay` (pesos de
  Conv2d/Linear/parámetro espectral) y `no_decay` (`weight_decay=0.0`: todo `.bias`,
  `t_embed.weight` vía detección de `nn.Embedding`, y las capas `gamma`/`beta` de
  `FiLMSpectralBlock` completas). `main()` usa `AdamW(build_param_groups(...), lr=cfg.lr)`.
  **Verificación:** `tests/unit/test_optim.py` (4 tests) — cada parámetro entrenable
  aparece exactamente una vez entre ambos grupos (sin pérdidas ni duplicados), clasificación
  correcta verificada contra los nombres reales del modelo, funciona con `AdamW` real
  (forward + backward + step sin errores). 4/4 passed.
- **M8. ✅ Corregido en Fase 10 (y en fases previas).** Sin tests de lógica del entrenamiento.
  Más allá de los tests mínimos de la Fase 2b, añadir cobertura real: `DatasetLayers`
  (shapes, padding de inyección), `compute_loss_terms` (segmentación correcta para varios
  `time_steps`), métricas globales (C3), forward del modelo. **Estado de cada ítem:**
  `compute_loss_terms` con varios `time_steps` → cubierto en A1 (Fase 7, 9 tests
  parametrizados). Métricas globales → cubierto en C3 (Fase 6, 4 tests contra `sklearn`).
  Forward del modelo → cubierto en Fase 2b + Fase 8/9/10 (dropout, AMP, GroupNorm). Padding
  de inyección en `DatasetLayers` específicamente (lo único que faltaba) → nuevos
  `test_dataset_layers_pads_short_injection_series` y
  `test_dataset_layers_truncates_long_injection_series` en `tests/unit/test_dataset.py`,
  verificando el padding/truncado **dentro de `__getitem__`** (no solo en
  `load_injection_series` aislada) cuando la serie de inyección real tiene distinta
  longitud que `y.shape[0]` (el target). Suite final: **108 tests, 0 fallos** (incluye los
  marcados `slow`, antes bloqueados por A4).

### 🔵 Bajos — menores / higiene

- **B1. ✅ Corregido en Fase 10.** Estado global mutable de "emitir una vez"
  (`_MC_DROPOUT_WARNING_EMITTED`, `_CUDA_BATCH_REPORT_EMITTED`, reubicados en la Fase 2):
  encapsular en un objeto de estado o `logging` con filtro. **Aplicado:** nueva clase
  `utils/logging.py::EmitOnce` — `should_emit(key)` devuelve `True` solo la primera vez
  por clave, `reset(key=None)` permite reiniciar. Reemplaza ambas variables globales
  `global`/`_XXX_EMITTED` por instancias `_emit_once = EmitOnce()` a nivel de módulo en
  `training/loop.py` e `inference/uncertainty.py` — sin `global`, estado encapsulado y
  testeable de forma aislada.
  **Verificación:** `tests/unit/test_logging.py` (+5 tests) — solo `True` la primera vez
  por clave, claves independientes, `reset()` parcial/total, instancias aisladas entre sí.
- **B2. ✅ Corregido en Fase 10.** `torch.load(..., weights_only=False)` en `training/checkpoint.py` (← `:459`) y
  `load_pt` sin `weights_only` en `data/dataset.py` (← `:488`): revisar compatibilidad con
  PyTorch ≥2.6 (cambio de default) y seguridad al cargar `.pt` de terceros. **Investigado
  empíricamente:** todo lo que se guarda en el checkpoint (state dicts, `config`/`metrics`/
  `run_signature` ya convertidos a dict con `asdict`) y en los `.pt` de inyección del ETL
  (dict con tensor + metadata básica: str/list/dict) son tipos compatibles con
  `weights_only=True` — confirmado con una prueba de round-trip real antes de aplicar el
  cambio. **Aplicado:** `checkpoint.py::try_resume_training` y `data/dataset.py::load_pt`
  ahora usan `weights_only=True` explícitamente (más seguro al cargar `.pt`/checkpoints de
  terceros; ya era el default implícito en PyTorch 2.12.1 para `load_pt`, ahora es
  explícito y documentado).
  **Verificación:** toda la suite de tests de checkpoint/dataset (round-trip real en
  `test_scheduler.py`) sigue pasando sin cambios.
- **B3. ✅ Corregido en Fase 10 (utilidad opcional, no integrada en el pipeline).** Min-max
  sensible a outliers; evaluar clip de percentiles (p1/p99) o *robust scaling* antes de
  escalar (`src/modelo_itm/etl/normalize.py` ← `cmg2tensor/…/normalize.py`). **Aplicado:**
  nueva `clip_percentiles(values, p_low=1.0, p_high=99.0)` en `etl/normalize.py` — recorta
  valores fuera del rango de percentiles (ignora `NaN` al calcular los límites). **NO se
  integra automáticamente** en `normalize_cubes_minmax*`/`normalize_series_minmax*`:
  cambiar el comportamiento default de normalización requeriría reprocesar datos reales
  (mismo tipo de decisión que C1), fuera de alcance sin confirmación explícita y sin datos
  en esta sesión. Queda disponible para quien quiera evaluarla explícitamente componiéndola
  antes de normalizar.
  **Verificación:** `tests/etl/test_normalize_robust.py` (nuevo, 5 tests) — recorta
  outliers reales, no distorsiona datos sin outliers, maneja arrays vacíos/con solo `NaN`,
  ignora `NaN` al calcular los límites de percentil.
- **B4. ✅ Corregido en Fase 10.** Caché de inyección (`self._inj_cache` en `data/dataset.py` ← `:528`) sin cota
  superior (impacto bajo). **Aplicado:** `DatasetLayers` acepta `max_inj_cache_size=512`
  (nuevo parámetro); `_inj_cache` pasa de `dict` a `collections.OrderedDict` con política
  LRU manual (`move_to_end` en cada acceso/inserción, `popitem(last=False)` cuando se
  supera el límite). Nota: `inj_key` es por-simulación (compartido por hasta 97 capas), no
  por-muestra, así que el tamaño teórico ya era bajo (~n simulaciones); el límite protege
  explícitamente datasets con muchas simulaciones distintas.
  **Verificación:** `tests/unit/test_dataset.py` (+2 tests) — con `max_inj_cache_size=3` y
  6 simulaciones distintas, el caché nunca supera 3 entradas tras acceder a todas; con el
  default (512), las 6 simulaciones caben sin evicción.
- **B5. ✅ Corregido en Fase 10.** Tests desactualizados en `construir_histogramas_globales_por_capas`
  (`src/modelo_itm/etl/histograms.py` ← `cmg2tensor/…/histograms.py`, detectado en la
  verificación de equivalencia de la Fase A2.4, **confirmado preexistente** en el
  `cmg2tensor` original antes de la migración — no introducido por A2). La función añade
  deliberadamente una clave `_meta` al payload (paths de reportes, timestamps; consumida
  por el modo incremental) y el propio módulo expone `_variable_keys()` para excluirla al
  iterar variables, pero `tests/etl/test_histograms.py::test_construir_histogramas_globales_por_capas_matches_direct_histogram`
  y `::test_construir_histogramas_fallback_to_two_pass_when_report_ranges_missing`
  comparan `set(result.keys())` sin filtrar `_meta`, por lo que fallan aunque el
  comportamiento sea correcto (falso positivo que reduce la cobertura efectiva de esta
  función). **No es un problema de correctitud del pipeline** — los datos y el
  entrenamiento no se ven afectados. **Corrección esperada:** actualizar ambos tests para
  usar `_variable_keys(result)` (o `set(result.keys()) - {"_meta"}`) en las aserciones.
  **Aplicado:** ambos tests importan y usan `_variable_keys(result)` en vez de
  `result.keys()` directo. **Verificación:** 3/3 tests de `test_histograms.py` pasan — el
  último hallazgo pendiente del proyecto queda resuelto; la suite completa (incluyendo
  `slow`) da **108 passed, 0 failed**.

---

## B.1 — Fases de corrección

### Fase 6 — Correctitud crítica de datos y métricas (bloqueante) — [CÓDIGO DONE, DATOS PENDIENTES]
- **C1** ✅: normalizar `test/` con stats globales de train. Archivos (**ya migrados en la
  Parte A2**): `src/modelo_itm/etl/pipeline/parallel.py`, `src/modelo_itm/etl/cli.py`.
  `pipeline/serial.py` no requirió cambios (solo recibe `normalize` ya resuelto). Verificado
  con `tests/etl/test_c1_normalize_test_split.py` (sintético). **Pendiente:** regenerar
  `data/processed/test/` con datos reales (**⚠️ requiere confirmación**; sin datos ni GPU
  en esta sesión).
- **C3** ✅: acumulación global de R²/RMSE. Archivos: `src/modelo_itm/training/metrics.py`,
  `training/loop.py` (`evaluate_epoch`, `main`). Verificado con `tests/unit/test_metrics.py`
  contra `sklearn.metrics.r2_score`.
- **A4** ✅ (encontrado durante la verificación de C1, no en el B.0 original): import roto
  bloqueaba todo el pipeline paralelo. `src/modelo_itm/etl/pipeline/parallel.py`.
- **Validación pendiente:** una corrida corta con datos reales que muestre `val_loss` en la
  misma escala que `train_loss` y R²/SF/VD coherentes (requiere Fase A2 de datos + GPU, no
  disponibles en esta sesión).

### Fase 7 — Bugs de acoplamiento — [DONE]
- **A1** ✅ (segmentos de loss derivados de `time_steps` → `training/losses.py`).
- **A2** ✅ (normalización única de inyección, solo en el ETL → `data/dataset.py`
  simplificado, `etl/` sin cambios ya que ya normalizaba correctamente).
- **A3** ✅ (evaluación/calibración redundante → `training/loop.py`: métricas de train
  acumuladas en `run_one_epoch`, calibración tras guard `model_has_dropout`).
- **Nota:** `config.py` no requirió cambios para A1 (no se agregaron campos nuevos; los
  límites se derivan dinámicamente, no se parametrizan en `Config`).

### Fase 8 — Feature de incertidumbre (decisión de diseño) — [DONE]
- **C2** ✅ Opción A aplicada (confirmada vía `AskUserQuestion`: dropout real, no flag de
  aislamiento). Archivos: `src/modelo_itm/models/{fno,blocks}.py` (Dropout2d en ResBlock),
  `config.py` (`dropout_p`), `training/{loop,checkpoint}.py` (propagación + run_signature),
  `scripts/train.py` (`--dropout-p`), `CLAUDE.md` actualizado (arquitectura + tabla de
  hiperparámetros). `inference/uncertainty.py` no requirió cambios — ya estaba correctamente
  implementado, solo esperaba a que el modelo tuviera capas Dropout reales.

### Fase 9 — Buenas prácticas de entrenamiento — [DONE]
- **M1** ✅ (scheduler + estado en checkpoint), **M2** ✅ (AMP), **M6** ✅ (guardas NaN/Inf),
  **M7** ✅ (param groups), **M5** ✅ (logging). Archivos:
  `src/modelo_itm/training/{loop,checkpoint,optim}.py` (nuevo `optim.py`),
  `src/modelo_itm/utils/logging.py` (nuevo), `src/modelo_itm/models/{fno,blocks}.py`
  (AMP en FFT), `config.py`, `scripts/train.py`.
- **C4** ✅ (no planeado — descubierto al verificar M5 con un smoke test end-to-end):
  bug crítico de `in_c` en `models/fno.py` que bloqueaba **todo** entrenamiento real desde
  la Fase 2, introducido por una "corrección" mía basada en tests sintéticos mal
  construidos, nunca detectada por falta de un test end-to-end con el dataset real hasta
  ahora. Ver hallazgo C4 en B.0.

### Fase 10 — Robustez, reproducibilidad y tests — [DONE]
- **M3** ✅ (normalización interna, experimental, default desactivado), **M4** ✅
  (reproducibilidad: `deterministic`, `worker_init_fn`, RNG local en visualización), **M8**
  ✅ (cobertura de padding de inyección en `DatasetLayers`; el resto ya cubierto en fases
  previas), **B1–B5** ✅ todos corregidos. Archivos: `models/blocks.py` (`use_group_norm`),
  `models/fno.py`, `config.py`, `utils/device.py` (`deterministic`), `data/loaders.py`
  (`worker_init_fn`/`generator`), `visualization/plots.py` (RNG local),
  `utils/logging.py` (`EmitOnce`), `training/{loop,checkpoint}.py`, `data/dataset.py`
  (`weights_only`, caché LRU), `etl/normalize.py` (`clip_percentiles`),
  `tests/etl/test_histograms.py` (B5).
- **Estado final del spec-000:** todas las fases (A, A2, 6–10) completadas. Suite completa:
  **108 passed, 0 failed** (incluye tests `slow`, antes bloqueados por A4). Smoke test
  end-to-end con todas las features combinadas (scheduler + AMP + GroupNorm +
  deterministic + auto-resume + save_epoch_pngs) pasa sin errores.

---

## 1. Archivos impactados (resumen consolidado)

| Archivo / carpeta | Fases | Naturaleza |
|---|---|---|
| `01-Modelo-ITM/pyproject.toml` | 1 | Fijar `requires-python` |
| `01-Modelo-ITM/.venv/` | 1 | Nuevo, ignorado por git |
| `01-Modelo-ITM/src/modelo_itm/**` | 2, 6–10 | Código migrado + correcciones |
| `01-Modelo-ITM/src/modelo_itm/etl/**` | A2, 6, 10 | ETL migrado (`cmg2tensor` plegado) + correcciones C1/B3 |
| `01-Modelo-ITM/scripts/train.py` | 2 | Nuevo entrypoint CLI |
| `01-Modelo-ITM/scripts/etl/**` | A2 | Scripts ETL migrados (`make_split`, `etl_mysql`, …) |
| `01-Modelo-ITM/sql/**` | A2 | DDL MySQL/SQL Server migrado |
| `01-Modelo-ITM/tests/**` | 2b, 10 | Tests de migración + cobertura real |
| `01-Modelo-ITM/tests/etl/**` | A2, 10 | Tests del ETL migrados + nuevos |
| `01-Modelo-ITM/pyproject.toml` | A2 | Deps del ETL (`pandas`, `openpyxl`, `scikit-learn`; extra `[db]`) |
| `01-Modelo-ITM/.git/` | 3 | Nuevo repositorio local |
| Remoto GitHub (`01-Modelo-ITM`) | 4 | Nuevo recurso externo |
| `Codigo Entrenamiento/train_dataset.py` | 5 | Conservado como referencia (no se borra) |
| `cmg2tensor/` (repo de Reinaldo-06) | A2 | Conservado como referencia (copia limpia; no se borra ni sincroniza) |
| `01-Modelo-ITM/CLAUDE.md` | 1, 5, 8, A2 | Comandos, estructura, ETL integrado, alcance de incertidumbre |
| `01-Modelo-ITM/README.md` | 5, A2 | Estado del proyecto y pipeline completo |
| `data/processed/test/` | 6 | Reproceso (**⚠️ requiere confirmación**) |

---

## 2. Riesgos y precondiciones

- **Orden obligatorio:** la Parte B referencia módulos que crean las Partes A y A2. No
  iniciar correcciones antes de completar ambas migraciones y verificar equivalencia. En
  particular, C1 y B3 operan sobre `src/modelo_itm/etl/`, que **no existe** hasta la Parte A2.
- **Sin GPU/datos en esta sesión:** la equivalencia numérica de la Fase 2 y la validación de
  la Fase 6 requieren acceso a datos procesados y, idealmente, GPU. Documentar si quedan
  pendientes.
- **Migración del ETL (Parte A2):** es copia **solo de código**; los datos de `cmg2tensor`
  no están versionados y no se tocan. Riesgo principal: imports mal reescritos → validar con
  `grep` que no queda `cmg2tensor` en el código migrado y correr los tests del ETL.
- **Dependencias nuevas del ETL:** `pandas`, `openpyxl`, `scikit-learn` (core) y
  `mysql-connector-python`, `pyodbc` (extra `[db]`) — **mencionar y confirmar** antes de
  instalarlas (§Dependencias de `CLAUDE.md`).
- **`gh` CLI no instalado:** la Fase 4 requiere instalarlo o usar la web de GitHub.
- **Cuenta/organización de GitHub sin confirmar:** no crear el repo remoto sin que el usuario
  indique cuenta, visibilidad y nombre.
- **Reproceso de datos (C1):** regenerar `data/processed/test/` toca datos derivados;
  respaldar antes y **no ejecutar sin confirmación explícita**. No sobrescribir
  `reports/global_normalization/train.json`.
- **Compatibilidad de checkpoints:** cambiar `Config`, el modelo (C2/M3) o añadir scheduler
  (M1) altera `run_signature`/el `state_dict`; los checkpoints previos quedarán
  incompatibles con `--auto-resume`. Documentarlo y respaldar `best.pt`/`latest.pt` antes.
- **Decisión (C2):** resuelta — Opción A confirmada con el usuario en la Fase 8. No hay
  checkpoints reales previos en este proyecto todavía (§0), así que la invalidación de
  `--auto-resume` por `dropout_p` en `run_signature` no afecta datos existentes.
- **Sin pérdida de datos en la migración:** al momento de escribir este spec no existen
  checkpoints ni outputs previos; si aparecen antes de ejecutar la Fase 5, respaldarlos.

---

## 3. Criterios de aceptación

**Parte A — Migración:**
- [ ] `python --version` en `01-Modelo-ITM/.venv` reporta `3.12.x`.
- [ ] `pip install -e ".[dev]"` instala `modelo_itm` sin errores.
- [ ] Todo el código de `train_dataset.py` tiene destino 1:1 en `src/modelo_itm/` según el
      Mapa de módulos, sin funciones perdidas.
- [ ] `python scripts/train.py --help` expone los mismos flags que el script original.
- [ ] `pytest tests/unit -v` pasa (incluye los 3 tests mínimos de la Fase 2b).
- [ ] `01-Modelo-ITM` es repositorio git con ramas `main` y `development`.
- [ ] (Si se completó la Fase 4) El remoto de GitHub está configurado y ambas ramas
      publicadas.
- [ ] No quedan referencias rotas a `Codigo Entrenamiento/` en `CLAUDE.md`, `README.md` ni
      `specs/` (se conserva como referencia por decisión del usuario).

**Parte A2 — Migración del ETL:**
- [ ] `python -c "import modelo_itm.etl"` no falla.
- [ ] `python -m modelo_itm.etl --help` expone los mismos flags que `python -m cmg2tensor`.
- [ ] `grep -rn "import cmg2tensor\|from cmg2tensor" src/ scripts/ tests/` sale vacío.
- [ ] `pytest tests/etl -m "not slow" -v` pasa.
- [ ] `pyproject.toml` declara las deps del ETL; `pip install -e ".[dev]"` reinstala sin
      errores.
- [ ] El repo original de `cmg2tensor` queda intacto (copia limpia, no se sincroniza).

**Parte B — Correcciones:**
- [x] `val_loss` y métricas de validación en la misma escala que train (C1) — corregido en
      código y verificado con datos sintéticos; **regenerar `data/processed/test/` con
      datos reales queda pendiente** (requiere confirmación explícita, sin datos en esta
      sesión).
- [x] R² y RMSE calculados globalmente y verificados contra un cómputo de referencia (C3).
- [x] La loss es correcta para al menos dos valores de `time_steps` distintos (A1) —
      verificado para 3 valores no default (4/30/97) + 7 valores de cobertura exacta.
- [x] La incertidumbre está operativa **o** aislada tras flag y documentada (C2) — Opción A:
      MC Dropout real, `pred_std.max() > 1e-4` verificado con dropout activo.
- [x] Buenas prácticas de Fase 9 aplicadas y verificadas (M1 scheduler, M2 AMP, M5 logging,
      M6 guardas NaN/Inf, M7 param groups) — 20 tests nuevos entre `test_scheduler.py`,
      `test_amp.py`, `test_logging.py`, `test_optim.py` y `+3` en `test_training_loop.py`.
- [x] `PhysicalFNOArchitecture` funciona con el dataset real end-to-end (C4) — smoke test
      completo de `main()` con `DatasetLayers` real sobre datos sintéticos en disco pasa
      sin errores de shape.
- [x] Existe cobertura de tests para dataset, loss, métricas globales y forward del modelo
      (M8) y `pytest -m "not slow"` pasa — 108/108 tests pasan (incluidos los `slow`).
- [x] `CLAUDE.md` refleja el alcance real de la feature de incertidumbre y la nueva
      estructura de módulos — arquitectura, tabla de hiperparámetros (incluye
      `use_group_norm`/`deterministic`/`use_amp`/`lr_scheduler`/`dropout_p`), pipeline ETL
      integrado y comandos actualizados en todas las fases.

---

> **Nota post-spec (2026-07-02, incidente de git):** este documento fue restaurado tras una
> pérdida accidental de datos causada por operaciones de `git checkout`/`merge` que
> sobrescribieron y luego eliminaron la versión en disco (un stub antiguo de 449 líneas
> había quedado trackeado en `main` en un commit temprano, mientras que esta versión
> completa de 925 líneas vivía como archivo local sin trackear tras añadir `specs/` al
> `.gitignore`). Contenido reconstruido íntegramente desde el contexto de la sesión que lo
> redactó; no se perdió información. **Lección aplicada:** `specs/` debe tratarse como
> contenido valioso no versionado — evitar operaciones de `git checkout`/`merge` sin
> verificar antes si hay archivos locales gitignored en las rutas que el commit destino
> todavía trackea.
