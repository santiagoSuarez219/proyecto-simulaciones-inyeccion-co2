# spec-000 — Migración a 01-Modelo-ITM + correcciones de entrenamiento y pipeline [IN PROGRESS]

> **Autor:** revisión de código (rol `@architect`)
> **Fecha:** 2026-07-02
> **Resultado de:** unificación de los antiguos `spec-000` (migración/entorno/repo) y
> `spec-001` (correcciones DL) en un solo documento.
> **Objetivo global:** llevar el código de entrenamiento a la estructura de paquete de
> `01-Modelo-ITM/`, con Python 3.12, entorno virtual y repositorio git/GitHub propio, y
> **luego** corregir sobre esa estructura los errores de correctitud científica, los bugs
> de acoplamiento y las omisiones de buenas prácticas de Deep Learning detectados en la
> revisión — sin cambiar la arquitectura del modelo (`PhysicalFNOArchitecture`) salvo
> donde se indique explícitamente.

**El spec tiene dos partes secuenciales:**
- **Parte A — Migración e infraestructura** (Fases 1–5): mueve el código sin cambiar su
  comportamiento y monta entorno + repo.
- **Parte B — Correcciones sobre la nueva estructura** (Fases 6–10): aplica los arreglos.

⚠️ **La Parte A debe completarse antes de la Parte B.** Las correcciones referencian los
módulos de `src/modelo_itm/` que la migración crea. Las referencias `train_dataset.py:NNN`
que aparecen en los hallazgos son los anclajes originales (traducidos al módulo destino
según el "Mapa de módulos" de la Fase 2).

---

## 0. Contexto y decisiones ya tomadas

El modelo predice la evolución espacio-temporal del **Factor de Seguridad (SF)** y la
**Deformación Volumétrica (VD)** por capa. El pipeline `cmg2tensor` transforma salidas
CMG `.txt` → tensores `.pt` y el script de entrenamiento consume esos tensores.

Decisiones confirmadas con el usuario en sesiones previas:

| Decisión | Valor |
|---|---|
| Alcance de la migración | Solo el **código de entrenamiento** (`Codigo Entrenamiento/train_dataset.py`) → `01-Modelo-ITM/src/modelo_itm/`. Papers, `.pptx` y `.docx` **no** se mueven. |
| Relación git con `cmg2tensor` | Repositorio **completamente independiente**; `01-Modelo-ITM` solo lo referencia por ruta relativa (`--data-root ../cmg2tensor/data/processed`). Sin submódulo ni fusión de historial. |
| Versión de Python | **3.12** (`requires-python = ">=3.12,<3.13"`) — soportado por PyTorch 2.12.x; ecosistema (`numpy`, `pandas`, `scikit-learn`, `mysql-connector-python`, `pyodbc`) maduro; se evita 3.13/3.14 (demasiado nuevas para dependencias de BD) y 3.9/3.10 (innecesariamente antiguas). |
| Gestor de entornos | `pip` + `venv` — **no conda**, ya establecido en `CLAUDE.md`. |
| Estructura destino | Ya creada como scaffold vacío: `src/modelo_itm/{data,models,training,inference,visualization,utils}`, `scripts/`, `tests/{unit,integration}`, `configs/`, `outputs/{checkpoints,logs,figures}`, `docs/`, `specs/`, `pyproject.toml`, `pytest.ini`, `.gitignore`, `README.md`. |

**Estado verificado del ecosistema (previo a este spec):**
- `09-Proyecto-Deep-Learning/` (raíz) **no** es repositorio git.
- `01-Modelo-ITM/` **no** es repositorio git (aún sin `git init`).
- `cmg2tensor/` **sí** es repositorio git independiente, remoto
  `https://github.com/Reinaldo-06/cmg2tensor.git`, rama `main`.
- `Codigo Entrenamiento/` contiene únicamente `train_dataset.py` (sin checkpoints ni
  outputs que respaldar — bajo riesgo de pérdida de datos al migrar).
- Python en el sistema: 3.9.6 (`/usr/bin/python3`), 3.13.11 (conda `base`), 3.14.6
  (Homebrew). **Ninguno es 3.12** — debe instalarse.
- `gh` (GitHub CLI) **no está instalado**.
- Inconsistencia documentada (fuera de alcance): el `README.md` de `cmg2tensor` indica
  `pip install -e .` pero ese repo no tiene `pyproject.toml`/`setup.py`. No se corrige aquí.

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

# PARTE B — Correcciones sobre la nueva estructura

> Cada fase va en su propia rama `bug/`, `feature/` o `exp/` desde `development` y no se
> cierra hasta pasar revisión (`@reviewer`) y pruebas (`@tester`). Las rutas apuntan a los
> módulos de `src/modelo_itm/` creados en la Parte A; entre paréntesis, el anclaje original
> en `train_dataset.py` para trazabilidad.

## B.0 — Hallazgos de la revisión

### 🔴 Críticos — correctitud / validez científica

#### C1. `test/` se guarda **sin normalizar** y se usa como set de validación
- **Dónde:** `cmg2tensor/src/cmg2tensor/pipeline/parallel.py:194`
  (`normalize_this = normalize and split != "test"`), `cli.py:402`
  (`if use_split_routing and split == "test": normalize_this_sim = False`). El
  entrenamiento usa `val_dir="test"` (`src/modelo_itm/config.py` ← `train_dataset.py:31`).
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

#### C2. La incertidumbre MC Dropout es **código muerto** (siempre cero)
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

#### C3. Métricas R² y RMSE promediadas por batch (estadísticamente incorrectas)
- **Dónde:** `evaluate_epoch` (`src/modelo_itm/training/loop.py` ←
  `train_dataset.py:798-847`) acumula `sf_r2`, `vd_r2`, `sf_rmse`, `vd_rmse` por batch y
  luego promedia (`finalize_running_stats` en `training/metrics.py`).
- **Problema:** el R² **no es aditivo**; promediar R² por batch ≠ R² global. El RMSE
  promediado (media de `sqrt(mse_batch)`) tampoco equivale al RMSE global, y con el último
  batch de tamaño distinto el sesgo crece. Las métricas del paper serían incorrectas.
- **Corrección esperada:** acumular globalmente `ss_res`/`ss_tot` para R², y suma de errores
  cuadrados + conteo para RMSE (`sqrt(sum_se / n)`), calculando la métrica una sola vez al
  final de la época. Las *loss* sí pueden promediarse por batch.

### 🟠 Altos — bugs / acoplamiento

#### A1. Índices temporales hardcodeados en la loss (acoplados a `time_steps=61`)
- **Dónde:** `compute_loss_terms` (`src/modelo_itm/training/losses.py` ←
  `train_dataset.py:685-689`): `pred[:, 0:1, 0]`, `pred[:, 1:21, 0]`, `pred[:, 21:61, 0]`.
- **Problema:** los segmentos `t0 / t1:20 / t21:60` y sus pesos asumen exactamente 61
  timesteps. Con otro `cfg.time_steps` los slices dejan de cubrir el tensor (p. ej.
  `time_steps=30` → `21:61` parcial), produciendo una loss silenciosamente incorrecta.
- **Corrección esperada:** derivar los límites de `cfg.time_steps` (o parametrizar los
  cortes en `Config` como fracciones/índices validados) y añadir una aserción de que los
  segmentos cubren exactamente `[0, time_steps)` sin solapamientos.

#### A2. Normalización de inyección doble / inconsistente
- **Dónde:** el ETL normaliza con stats globales
  (`normalize_series_minmax_with_global_stats`), pero `load_injection_series`
  (`src/modelo_itm/data/dataset.py` ← `train_dataset.py:499-521`) vuelve a aplicar
  `log1p` + normalización por `max` local por muestra.
- **Problema:** doble transformación no documentada; la escala efectiva depende del ETL.
  Dificulta reproducibilidad e interpretación física.
- **Corrección esperada:** un único punto de normalización de la inyección (ETL **o**
  dataset, no ambos), documentado. Si se mantiene en el dataset, consumir la serie cruda del
  ETL de forma consistente.

#### A3. Evaluación redundante y costosa por época
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

### 🟡 Medios — buenas prácticas de Deep Learning

- **M1. Sin scheduler de LR.** `lr` constante `8e-4` con `AdamW`. Añadir
  `CosineAnnealingLR`/`OneCycleLR`/warmup+decay; guardar el estado del scheduler en el
  checkpoint (`training/checkpoint.py`) para que `--auto-resume` sea consistente.
- **M2. Sin *mixed precision* (AMP).** El forward expande `z` a `(b·T, h_dim, H, W)`
  (`models/fno.py` ← `train_dataset.py:667`), muy pesado en memoria. Integrar
  `torch.autocast` + `GradScaler`; cuidar las FFT (`FiLMSpectralBlock`) que suelen requerir
  `float32`.
- **M3. Sin normalización interna en la red.** No hay `LayerNorm`/`GroupNorm` en encoder,
  bloques FNO ni decoder. Evaluar añadirla para estabilidad.
- **M4. Reproducibilidad incompleta.** `cudnn.benchmark=True` (no determinista), sin
  `worker_init_fn` para semillar workers del `DataLoader`, `random.randrange` en
  `save_epoch_visuals`. Añadir `worker_init_fn`, `generator` semillado y documentar el
  trade-off determinismo/rendimiento (o exponer flag `deterministic`).
- **M5. Logging por `print`.** `CLAUDE.md` §Convenciones pide evitar `print` para
  diagnóstico permanente. Migrar a `logging`/`tqdm.write` y, dado el paper, considerar
  TensorBoard o Weights & Biases.
- **M6. Sin guardas de NaN/Inf** en loss o gradientes (solo hay `nan_to_num` en la
  inyección). Añadir detección temprana de `NaN`/`Inf` en la loss para abortar controlado.
- **M7. `weight_decay` sobre embeddings y bias.** `AdamW` decae todos los parámetros,
  incluidos `t_embed`, `gamma/beta` de FiLM y bias. Separar param groups para no penalizar
  embeddings/bias.
- **M8. Sin tests de lógica del entrenamiento.** Más allá de los tests mínimos de la
  Fase 2b, añadir cobertura real: `DatasetLayers` (shapes, padding de inyección),
  `compute_loss_terms` (segmentación correcta para varios `time_steps`), métricas globales
  (C3), forward del modelo.

### 🔵 Bajos — menores / higiene

- **B1.** Estado global mutable de "emitir una vez" (`_MC_DROPOUT_WARNING_EMITTED`,
  `_CUDA_BATCH_REPORT_EMITTED`, reubicados en la Fase 2): encapsular en un objeto de estado
  o `logging` con filtro.
- **B2.** `torch.load(..., weights_only=False)` en `training/checkpoint.py` (← `:459`) y
  `load_pt` sin `weights_only` en `data/dataset.py` (← `:488`): revisar compatibilidad con
  PyTorch ≥2.6 (cambio de default) y seguridad al cargar `.pt` de terceros.
- **B3.** Min-max sensible a outliers; evaluar clip de percentiles (p1/p99) o *robust
  scaling* antes de escalar (`cmg2tensor/normalize.py`).
- **B4.** Caché de inyección (`self._inj_cache` en `data/dataset.py` ← `:528`) sin cota
  superior (impacto bajo).

---

## B.1 — Fases de corrección

### Fase 6 — Correctitud crítica de datos y métricas (bloqueante)
- **C1**: normalizar `test/` con stats globales de train (o rediseñar validación). Archivos:
  `cmg2tensor/src/cmg2tensor/pipeline/parallel.py`, `.../cli.py`, `.../pipeline/serial.py`;
  regenerar `data/processed/test/` (**⚠️ requiere confirmación** antes de reprocesar datos).
- **C3**: acumulación global de R²/RMSE. Archivos: `src/modelo_itm/training/metrics.py`,
  `training/loop.py` (`evaluate_epoch`).
- **Validación:** una corrida corta muestra `val_loss` en la misma escala que `train_loss` y
  R²/SF/VD coherentes.

### Fase 7 — Bugs de acoplamiento
- **A1** (segmentos de loss derivados de `time_steps` → `training/losses.py`, `config.py`),
  **A2** (normalización única de inyección → `data/dataset.py` + contrato en `cmg2tensor`),
  **A3** (evaluación/calibración redundante → `training/loop.py`).

### Fase 8 — Feature de incertidumbre (decisión de diseño)
- **C2** según Opción A o B (**requiere `AskUserQuestion`** sobre el rumbo). Archivos:
  `src/modelo_itm/models/{fno,blocks}.py`, `config.py`, `inference/uncertainty.py`,
  actualización de `CLAUDE.md` si cambia el alcance.

### Fase 9 — Buenas prácticas de entrenamiento
- **M1** (scheduler + estado en checkpoint), **M2** (AMP), **M6** (guardas NaN/Inf),
  **M7** (param groups), **M5** (logging/experiment tracking). Archivos:
  `src/modelo_itm/training/{loop,checkpoint}.py`, `utils/`, `config.py`.

### Fase 10 — Robustez, reproducibilidad y tests
- **M3** (normalización interna, experimental → rama `exp/`), **M4** (reproducibilidad),
  **M8** (tests completos), menores **B1–B4**. Archivos: `src/modelo_itm/**`,
  `tests/unit/`, `tests/integration/`, `cmg2tensor/normalize.py`, `cmg2tensor/tests/`.

---

## 1. Archivos impactados (resumen consolidado)

| Archivo / carpeta | Fases | Naturaleza |
|---|---|---|
| `01-Modelo-ITM/pyproject.toml` | 1 | Fijar `requires-python` |
| `01-Modelo-ITM/.venv/` | 1 | Nuevo, ignorado por git |
| `01-Modelo-ITM/src/modelo_itm/**` | 2, 6–10 | Código migrado + correcciones |
| `01-Modelo-ITM/scripts/train.py` | 2 | Nuevo entrypoint CLI |
| `01-Modelo-ITM/tests/**` | 2b, 10 | Tests de migración + cobertura real |
| `01-Modelo-ITM/.git/` | 3 | Nuevo repositorio local |
| Remoto GitHub (`01-Modelo-ITM`) | 4 | Nuevo recurso externo |
| `Codigo Entrenamiento/train_dataset.py` | 5 | Eliminado tras migración verificada |
| `01-Modelo-ITM/CLAUDE.md` | 1, 5, 8 | Comandos, estructura, alcance de incertidumbre |
| `01-Modelo-ITM/README.md` | 5 | Estado del proyecto |
| `cmg2tensor/src/cmg2tensor/pipeline/parallel.py` | 6 | Normalización de `test/` |
| `cmg2tensor/src/cmg2tensor/cli.py` | 6 | Ruteo de normalización por split |
| `cmg2tensor/src/cmg2tensor/pipeline/serial.py` | 6 | Aplicación de normalización |
| `cmg2tensor/src/cmg2tensor/normalize.py` | 10 | Robust scaling (opcional) |
| `cmg2tensor/tests/` | 10 | Nuevos tests |
| `data/processed/test/` | 6 | Reproceso (**⚠️ requiere confirmación**) |

---

## 2. Riesgos y precondiciones

- **Orden obligatorio:** la Parte B referencia módulos que crea la Parte A. No iniciar
  correcciones antes de completar la migración y verificar equivalencia.
- **Sin GPU/datos en esta sesión:** la equivalencia numérica de la Fase 2 y la validación de
  la Fase 6 requieren acceso a `cmg2tensor/data/processed` y, idealmente, GPU. Documentar si
  quedan pendientes.
- **`gh` CLI no instalado:** la Fase 4 requiere instalarlo o usar la web de GitHub.
- **Cuenta/organización de GitHub sin confirmar:** no crear el repo remoto sin que el usuario
  indique cuenta, visibilidad y nombre.
- **Reproceso de datos (C1):** regenerar `data/processed/test/` toca datos derivados;
  respaldar antes y **no ejecutar sin confirmación explícita**. No sobrescribir
  `reports/global_normalization/train.json`.
- **Compatibilidad de checkpoints:** cambiar `Config`, el modelo (C2/M3) o añadir scheduler
  (M1) altera `run_signature`/el `state_dict`; los checkpoints previos quedarán
  incompatibles con `--auto-resume`. Documentarlo y respaldar `best.pt`/`latest.pt` antes.
- **Decisión pendiente (C2):** definir con el usuario Opción A vs B antes de la Fase 8.
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
- [ ] `Codigo Entrenamiento/` ya no existe y no quedan referencias rotas a esa ruta en
      `CLAUDE.md`, `README.md` ni `specs/`.

**Parte B — Correcciones:**
- [ ] `val_loss` y métricas de validación en la misma escala que train (C1).
- [ ] R² y RMSE calculados globalmente y verificados contra un cómputo de referencia (C3).
- [ ] La loss es correcta para al menos dos valores de `time_steps` distintos (A1).
- [ ] La incertidumbre está operativa **o** aislada tras flag y documentada (C2).
- [ ] Existe cobertura de tests para dataset, loss, métricas globales y forward del modelo
      (M8) y `pytest -m "not slow"` pasa.
- [ ] `CLAUDE.md` refleja el alcance real de la feature de incertidumbre y la nueva
      estructura de módulos.
