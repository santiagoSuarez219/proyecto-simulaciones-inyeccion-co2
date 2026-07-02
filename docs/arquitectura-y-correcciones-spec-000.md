# Arquitectura del pipeline y correcciones del spec-000

> **Alcance:** este documento describe (1) la arquitectura original propuesta en
> `Codigo Entrenamiento/train_dataset.py` y `cmg2tensor/`, (2) los hallazgos de la
> revisión de código y la ejecución del `spec-000` (críticos, altos, medios, bajos, con
> su consecuencia y la solución aplicada), y (3) la arquitectura resultante tras ejecutar
> el spec.
>
> **Fuente:** `specs/spec-000-migracion-y-correcciones-fno-co2.md` (estado `[TESTING]`,
> 108/108 tests). **Fecha:** 2026-07-02.

---

## 1. Arquitectura original (antes del spec-000)

El proyecto **Modelo-ITM** es un modelo de deep learning para **predicción
espacio-temporal del Factor de Seguridad (SF) y la Deformación Volumétrica (VD)** en
reservorios depletados bajo inyección de CO₂ (almacenamiento geológico de carbono, CCS).
Originalmente vivía en dos piezas desacopladas:

- **`cmg2tensor/`** — repositorio git independiente (de Reinaldo-06) con el **ETL**:
  convierte salidas del simulador CMG (`.txt`) + `inyeccion.xlsx` en tensores `.pt`.
- **`Codigo Entrenamiento/train_dataset.py`** — un **único archivo monolítico de 1.432
  líneas** con todo el código de entrenamiento (modelo, dataset, losses, métricas, loop,
  inferencia, visualización).

### 1.1 El ETL `cmg2tensor` (20 módulos)

Transforma, por simulación, archivos crudos → tensores por capa normalizados globalmente,
en un patrón de **dos fases**.

```
data/raw/{train,test}/<sim>/
   SF.txt, VD.txt (obligatorios),
   Permeability/Porosity/Cohesion/AFI/Pressure*.txt, inyeccion.xlsx
        │
 [Pre-ETL]  make_split / simulations_dataset → split estratificado 80/20 (CSV)
            standardize_formats → renombra a nombres canónicos
        │
 [Discovery] discovery.py → mapea cada archivo a su rol por heurística de tokens del nombre
        │
 [Phase 1]  stats.py (SOLO split=train) → min/max por variable
            → reports/global_normalization/train.json
        │
 [Phase 2]  pipeline/serial.py o parallel.py:
              parse_txt streaming (.txt → tensor 4D) → corte por capa K
              normalize.py: (x-min)/span con train.json
              alinea inyección al eje temporal de SF
        │
data/processed/{train,test}/<sim>/
   layer_cubes/layer_cube_kXXX.pt             ← cubos SF+VD, shape (V=2, T, NJ, NI)
   {permeability,porosity,...}_layer_cubes/   ← shape (1, T, NJ, NI)
   injection_name_tensors/injection_tene_{1,2}.pt   ← series (T,)
   layer_cubes_report.json, timeline.json, ...
```

**Piezas clave:**

| Módulo | Rol |
|---|---|
| `parse_txt.py` | Parser CMG *streaming* (O(1) memoria): cuenta cabeceras `** TIME =` para descubrir T y `** K=k, J=j` para capas/filas; produce tensor `(T, NZ, NJ, NI)` cortado por capa. `NZ/NJ/NI` son parámetros (default 20/100/100). |
| `stats.py` (Phase 1) | Busca líneas `RESULTS PROP ... Minimum/Maximum Value` para min/max sin cargar el tensor ("header path"); fallback parseando el tensor completo. **Solo escanea `train`** → sin fuga de datos. |
| `normalize.py` (Phase 2) | Min-max `(x-min)/span` por variable con los `global_stats` de train. |
| `pipeline/serial.py` vs `parallel.py` | Misma lógica de transformación; `parallel.py` la ejecuta con `ProcessPoolExecutor` en contexto *spawn* (workers a nivel de módulo, picklables), aislando fallos por simulación y soportando `--retry-failed`. |
| `discovery.py` | Mapea archivo → rol por tokens del nombre; lanza `ValueError` si faltan SF o VD. |
| Series de inyección | `inyeccion.xlsx` (hoja "Well Summary", pozos `TENE-1`/`TENE-2`, parámetro `"Gas Rate SC - Monthly"`): día→mes, alineada al eje temporal de SF (`timeline.json`), normalizada. Alimenta `inj: (B, T, 2)`. |

**Salida canónica** `layer_cube_kXXX.pt`: dict con `cube (V,T,NJ,NI)`, `time_ids` (meses),
`time_ids_days`, `variables`, `layer_k` y bloque `normalization`.

### 1.2 El modelo — `PhysicalFNOArchitecture`

Un **Fourier Neural Operator (FNO)** con condicionamiento **FiLM** (no un U-Net, pese al
título del paper en redacción). Mapea propiedades estáticas del reservorio + tasas de
inyección temporales → evolución 2D por capa de SF y VD.

```
Entrada por muestra (una capa k):
  x   : (B, 4, H, W)   — 4 propiedades estáticas: AFI, COH, PERM, PORO
  d   : (B, 1)          — profundidad normalizada de la capa: (k-1)/96
  inj : (B, T, 2)       — series de inyección TENE-1 y TENE-2

Encoder:
  cat([x, depth_map]) → (B, 5, H, W)          # ← in_c=5 (4 + 1 profundidad)
  Conv2d(5→128) + GELU + ResBlock(128)

Condicionamiento temporal (FiLM):
  t_embed  : Embedding(T=61, 128)              # embedding por timestep
  cond_mlp : Linear(3→128) → GELU → Linear(128→128)   # proyecta [inj_t1, inj_t2, depth]
  cond_seq = t_emb + cond_mlp(cond_input)

FNO Blocks (× 4) — FiLMSpectralBlock:
  FFT2 → multiplicación espectral (16 modos truncados) → iFFT2
  + Conv2d 1×1 local → GELU
  + modulación FiLM: y·(1+γ(cond)) + β(cond)

Decoder:
  ResBlock(128) → Conv2d(128→64) → GELU → Conv2d(64→2)

Salida:
  (B, T=61, 2, H, W)   — SF y VD para los T pasos temporales
```

**Función de pérdida** (`compute_loss_terms`): SmoothL1 + pérdida de gradiente espacial,
con SF segmentado en 3 tramos temporales ponderados (`t0` / `t1:20` / `t21:60`) y VD como
un único término:

```
loss = sf_weight·L_SF + vd_weight·L_VD    (2.5 / 1.0)
L_SF = Σ seg_weight · [SmoothL1(SF_seg) + grad_weight·grad_loss(SF_seg)]
L_VD = SmoothL1(VD) + grad_weight·grad_loss(VD)
```

**Dataset `DatasetLayers`**: recorre `data/processed/<caso>/`, empareja los 4 cubos
estáticos + el target + las series de inyección por capa (k=1..97), y devuelve
`(x, depth, inj, y)`.

**Features anunciadas**: MC Dropout para incertidumbre, auto-resume por checkpoint,
auto-pausa por hora, early stopping, visualizaciones PNG por época.

---

## 2. Hallazgos de la revisión y ejecución del spec-000

El spec-000 tuvo **dos objetivos secuenciales**:

- **Partes A y A2 — Migración quirúrgica**: mover `train_dataset.py` (monolito) a un
  paquete `src/fno_co2/` modular (16 módulos) y plegar `cmg2tensor` como subpaquete
  `src/fno_co2/etl/` (copia limpia sin historial, imports `cmg2tensor.*` →
  `fno_co2.etl.*`). Comportamiento idéntico, solo reorganizado.
- **Parte B — Correcciones** de correctitud científica, bugs de acoplamiento y buenas
  prácticas de DL, sobre la nueva estructura.

La revisión clasificó **17 hallazgos**. Todos quedaron corregidos en código, con
**108/108 tests pasando** (incl. `slow`). El spec está en `[TESTING]` (no `[DONE]`) porque
3 verificaciones requieren GPU/datos reales no disponibles en la sesión: **C1** (regenerar
`test/` real), **M2** (path real de AMP en CUDA) y **M3** (`use_group_norm` en
entrenamiento completo).

### 🔴 Críticos — correctitud / validez científica

| ID | Problema | Consecuencia | Solución aplicada |
|---|---|---|---|
| **C1** | `test/` se guardaba **sin normalizar** (unidades físicas crudas), pero se usa como set de **validación**. El modelo entrena en `[0,1]` y valida contra otra escala. (`etl/pipeline/parallel.py:194`, `etl/cli.py:402`) | `val_loss`, RMSE y R² de validación **no comparables** con train → early stopping y selección de `best.pt` guiados por una señal **inválida**. | Removido el forzado `normalize=False` para `split=="test"` en ambos pipelines. El `global_stats` (calculado solo de train, sin fuga) ya se pasaba a todos los splits. Test se normaliza ahora con las mismas stats de train. **Verificado con test sintético; falta reprocesar `test/` real (requiere GPU/datos).** |
| **C2** | El modelo **no tenía ninguna capa `nn.Dropout`**, así que `model_has_dropout()` era siempre `False` y toda la maquinaria de MC Dropout hacía short-circuit. | Incertidumbre = **0.0** y confianza = **1.0 siempre** (código muerto). Contradecía CLAUDE.md y el paper: se pagaba complejidad/cómputo sin señal. | **Opción A** (confirmada con el usuario vía `AskUserQuestion`): `nn.Dropout2d(dropout_p)` insertado en `ResBlock` (encoder y decoder). Nuevo `Config.dropout_p=0.1`, propagado y expuesto como `--dropout-p`; incluido en `run_signature`. MC Dropout ahora produce `pred_std.max() > 1e-4` real. |
| **C3** | R² y RMSE de validación **promediados por batch** (`torch_r2_score` por batch → media). (`train_dataset.py:702-703`) | R² **no es aditivo**: promediar por batch ≠ R² global; el RMSE promediado y el último batch parcial sesgan más. **Las métricas del paper serían incorrectas.** | Acumulación **global** en `training/metrics.py`: `sum_sq_error`, `sum_gt`, `sum_gt_sq`, `count` por variable a través de todos los batches; R² con la identidad `SS_tot = Σgt² − n·mean²` (sin segunda pasada), RMSE = `sqrt(ΣSE/n)`, calculados una sola vez al final. Verificado contra `sklearn.metrics.r2_score`. |
| **C4** | **Bug introducido por la propia migración**: al escribir tests sintéticos mal construidos (x con 5 canales) se "corrigió" el encoder a `Conv2d(in_c+1, …)`. Pero el dataset real da 4 canales → `4+1=5=in_c`, y el encoder esperaba 6. | El mismatch de shape **bloqueaba TODO entrenamiento con datos reales desde la Fase 2**; nunca se detectó porque no había test end-to-end con el dataset real. | Revertido a `Conv2d(in_c, …)` (como el original); docstring aclarando que `in_c` es el total post-concatenación. Corregidos 6 archivos de test. **Descubierto** al escribir un smoke test end-to-end con `DatasetLayers` real. |

### 🟠 Altos — bugs / acoplamiento

| ID | Problema | Consecuencia | Solución |
|---|---|---|---|
| **A1** | Índices temporales **hardcodeados** en la loss: `pred[:, 1:21, 0]`, `pred[:, 21:61, 0]` — asumen exactamente 61 timesteps. (`train_dataset.py:687-688`) | Con otro `time_steps` los slices no cubren el tensor → **loss silenciosamente incorrecta**. | `_segment_boundaries(time_steps)` deriva `(b1,b2)` escalando proporcionalmente sobre 61 (reproduce `(1,21)` con 61); usa `pred.shape[1]` real y afirma `0 < b1 < b2 < time_steps`. |
| **A2** | **Doble normalización** de inyección: el ETL ya normaliza con stats globales, pero `load_injection_series` volvía a aplicar `log1p` + reescalado por máximo local. (`train_dataset.py:518`) | Escala efectiva dependiente de cada muestra individual → **irreproducible y físicamente no interpretable**. | Punto único de normalización (**solo el ETL**). `load_injection_series` ya no aplica `log1p`/reescalado; solo alinea longitud y sanea NaN/Inf. |
| **A3** | Evaluación redundante: cada época hacía `evaluate_epoch(train)` **además** de `evaluate_epoch(val)`, + recalibración de incertidumbre incondicional. (`train_dataset.py:1348-1349`) | **~50% de cómputo extra** por época (un forward completo sobre todo train) para métricas que podían acumularse; calibración inútil sin dropout. | Métricas de train acumuladas **dentro de `run_one_epoch`** (reusa el `pred/y` de la loss, sin forward extra). Recalibración tras guard `if model_has_dropout(model)`. |
| **A4** | Import roto **preexistente** en el ETL original: `from ..pipelines import …` (plural, módulo inexistente) dentro del worker de Phase 2. | El **pipeline paralelo nunca produjo un solo archivo exitoso** (fallo enmascarado: import diferido en proceso *spawn*, y sus tests marcados `slow`). | `from ..pipelines import …` → `from .serial import …`. **Descubierto** al escribir los tests de C1. |

### 🟡 Medios — buenas prácticas de Deep Learning

| ID | Problema / Consecuencia | Solución |
|---|---|---|
| **M1** | LR constante `8e-4` sin scheduler → convergencia subóptima. | `training/optim.py::build_scheduler` (`CosineAnnealingLR`, `T_max=epochs`, `eta_min=lr_min`); estado guardado en checkpoint; `scheduler.step()` tras guardar. `--lr-scheduler`/`--lr-min`. |
| **M2** | Sin mixed precision; el forward expande `z` a `(b·T, h_dim, H, W)`, muy pesado en memoria. | `Config.use_amp` (opt-in, solo CUDA); `autocast` + `GradScaler` en `run_one_epoch`/`evaluate_epoch`. La FFT de `FiLMSpectralBlock` se fuerza a **float32** con `autocast(enabled=False)`. **Path real en CUDA no ejercido (sin GPU).** |
| **M3** | Sin normalización interna (LayerNorm/GroupNorm) → posible inestabilidad. | `ResBlock` acepta `use_group_norm` (GroupNorm tras cada Conv2d), con `_group_norm_num_groups` que evita divisores inválidos. **EXPERIMENTAL, `Config.use_group_norm=False` por defecto** (no cambia la arquitectura default). No verificado en entrenamiento real. |
| **M4** | Reproducibilidad incompleta: `cudnn.benchmark=True`, sin semillado de workers, `random.randrange` global en visualización. | `resolve_device(deterministic=)`; `worker_init_fn`+`generator` semillados **siempre** en `build_loader`; RNG local `random.Random(seed+epoch)` en `save_epoch_visuals`. `--deterministic`. |
| **M5** | Logging por `print` (viola convenciones de CLAUDE.md). | `utils/logging.py` con handler que enruta a `tqdm.write()`; los 23 `print()` migrados a `logging`. Verificado con `grep`. |
| **M6** | Sin guardas de NaN/Inf en loss/gradientes. | `run_one_epoch` lanza `RuntimeError` explícito si la loss o la norma de gradiente no son finitas (antes de corromper el modelo). |
| **M7** | `weight_decay` aplicado a embeddings, bias y γ/β de FiLM. | `build_param_groups` separa `decay` / `no_decay` (bias, `t_embed`, γ/β con `weight_decay=0.0`). |
| **M8** | Sin tests de la lógica de entrenamiento. | Cobertura real: dataset (padding de inyección), loss (varios `time_steps`), métricas globales vs sklearn, forward. **Suite final: 108 tests, 0 fallos.** |

### 🔵 Bajos — higiene

| ID | Problema | Solución |
|---|---|---|
| **B1** | Estado global mutable `_XXX_EMITTED` para "emitir una vez". | Clase `EmitOnce` (`should_emit(key)`/`reset`), sin `global`. |
| **B2** | `torch.load(weights_only=False)` — riesgo con PyTorch ≥2.6. | `weights_only=True` explícito en `checkpoint.py` y `load_pt` (verificado round-trip). |
| **B3** | Min-max sensible a outliers. | `clip_percentiles(p1/p99)` disponible en `etl/normalize.py`, **no integrado** (cambiar el default requeriría reprocesar datos). |
| **B4** | Caché de inyección sin cota. | `OrderedDict` LRU con `max_inj_cache_size=512`. |
| **B5** | Tests desactualizados en `histograms.py` (comparaban `keys()` sin filtrar `_meta`) — **preexistente**, falso positivo. | Tests usan `_variable_keys(result)`. Resuelto el último hallazgo → **108 passed**. |

---

## 3. Arquitectura resultante (después del spec-000)

El resultado es un **paquete Python único e instalable** (`fno_co2`, editable vía
`pyproject.toml`, Python 3.12) que contiene **todo el pipeline** —ETL → modelo →
entrenamiento → inferencia → visualización— reemplazando el monolito de 1.432 líneas y el
repo `cmg2tensor` externo.

```
01-Modelo-ITM/                         # repo git propio (main + development)
├── src/fno_co2/
│   ├── config.py                      # dataclass Config (única fuente de hiperparámetros)
│   ├── etl/                           # ← ex-cmg2tensor, plegado (imports fno_co2.etl.*)
│   │   ├── cli.py, orchestrator.py, discovery.py, parse_txt.py,
│   │   │   build_tensors.py, normalize.py, stats.py, histograms.py, config.py
│   │   ├── pipeline/{serial,parallel}.py
│   │   └── utils/{apply_train_test_split, standardize_formats, ...}
│   ├── data/{dataset,loaders}.py      # DatasetLayers, build_loader (worker_init_fn/generator)
│   ├── models/{fno,blocks}.py         # PhysicalFNOArchitecture, ResBlock(+Dropout2d/GroupNorm), FiLMSpectralBlock
│   ├── training/{loop,losses,metrics,checkpoint,optim}.py   # + optim.py nuevo
│   ├── inference/uncertainty.py       # MC Dropout (ahora funcional)
│   ├── visualization/plots.py
│   └── utils/{device,io,time,logging}.py    # + logging.py nuevo (get_logger, EmitOnce)
├── scripts/{train.py, etl/*}          # entrypoints delgados
├── sql/, tests/{unit,etl,integration}/, configs/, outputs/, docs/, specs/
```

**Cambios estructurales frente a la propuesta original:**

1. **Modularidad**: el monolito se descompone en 16 módulos temáticos (`config`, `data`,
   `models`, `training`, `inference`, `visualization`, `utils`), ninguno >~300 líneas.
   `scripts/train.py` es un entrypoint delgado que llama a `training.loop.main()`.

2. **Pipeline unificado**: el ETL ya no es un repo externo referenciado por ruta; vive en
   `src/fno_co2/etl/`. `python -m cmg2tensor` → `python -m fno_co2.etl`. El repo de
   Reinaldo-06 y `Codigo Entrenamiento/` quedan **solo como referencia** (no se borran).

3. **El modelo evoluciona levemente** (única desviación arquitectónica autorizada):
   `ResBlock` ahora incluye `nn.Dropout2d(dropout_p=0.1)` — habilitando **MC Dropout
   real** — y opcionalmente `GroupNorm` (experimental, off). Encoder `Conv2d(5→128)`
   correcto (in_c=5 = 4 estáticas + profundidad). El resto del FNO/FiLM es idéntico.

4. **Bucle de entrenamiento corregido y enriquecido**:
   - Métricas R²/RMSE **globales** (no promedio por batch), acumuladas durante el propio
     paso de train (sin forward extra).
   - `AdamW` con **param groups** (sin decay en bias/embeddings/FiLM) + **`CosineAnnealingLR`**
     con estado en checkpoint.
   - **AMP opcional** (autocast + GradScaler, FFT forzada a float32).
   - **Guardas NaN/Inf**, calibración de incertidumbre solo con dropout activo, logging vía
     `tqdm.write`, reproducibilidad opcional (`--deterministic`, workers semillados).

5. **ETL corregido**: `test/` se normaliza con stats de train (C1), import paralelo
   reparado (A4), inyección normalizada **una sola vez** en el ETL (A2), utilidad
   `clip_percentiles` disponible (B3).

6. **Testing**: de cero tests de entrenamiento a **108 tests** (`unit/` + `etl/` +
   `integration/`), incluyendo smoke tests end-to-end con el formato real de datos.

### Estado de verificación (pendientes de hardware/datos reales)

Estos 3 puntos marcan el estado `[TESTING]` en vez de `[DONE]`:

- **C1**: regenerar `data/processed/test/` con datos reales y confirmar `val_loss` en la
  misma escala que `train_loss`.
- **M2**: ejercitar el path real de float16/GradScaler en CUDA.
- **M3**: probar `use_group_norm` en un entrenamiento completo.

---

## 4. Nota de mantenimiento — normalización de `test/` (post-C1)

Antes del spec-000, el ETL **no** normalizaba `test/` (hallazgo C1). Tras la corrección,
**`test/` se normaliza con las mismas estadísticas globales de train** (sin fuga: las
stats provienen solo de `train/`), de modo que train y validación quedan en la misma
escala `[0,1]`. Cualquier documentación que afirme que "las simulaciones `test/` nunca se
normalizan" es **anterior a C1** y debe leerse como comportamiento histórico, no actual.
`CLAUDE.md` fue actualizado para reflejar esto.
