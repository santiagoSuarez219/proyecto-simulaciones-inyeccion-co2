# spec-004 — Orquestación de campañas de experimentos (matriz arquitectura × seeds) [IN PROGRESS]

> **Autor:** rol `@architect`
> **Fecha:** 2026-07-02 · **Actualizado:** 2026-07-16 (revisión contra el estado real del
> repo tras cerrar `spec-001`/`spec-002`/`spec-003`; Fases 0–6 completadas)
> **Estado:** `[IN PROGRESS]` — Fases 0–6 completas y verificadas en rama
> `feature/campaign-orchestration`: precondiciones, esquema+preflight, reproducibilidad
> (conectada a la ejecución real), runner con resume, tracking, agregación+reporte
> cross-arquitectura, y un test de integración real (CPU, 3 seeds, 1 época, overfit) que
> corrió de punta a punta sin mocks. Solo queda **Fase 7 — ejecución real de la campaña
> completa** (múltiples arquitecturas × ≥3 seeds, datos completos, GPU): **requiere
> confirmación explícita del usuario**, no se lanza sola.
> **Depende de:** `spec-001` (framework de experimentación: `--model-variant`,
> `build_model`, loader YAML, `run_experiment.py` multi-seed, `aggregate_experiments.py`,
> `docs/experiments.md`) — **entregado**. Consume las variantes de `spec-002` (U-Net,
> `[DONE]`) y `spec-003` (FNO+atención axial, `[TESTING]`): **ambas ya registradas y con su
> `configs/experiments/<v>.yaml`**, así que la campaña puede correr las 3 (baseline +
> unet_film + fno_axial_attn) desde el arranque, no solo la línea base.
>
> **Relación con el backlog (importante):** ejecutar esta campaña es el **vehículo que cierra
> los pendientes multi-seed abiertos**: `spec-002-debt-002` (Fase 5.3: corrida real ≥3 seeds
> de `unet_film`, hoy con la tabla de `docs/experiments.md` en "pendiente re-run") y
> `spec-003` Fase 5 (multi-seed de `fno_axial_attn`). El diseño **secuencial en 1 GPU** de
> esta campaña está además **validado por `spec-002-debt-001`** (RESUELTO): el OOM que se
> temía era un artefacto de correr **seeds en paralelo**, no un problema por-seed; en
> ejecución secuencial el pico de memoria de `unet_film` es ~3.89 GiB a `batch_size=2`.
> **Objetivo:** una **capa de campaña** que corre de forma **automática** la matriz
> completa `{arquitecturas registradas} × {≥3 seeds}` en **1 GPU, secuencialmente**, con
> **resume idempotente**, **aislamiento de fallos**, **captura de reproducibilidad**
> (git, entorno, checksum del split, snapshots de config), **tracking** (archivos siempre;
> MLflow/W&B opcional) y **agregación estadística cross-arquitectura** — con rigor
> científico (`spec-001` Fase 6) por construcción.

---

## 0. Contexto y relación con `spec-001`

`spec-001` ya resuelve **una arquitectura, N seeds**: `scripts/run_experiment.py` corre N
seeds de **una** config y `scripts/aggregate_experiments.py` agrega esas seeds. Lo que
falta —y es el alcance de este spec— es la **campaña**: correr **muchas arquitecturas ×
muchas seeds** de una sola invocación, de forma automática, reanudable y trazable, y
compararlas **entre sí y contra la línea base** en un único reporte.

**Delimitación estricta (no duplicar `spec-001`):**

| Nivel | Artefacto | De quién |
|---|---|---|
| 1 corrida (arch, seed) | `scripts/train.py` | ya existe / `spec-001` F1 |
| N seeds de 1 arch | `scripts/run_experiment.py` | `spec-001` F4 |
| Agregación de seeds de 1 arch | `scripts/aggregate_experiments.py` | `spec-001` F5 |
| **M archs × N seeds (campaña)** | **`scripts/run_campaign.py`** | **este spec (F3)** |
| **Reproducibilidad + tracking + reporte cross-arch** | **este spec (F2, F4, F5)** | **este spec** |

**Principio rector (hereda de `spec-001` §0):** una campaña solo produce comparaciones
válidas si **todo lo no-arquitectónico está congelado e idéntico** entre corridas: mismos
datos, **mismo split** (verificado por checksum, no por confianza), mismas seeds evaluadas,
mismo criterio de paro, mismas métricas. La campaña **hace cumplir** esto automáticamente
(guardas), en vez de dejarlo a la disciplina manual.

**Qué ya existe (no reinventar) — verificado en el árbol a 2026-07-16:**

- `scripts/run_experiment.py` **ya expande N seeds de una config**, copia el YAML al output
  (reproducible), corre cada seed como subproceso a `train.py`, **aísla el fallo de una seed**
  (una seed caída no aborta las demás) y escribe `outputs/<exp>/run_manifest.json` con el
  estado por seed. → El runner de campaña **debe reutilizarlo por variante** en vez de
  reimplementar el loop de seeds (ver 1.3, decisión de diseño).
- `scripts/aggregate_experiments.py` **ya compara una variante vs. la línea base**: Wilcoxon
  pareado por seed (o Mann-Whitney U si difieren), **tamaño de efecto**, **valores crudos por
  seed**, veredicto con **mínimo 3 seeds** y **chequeo de solapamiento mean±std**, y hace
  *upsert* idempotente de la sección de esa variante en `docs/experiments.md`. → La Fase 5 de
  este spec es **más ligera de lo planeado en 2026-07-02**: se envuelve/itera esta lógica a
  nivel campaña + genera un reporte consolidado; **no reimplementa** la estadística (ver 1.6).
- `build_model` resuelve variantes por **discovery por convención** (`fno_co2.models.variants.
  <name>.build(cfg)`), no por un dict de registro; el preflight (1.2.2) valida importabilidad
  + presencia de `build`, no una entrada en una tabla.

**Escalabilidad = por configuración, no por hardware.** Añadir una arquitectura a una
campaña es: registrar su variante (`spec-001` F3) + su `configs/experiments/<v>.yaml`
(`spec-001` F2) + **una línea** en el YAML de campaña. Cero cambios en el orquestador. La
cola admite M×N arbitrario; en 1 GPU corre secuencialmente y **reanuda** si se interrumpe.

---

## 1. Diseño

### 1.1 Config de campaña — `configs/campaigns/<nombre>.yaml`

Artefacto declarativo y **autocontenido** que define la matriz. Ejemplo conceptual:

```
campaign_name: fno_vs_unet_vs_attn
description: "Comparacion baseline FNO vs U-Net vs FNO+atencion axial"
seeds: [42, 43, 44]                 # >=3 (Fase 6); explicitos o n_seeds -> derivadas det.
variants:                           # cada entrada apunta a una config de spec-001 F2
  - name: baseline
    config: configs/experiments/baseline.yaml
    success_criterion: "referencia (linea base); no se evalua contra si misma"
  - name: unet_film
    config: configs/experiments/unet_film.yaml
    # criterios ya fijados y registrados en docs/experiments.md (spec-002):
    success_criterion:
      metric: val_sf_r2
      op: ">="
      threshold: 0.974              # <=2% bajo baseline 0.9937
      guard: {metric: val_vd_r2, op: ">=", threshold: 0.9430}   # <=2% bajo baseline 0.9626
  - name: fno_axial_attn
    config: configs/experiments/fno_axial_attn.yaml
    # criterio ya fijado y registrado en docs/experiments.md (spec-003):
    success_criterion:
      metric: val_sf_rmse
      op: "<="
      threshold: 0.00864            # reduce >=5% el baseline 0.0091
      guard: {metric: val_vd_r2, op: ">=", threshold: 0.9598}   # no degradar (baseline 0.9626 - std)
tracking:
  backend: file                     # file | mlflow | wandb  (ver 1.5)
  mlflow_tracking_uri: null         # local por defecto si backend=mlflow
epochs_override: null               # opcional; si no, usa el de cada config
```

- **`success_criterion` obligatorio y predefinido** por variante (`spec-001` Fase 6.3): se
  fija **antes** de correr, evita p-hacking. El preflight (1.2) **rechaza** una variante
  sin criterio (salvo la línea base). Los criterios de arriba **no son inventados**: son los
  ya registrados en `docs/experiments.md` para `unet_film` y `fno_axial_attn`; este spec solo
  los **estructura** para que la Fase 5 pueda auto-evaluarlos.
- **`success_criterion` estructurado, no texto libre (ajuste 2026-07-16).** Hoy en
  `aggregate_experiments.py`/`docs/experiments.md` el criterio es una **cadena legible** que un
  humano interpreta; el veredicto automático solo cubre "¿supera la línea base?" sobre
  `val_sf_rmse`. Para que la Fase 5 marque **cumplido/no-cumplido por criterio arbitrario**
  (§1.6.3) sin ambigüedad, la campaña adopta un criterio **estructurado** (`metric`/`op`/
  `threshold` + `guard` opcional). El renderizador de reporte debe seguir imprimiendo también
  la forma legible para `docs/experiments.md`.
- **`seeds` ≥ 3** (o `n_seeds` ≥ 3 con derivación determinística `base+i`, igual que
  `run_experiment.py::parse_seeds`, que ya deriva `42, 43, …`). El preflight rechaza < 3.

### 1.2 Preflight (validación antes de gastar GPU)

`run_campaign.py --dry-run` corre **solo** estas comprobaciones y **imprime la cola** sin
entrenar:

1. Cada `config` referenciada existe y carga (loader de `spec-001` F2).
2. Cada `variant.name` está registrado en `build_model` (`spec-001` F3) — variante no
   registrada aborta con error explícito (no se descubre a mitad de campaña).
3. `seeds` ≥ 3 y `success_criterion` presente para toda variante no-baseline.
4. Datos presentes en `data/processed/{train,test}/` (reutiliza `check_missing_processed.py`).
5. **Checksum del split** (`train_test_split_80_20.csv`) calculado y **fijado** en el
   manifiesto — cualquier corrida futura de la misma campaña que vea otro checksum **aborta**
   (protege la comparabilidad; cierra el riesgo abierto de `spec-001` §2 "split inmutable").
6. GPU disponible (`torch.cuda.is_available()`), espacio en disco suficiente para M×N
   checkpoints (estimado y avisado).
7. Backend de tracking disponible si no es `file` (import de MLflow/W&B) — si falta,
   **degrada a `file` con warning**, no aborta (el rigor no depende del dashboard).

### 1.3 Runner de campaña — `scripts/run_campaign.py`

- **Expansión de la matriz:** `variants × seeds` → cola de trabajos `(variant, seed)`.
- **Ejecución secuencial en 1 GPU — decisión de diseño (ajustada 2026-07-16):** hay dos
  granularidades de reutilización, y **la campaña reutiliza `scripts/run_experiment.py` por
  variante** (no invoca `train.py` directamente por trabajo, como decía el borrador original):
  - `run_experiment.py` **ya** expande las seeds de una variante, corre cada una como
    subproceso a `train.py`, **aísla el fallo de una seed** y escribe `run_manifest.json` con
    el estado por seed. Reutilizarlo evita duplicar ese loop (que ya cubren los tests de
    `spec-001`) y mantiene una sola fuente de verdad para "N seeds de 1 arch".
  - **Gap a cubrir sobre `run_experiment.py`:** hoy **no** tiene `--resume` (no salta seeds ya
    completas) ni escribe marcadores `run.done`. La campaña añade el resume a nivel de trabajo
    **por encima** (comprueba `run.done` antes de lanzar cada seed) o extiende
    `run_experiment.py` con un `--resume` mínimo — **decisión a fijar en Fase 3** (preferible lo
    primero: no tocar `run_experiment.py`, coherente con "scripts de spec-001 sin modificar",
    §3). El comando por seed sigue siendo `train.py --config <snapshot> --seed <s>
    --model-variant <v> --experiment-name <v>`.
  - **Aislamiento total** se mantiene: un `RuntimeError` (p. ej. guardas NaN/Inf M6) de una
    corrida **no** aborta la campaña; se marca `failed` y sigue.
- **Salida por trabajo:** `outputs/campaigns/<campaign_name>/<variant>/seed_<s>/`
  (`metrics_history.json`, `best.pt`, `config.json`, logs). **Integración con lo existente:**
  `train.py::resolve_config` ya deriva `outputs/<experiment_name>/seed_<seed>` cuando se pasa
  `--experiment-name` sin `--output-dir`; la campaña obtiene la ruta anidada pasando
  `experiment_name = "campaigns/<name>/<variant>"` (o exponiendo un `outputs_root` en
  `run_experiment.py`, hoy parámetro de función pero no flag CLI). Esto hay que **verificarlo
  en Fase 3** — es el único punto donde las rutas de campaña tocan la derivación de spec-001.
- **Resume idempotente (`--resume`):** cada trabajo escribe un marcador `run.done` con su
  `run_signature` (`spec-001` F1) al completar. Al reanudar, un trabajo con `run.done` de
  **firma compatible** se **salta**; uno `failed` o inexistente se **re-ejecuta**. Permite
  matar y relanzar campañas largas sin recomputar lo hecho.
- **Estado de campaña:** `outputs/campaigns/<name>/campaign_state.json` (cola, estado de
  cada trabajo `pending`/`running`/`completed`/`failed`, timestamps), escrito
  **atómicamente** (write-to-temp + rename) tras cada trabajo — sobrevive a interrupciones.
- **`--dry-run`:** solo preflight + imprime la cola (1.2). **Gate de confirmación:** correr
  la campaña real (con entrenamiento) **exige `--yes` o confirmación explícita del usuario**
  (§Despliegue de `CLAUDE.md`; no lanzar GPU sin consentimiento).

### 1.4 Captura de reproducibilidad — módulo reutilizable

Al **iniciar** la campaña, `run_campaign.py` escribe
`outputs/campaigns/<name>/reproducibility/`:

- `git.json`: commit hash + `is_dirty` (si el árbol está sucio, **warning** — los
  resultados no serían reproducibles desde un commit limpio).
- `environment.txt`: `pip freeze` + versiones de Python / `torch` / CUDA / driver.
- `split.sha256`: checksum del `train_test_split_80_20.csv` usado (1.2.5).
- `configs/`: **copia** (no referencia) de cada `configs/experiments/<v>.yaml` de la
  campaña — el snapshot que realmente se corrió, inmune a ediciones posteriores.
- `campaign_manifest.json`: nombre, seeds, variantes, criterios, checksum del split, hash
  de git, timestamp de inicio, y ruta a todo lo anterior.

> Estos artefactos son lo que hace la campaña **replicable meses después**: cualquiera puede
> reconstruir el entorno, el código exacto y los datos exactos que produjeron cada número.

### 1.5 Tracking — abstracción con adaptadores (archivos siempre + backend opcional)

Interfaz `ExperimentTracker` con métodos `log_params`, `log_metrics(step)`,
`log_artifact(path)`, `finish()`. Implementaciones:

- **`FileTracker` (siempre activa, cero deps):** ya cubierto por los `metrics_history.json` /
  `config.json` que `training/loop.py` escribe; el tracker solo consolida rutas en el
  manifiesto. Es la fuente de verdad para la agregación (1.6) — **no** depende de ningún
  backend externo.
- **Adaptador opcional (`backend: mlflow | wandb`):** espeja params/métricas/artefactos al
  backend para dashboards y comparación visual. **Recomendado: MLflow local**
  (`mlflow_tracking_uri` local, sin servicio externo ni cuenta) — W&B es **cloud** y
  **publica los datos de entrenamiento fuera** (considerar antes de activarlo en un proyecto
  académico; requiere consentimiento explícito).
- **Dependencia nueva (⚠️):** activar MLflow o W&B requiere `pip install mlflow` (o `wandb`),
  que es una dependencia nueva → **mencionar y confirmar antes de instalar** (§Dependencias
  de `CLAUDE.md`). El core de la campaña funciona sin ella (`backend: file`).

### 1.6 Agregación y comparación cross-arquitectura

`scripts/aggregate_experiments.py` (de `spec-001` F5) se **envuelve** (preferido) para operar
a nivel campaña. **Estado real (2026-07-16):** los puntos 1 y 2 **ya están implementados** en
`aggregate_experiments.py` a nivel de *una* variante; la campaña los **itera** sobre todas las
variantes y **añade** el veredicto por criterio estructurado (3) y el reporte consolidado (4).

1. **[ya existe]** Por variante, agrega sus seeds (mean±std de `val_sf_r2`, `val_vd_r2`,
   `val_sf_rmse`, `val_vd_rmse` en la época del `best.pt` de cada seed, vía
   `aggregate_experiment` + `mean_std`).
2. **[ya existe]** **Comparación vs. la línea base** con test no paramétrico apropiado para n
   pequeño (`compare_groups`: Wilcoxon pareado por seed si comparten seeds; si no, Mann-Whitney
   U), **tamaño de efecto** (`effect_size`) y **valores crudos por seed** (`render_experiment_
   section` los tabula). Nada se oculta (`spec-001` F5.3). La campaña solo **repite esto por
   cada variante** contra `baseline`.
3. **[a implementar en esta campaña]** **Veredicto por `success_criterion` estructurado:** para
   cada variante, evalúa `metric op threshold` (+ `guard`) del §1.1 y marca **cumplido/no** de
   forma **mecánica** (hoy `compute_verdict` solo cubre "¿supera la línea base?" sobre
   `val_sf_rmse`; el criterio arbitrario es texto libre sin evaluar). Se mantienen los guardas
   de `spec-001` Fase 6: ninguna variante se declara "mejor" con < 3 seeds
   (`MIN_SEEDS_FOR_VERDICT`) ni con rangos mean±std solapados (`compute_verdict` ya lo checa).
4. **[a implementar]** Genera/actualiza `docs/experiments.md` (reutilizando el *upsert*
   idempotente por marcadores `<!-- experiment: <v> -->` ya existente) **y** un
   `outputs/campaigns/<name>/campaign_report.md` autocontenido: tabla de **todas** las variantes
   de la campaña vs. baseline en un solo lugar, con enlaces a los manifiestos de
   reproducibilidad (§1.4). Este reporte consolidado multi-variante es lo genuinamente nuevo:
   `aggregate_experiments.py` hoy escribe **una** sección por variante, no una vista de campaña.

---

## 2. Escalabilidad y buenas prácticas (checklist de diseño)

- **Añadir arquitectura = 1 línea** en el YAML de campaña (+ su variante y config, de
  `spec-001`). El orquestador no cambia.
- **Reanudable:** matar y relanzar no recomputa lo completo (marcadores `run.done` + estado
  atómico).
- **Aislamiento de fallos:** una corrida caída no tumba la campaña; queda `failed` y se
  re-ejecuta con `--resume`.
- **Idempotencia:** correr `run_campaign.py` dos veces con `--resume` converge al mismo
  estado; sin `--resume` no sobrescribe checkpoints sin respaldo (§Acciones prohibidas de
  `CLAUDE.md`).
- **Determinismo controlado:** `--seed` por corrida + `cfg.deterministic` opcional; seeds
  registradas en el manifiesto.
- **Trazabilidad total:** git hash, entorno, checksum del split, snapshot de config — todo
  versionado en el árbol de outputs.
- **Guarda de comparabilidad:** checksum del split fijado; si cambia, la campaña aborta
  (no compara peras con manzanas).
- **Dry-run + preflight:** nada de GPU hasta validar cola, configs, datos y criterios.
- **Logging estructurado:** vía `fno_co2.utils.get_logger` (no `print`), compatible con
  `tqdm` (§Convenciones de `CLAUDE.md`).
- **Gate de confirmación** antes de entrenar de verdad (§Despliegue de `CLAUDE.md`).

---

## Fase 0 — Precondiciones (bloqueantes) — ✅ COMPLETA (2026-07-16)

> Las 5 precondiciones están satisfechas. Desbloqueado para iniciar la Fase 1.

1. ✅ **`spec-001` Fases 1–5 implementadas** (`[DONE]`): `--model-variant`/`--seed`/
   `--experiment-name`/`--config` en `train.py`, `build_model` (discovery por convención),
   loader YAML (`load_config_from_yaml`), `scripts/run_experiment.py`,
   `scripts/aggregate_experiments.py`, `docs/experiments.md`. **Todo presente en el árbol.**
2. ✅ **Variantes registradas:** `unet_film` (`spec-002` `[DONE]`) y `fno_axial_attn`
   (`spec-003` `[TESTING]`) existen como `fno_co2.models.variants.<v>` con `build(cfg)` y su
   `configs/experiments/<v>.yaml`. La campaña puede arrancar con las 3 (baseline + ambas).
   > Nota: `fno_axial_attn` está en `[TESTING]` porque su Fase 5 (multi-seed real con GPU)
   > **es justamente** la que esta campaña ejecuta — incluirlo aquí cierra esa fase.
3. ✅ **Datos y split estables:** `reports/train_test_split_80_20.csv` presente; línea base
   **congelada** (`baseline-v1`, commit `ce9cbfa`) y **re-agregada con 3 seeds** (42,43,44) en
   `docs/experiments.md` tras el fix de `data_root`. Verificar `data/processed/{train,test}`
   completos con `scripts/etl/check_missing_processed.py` justo antes de la Fase 7.
4. ✅ **Tracking confirmado con el usuario (2026-07-16): `file`.** Sin dependencias nuevas;
   MLflow/W&B queda descartado por ahora (se puede reconsiderar más adelante sin rehacer nada,
   ya que `file` es el backend que siempre está activo, §1.5).
5. ✅ **Rama `feature/campaign-orchestration` creada desde `development`.**

**Verificación (ejecutada 2026-07-16):**
- `build_model` despacha las 3 variantes correctamente: `fno_baseline` (17,670,338 params),
  `unet_film` (69,893,250 params con `Config()` default `hidden_dim=128`; el YAML de la
  variante usa `hidden_dim=64` → 17.7M, ver backlog `spec-002-debt-001`), `fno_axial_attn`
  (18,461,890 params).
- Checksum del split calculado:
  `f51dfe25cd14120e5624aefdc4b212b88b16610b71a11aaf143d3b8fa529c95d` —
  `reports/train_test_split_80_20.csv` (valor de referencia para la guarda de comparabilidad
  del preflight, §1.2.5).
- `run_campaign.py --help` **pendiente** (se implementa en Fase 1; no es precondición de
  Fase 0, es su primer entregable).

---

## Fase 1 — Esquema de campaña + preflight — ✅ COMPLETA (2026-07-16)

**Dónde:** `configs/campaigns/` (nuevo), `scripts/run_campaign.py` (parcial),
`src/fno_co2/experiments/campaign_config.py` (nuevo).

1. ✅ Definir y cargar el YAML de campaña (§1.1), reutilizando el loader de `spec-001` F2
   (`load_config_from_yaml`) para las configs por-variante.
2. ✅ Preflight implementado (`run_preflight`): los 7 puntos de §1.2 — configs cargan,
   variante despachable por `build_model` (vía `cfg.model_variant`, no `variant.name` —
   ambos difieren en `baseline`/`fno_baseline`), `seeds >= 3` + `success_criterion` presente
   salvo `baseline`, datos presentes, **guarda de checksum del split** (aborta si difiere de
   un `split.sha256` ya registrado), GPU/disco como *warnings* (no abortan), tracking
   degrada a `file` con warning si el backend pedido no está instalado.
3. ✅ `configs/campaigns/fno_vs_unet_vs_attn.yaml`: baseline + `unet_film` + `fno_axial_attn`,
   con los `success_criterion` estructurados ya fijados en `docs/experiments.md`.
4. ✅ `scripts/run_campaign.py` (parcial, solo `--dry-run`): carga la campaña, corre el
   preflight, imprime la cola. Sin `--dry-run` rechaza explícito (exit 2) — la ejecución real
   es Fase 3.

**Verificación (ejecutada):** `tests/unit/test_campaign_config.py` — **13 tests, todos en
verde**: YAML válido carga y expande la cola esperada; `n_seeds` deriva `42,43,44`; < 3 seeds,
criterio ausente (no-baseline), config inexistente, variante no registrada, datos ausentes,
checksum de split distinto y split inexistente fallan explícito; baseline exento de criterio;
backend de tracking ausente degrada a `file` con warning; el YAML de ejemplo real
(`fno_vs_unet_vs_attn.yaml`) carga con las 3 variantes y sus criterios. Además, verificado
manualmente: `run_campaign.py --config configs/campaigns/fno_vs_unet_vs_attn.yaml --dry-run`
imprime las 9 corridas (3 variantes × 3 seeds) y pasa preflight contra los datos reales del
repo; sin `--dry-run` sale con código 2 sin ejecutar nada. Suite completa
(`pytest tests/ -m "not slow"`): **164 passed, 0 failed** (254s).

---

## Fase 2 — Captura de reproducibilidad

## Fase 2 — Captura de reproducibilidad — ✅ COMPLETA (2026-07-16)

**Dónde:** `src/fno_co2/experiments/reproducibility.py` (nuevo).

1. ✅ `capture_git_info` (hash + `is_dirty` vía `git rev-parse HEAD` / `git status
   --porcelain`), `capture_environment_info` (Python/torch/CUDA/cudnn/GPU + `pip freeze`),
   `compute_file_checksum` (reutilizado de `campaign_config.py`, Fase 1) y
   `copy_config_snapshots` (copia — no referencia — el config de cada variante).
2. ✅ `capture_reproducibility` orquesta todo y escribe atómicamente (write-to-temp +
   `Path.replace`, vía `atomic_write_text`/`atomic_write_json`) `outputs/campaigns/<name>/
   reproducibility/{git.json,environment.txt,split.sha256,configs/}` +
   `campaign_manifest.json` en `outputs/campaigns/<name>/`.

**Verificación (ejecutada):** `tests/unit/test_campaign_reproducibility.py` — **7 tests, todos
en verde**: hash + árbol limpio/sucio correctos (repo git aislado en `tmp_path`, nunca el
repo real); `environment.txt` incluye Python/torch/pip freeze; roundtrip atómico texto/JSON
sin dejar `.tmp` residual; snapshots de config copiados byte-a-byte; `capture_reproducibility`
end-to-end escribe manifiesto + los 4 artefactos con los valores esperados; árbol sucio se
refleja en el manifiesto. Verificado manualmente contra el repo real (campaña de ejemplo,
`outputs_root` de prueba fuera del árbol): `split_checksum` coincide exactamente con el
calculado en Fase 0 (`f51dfe25...529c95d`) y el `commit_hash` coincide con el commit de Fase 0
(`c99dd1f`), con `is_dirty=True` correctamente detectado (cambios de Fase 1/2 sin commitear).
> **Nota:** el nombre original del spec (`test_reproducibility.py`) **colisionaba** con un
> archivo ya existente de otro spec (M4: determinismo de `DataLoader`/`resolve_device`, sin
> relación con campañas) — se usó `test_campaign_reproducibility.py` para no pisarlo.

---

## Fase 3 — Runner de campaña (cola secuencial + resume) — ✅ COMPLETA (2026-07-16)

**Dónde:** `src/fno_co2/experiments/campaign_runner.py` (nuevo, lógica), `scripts/run_campaign.py`
(completado: `--resume`, `--yes`, ejecución real).

1. ✅ **Decisión de diseño fijada:** `run_campaign()` filtra, **por variante**, las seeds ya
   completas (según `run.done` + firma compatible) y llama a `run_experiment.run_experiment()`
   **una vez por variante** con solo las seeds pendientes — no una vez por seed (eso
   duplicaría/pisaría el `run_manifest.json` de `run_experiment.py` en cada llamada) ni
   reimplementa el subproceso a `train.py`. `run_experiment.py` **queda sin modificar**; el
   resume vive enteramente en la capa de campaña (`run.done` se comprueba *antes* de invocar
   `run_experiment`, no dentro de él).
2. ✅ **Firma de corrida reutilizada, no reinventada:** el "firma compatible" del `run.done`
   usa exactamente `training/checkpoint.py::build_run_signature` +
   `check_resume_compatibility` (`spec-001` F1) — si la config de una variante cambia después
   de completar una seed (p. ej. otro `hidden_dim`), esa seed se re-ejecuta en vez de darse
   por buena silenciosamente.
3. ✅ **Ruta de salida anidada confirmada:** pasando `experiment_name="campaigns/<name>/<variant>"`
   a `run_experiment_fn`, `train.py::resolve_config` (sin tocarlo) deriva exactamente
   `outputs/campaigns/<name>/<variant>/seed_<s>/` — confirmado end-to-end en los tests.
4. ✅ **Aislamiento de fallos:** una seed con `returncode != 0` (vía `run_experiment.py`, que
   ya lo maneja) queda `failed` en `campaign_state.json`, sin `run.done`, y no aborta las
   demás seeds/variantes.
5. ✅ **`campaign_state.json` atómico** (reutiliza `atomic_write_json` de Fase 2), escrito tras
   cada cambio de estado (antes y después de cada llamada a `run_experiment_fn`).
6. ✅ **Guarda "sin `--resume` no pisa salidas existentes":** `NoResumeOutputExistsError` si
   ya hay contenido en algún `job_dir` y no se pasó `resume=True` — exige `--resume` o limpiar
   manualmente (`CLAUDE.md` §Acciones prohibidas).
7. ✅ **Gate de confirmación real:** `scripts/run_campaign.py` sin `--dry-run` exige `--yes`
   explícito (verificado: sale con código 2 y mensaje claro si falta).
8. ✅ Logging estructurado con `get_logger` en todo el flujo.

**Verificación (ejecutada):** `tests/unit/test_run_campaign.py` — **9 tests, todos en
verde**, con `train.py` **mockeado** (un script Python real minimal, invocado como subproceso
genuino vía el `run_experiment.py` **real**, sin gastar GPU): la cola corre y escribe
`run.done`/`campaign_state.json`; una seed fallida (`999`, hardcodeada para fallar en el fake
script) no aborta el resto; `--resume` salta seeds con firma compatible y NO vuelve a invocar
`run_experiment_fn` si no hay nada pendiente; `--resume` re-ejecuta una seed `failed` pero
salta la ya completa; `--resume` re-ejecuta si la firma quedó incompatible (config cambiada);
`campaign_state.json` no deja `.tmp` residual; sin `--resume`, una segunda corrida sobre
salidas existentes lanza `NoResumeOutputExistsError`; `build_parser()` reconoce `--resume`/
`--yes`; sin `--yes` el script rechaza explícito (exit 2) antes de tocar `run_experiment.py`.
Verificado manualmente: `run_campaign.py --dry-run`/sin `--yes` contra la campaña real siguen
funcionando como en Fase 1. (Integración real contra GPU: Fase 7.)

---

## Fase 4 — Abstracción de tracking — ✅ COMPLETA (2026-07-16)

**Dónde:** `src/fno_co2/experiments/tracking.py` (nuevo), `campaign_runner.py` (integración).

1. ✅ Interfaz `ExperimentTracker` (`log_params`/`log_metrics`/`log_artifact`/`finish`) +
   `FileTracker` (siempre activo, cero deps: no duplica `metrics_history.json`/`config.json`
   de `training/loop.py`, solo consolida params + rutas de artefactos en
   `<job_dir>/tracker_paths.json`) + adaptadores opcionales `MlflowTracker`/`WandbTracker`
   con **import diferido** (solo al construirse, nunca a nivel de módulo). `build_tracker(
   backend, run_dir, ...)` selecciona según `campaign.tracking_backend` (decidido en Fase 0:
   **`file`**, sin dependencias nuevas).
2. ✅ Degradación grácil: `build_tracker` atrapa `ModuleNotFoundError` al construir
   `MlflowTracker`/`WandbTracker` y degrada a `FileTracker` con warning — no aborta.
   (Complementa, no duplica, el chequeo de solo-importabilidad de `run_preflight` §1.2.7:
   ese es en tiempo de preflight —sin efectos secundarios—, `build_tracker` es en tiempo de
   ejecución real y si el backend existe, sí construye/arranca el tracker.)
3. ✅ **No se instaló MLflow ni W&B** — confirmado en Fase 0 que el backend es `file`; el
   código de los adaptadores existe y es funcional (testeado con la ausencia real de esos
   paquetes en el entorno), pero instalarlos sigue pendiente de una decisión futura del
   usuario si se desea dashboard (`CLAUDE.md` §Dependencias).
4. ✅ **Integrado en `campaign_runner.run_campaign()`:** tras cada seed completada, consolida
   (`seed`, `model_variant`, `lr` como params + `metrics_history.json`/`best.pt`/`config.json`
   como artefactos, los que existan) vía el tracker — sin tocar `train.py`/`run_experiment.py`.

**Verificación (ejecutada):** `tests/unit/test_tracking.py` — **6 tests, todos en verde**:
`FileTracker` consolida params/artefactos a `tracker_paths.json` sin dependencias; no duplica
`metrics_history.json`; `build_tracker("file", ...)` retorna `FileTracker`; seleccionar
`mlflow`/`wandb` (**ninguno instalado en este entorno — degradación real, no mockeada**)
degrada a `file` con warning; un backend desconocido también degrada con warning. Además,
`test_run_campaign.py::test_run_campaign_runs_queue_and_writes_run_done` verifica la
integración end-to-end: cada seed completada escribe su `tracker_paths.json` junto al
`run.done`.

---

## Fase 5 — Agregación y reporte cross-arquitectura — ✅ COMPLETA (2026-07-16)

**Dónde:** `src/fno_co2/experiments/campaign_report.py` (nuevo), `scripts/aggregate_campaign.py`
(nuevo, CLI delgado), `outputs/campaigns/<name>/campaign_report.md` (generado).

1. ✅ **[reutiliza]** `aggregate_campaign()` itera todas las variantes de la campaña llamando
   a `aggregate_module.aggregate_experiment`/`compare_groups` **reales** (cargados por ruta,
   igual que `run_campaign.py` carga `run_experiment.py`) — no reimplementa la estadística.
   `scripts/aggregate_experiments.py` queda **sin modificar**.
2. ✅ **[nuevo]** `evaluate_structured_criterion(success_criterion, agg)`: evaluación
   **mecánica** `metric op threshold` (+ `guard` opcional) sobre las medias agregadas.
   `< MIN_SEEDS_FOR_VERDICT` (3) seeds → siempre `"inconcluso"`, sin excepciones. **Nota de
   diseño (simplificación deliberada):** a diferencia del veredicto informal de
   `compute_verdict` (que exige que los rangos mean±std no se solapen con el baseline antes
   de declarar "supera"), la evaluación estructurada compara la media directamente contra un
   **umbral absoluto ya fijado de antemano** (los criterios de `unet_film`/`fno_axial_attn`
   en `docs/experiments.md`) — no repite el chequeo de no-solapamiento porque el criterio en
   sí ya es la barra de rigor pre-registrada, no una comparación relativa contra el baseline.
3. ✅ `render_campaign_report`/`write_campaign_report` generan `campaign_report.md`
   autocontenido: tabla resumen (mean±std de las 4 métricas + criterio + veredicto por
   variante), tabla de comparación estadística vs. baseline (efecto/test/p-valor), enlace a
   `docs/experiments.md` para el detalle por seed, y sección de reproducibilidad (commit hash,
   `is_dirty`, split checksum, ruta a `reproducibility/`) leída de `campaign_manifest.json` si
   existe. `aggregate_campaign()` hace además el *upsert* de cada variante en
   `docs/experiments.md` reutilizando `render_experiment_section`/`upsert_experiments_doc`
   sin modificarlos.

**Verificación (ejecutada):** `tests/unit/test_campaign_report.py` — **10 tests, todos en
verde** — con `metrics_history.json` sintéticos de 3 variantes × 3 seeds (valores elegidos a
mano para forzar un caso "cumplido" —`unet_film`— y uno "no cumplido" —`fno_axial_attn`,
guard pasa pero la métrica principal no—): el reporte reproduce mean±std manuales
(`numpy.mean`/`std(ddof=1)`), aplica Wilcoxon/Mann-Whitney vía `compare_groups` real, marca
los veredictos correctamente, hace *upsert* de las 3 secciones en `docs/experiments.md`, y
`evaluate_structured_criterion` cubre por separado: `<3` seeds → inconcluso; criterio de
texto libre (línea base) → `"N/A"`; fallo de `guard` reportado explícito en el mensaje.
Verificado manualmente: `aggregate_campaign.py --help` funciona; contra la campaña real sin
datos (Fase 7 aún no corrida) falla con el mismo `FileNotFoundError` explícito que ya lanza
`aggregate_experiments.py` (comportamiento esperado y consistente, no un bug).

> **Corrección de un gap detectado durante esta fase:** `capture_reproducibility` (Fase 2)
> nunca quedó conectada a la ejecución real de la campaña — `scripts/run_campaign.py` jamás
> la invocaba, así que `campaign_manifest.json`/`reproducibility/split.sha256` **nunca se
> generaban**, y por lo tanto la guarda de checksum del split (§1.2.5) **nunca podía
> activarse en la práctica** (el archivo contra el que compara jamás existía). Corregido: al
> pasar el gate `--yes`, `run_campaign.py` ahora llama a `capture_reproducibility()` una sola
> vez (si `campaign_manifest.json` no existe ya) antes de correr la cola. De paso se añadió
> `--outputs-root` al CLI (ya existía como parámetro de `run_campaign()`) para poder testear
> este flujo completo de punta a punta sin tocar el `outputs/` real del repo
> (`test_run_campaign_script_with_yes_captures_reproducibility_and_runs`).

---

## Fase 6 — Suite de tests y no-regresión — ✅ COMPLETA (2026-07-16)

1. ✅ Todos los tests de Fases 1–5 pasan; `pytest tests/ -m "not slow"` completo sigue verde
   (no rompe `spec-000`..`spec-003`) — **197 passed** antes de esta fase (solo se agregó
   la Fase 6, sin tocar código de librería).
2. ✅ Test de integración corto (`@pytest.mark.slow`,
   `tests/integration/test_campaign_integration.py`): campaña mínima con `baseline`, **3
   seeds** (no 2 — ver nota de diseño abajo), 1 época, `--overfit-sample-idx 0`,
   `--device cpu` (esta workstation no tiene CUDA) → produce `campaign_state.json`
   consistente, `run.done` por corrida, manifiesto de reproducibilidad y un
   `campaign_report.md`. Corre la **CLI real** (`run_campaign.py` + `aggregate_campaign.py`
   como subprocesos, sin mocks) contra los datos reales del repo.

> **Corrección de diseño (2 seeds → 3):** el borrador original de este spec (2026-07-02)
> pedía una campaña de prueba con **2 seeds**, pero el preflight ya implementado en Fase 1
> **rechaza** cualquier campaña con `< 3` seeds (`MIN_SEEDS`, spec-001 Fase 6) — la propia
> ejecución del test lo confirmó (`Preflight fallido: la campaña declara 2 seeds; se
> requieren >= 3`). Se corrigió el test a 3 seeds; el guarda de `MIN_SEEDS` es la fuente de
> verdad, no el borrador de 2026-07-02.

> **Restricción arquitectónica real, descubierta al escribir este test (no introducida por
> esta fase, preexistente desde `spec-001`):** `train.py::resolve_config` deriva su
> `output_dir` real con el literal relativo `"outputs/<experiment_name>/seed_<seed>"`
> (relativo al `cwd` del proceso) — **sin enterarse** del `outputs_root` que
> `run_experiment.py`/`run_campaign()` usan para su propia contabilidad (`run_manifest.json`,
> `campaign_state.json`, `run.done`). Al correr por primera vez el test de integración con
> `--outputs-root` apuntando a un `tmp_path` aislado, la contabilidad de la campaña se
> escribió correctamente ahí, pero **`train.py` escribió los artefactos reales
> (`metrics_history.json`, `best.pt`, etc.) bajo el `outputs/campaigns/...` real del repo**
> (ignorado por git, limpiado manualmente tras detectarlo). Como `train.py` **no se puede
> modificar** (criterio de aceptación de este spec), esto es una limitación permanente, no
> un bug de esta fase: **una campaña real siempre escribe sus artefactos de entrenamiento
> bajo `<cwd>/outputs/campaigns/<name>/...`**; `--outputs-root` solo redirige la
> contabilidad propia de la campaña (útil únicamente en tests con `run_experiment`
> mockeado, que nunca invocan `train.py` de verdad — Fases 3 y 5). Documentado en el
> `--help` de `run_campaign.py`. El test de integración corre con `cwd` = raíz del repo
> (para que el preflight resuelva `data/processed`/el split reales) y **limpia
> expresamente** (`try`/`finally`) su directorio de salida (nombre de campaña distintivo,
> `spec004_fase6_integration_smoke`, para no colisionar con campañas reales) al terminar.

**Verificación (ejecutada):** `pytest tests/ -m "not slow"` verde (sin cambios de librería en
esta fase, solo el test slow nuevo); el test slow de integración **corrió realmente** (no
solo "pasa bajo demanda" en teoría): 247s (~4 min) en CPU, y confirmó sin mocks toda la
cadena Fases 1–5 — preflight, captura de reproducibilidad, ejecución de 3 seeds vía
`run_experiment.py` real, `run.done`/`campaign_state.json`, agregación+reporte, y **resume
idempotente** (segunda invocación con `--resume` salta las 3 seeds, `skipped=True`).

---

## Fase 7 — Ejecución real de la campaña

> **⚠️ Requiere GPU + datos post-C1 + baseline congelada + confirmación explícita.**

### 7.0 Preparación (2026-07-16, no requiere confirmación de entrenamiento — no entrena nada)

- **Reutilización de `baseline-v1` ya entrenado.** `outputs/baseline/seed_{42,43,44}/` ya
  existe (congelado, 3 seeds) — pero vive **fuera** del namespace de la campaña
  (`outputs/campaigns/<name>/baseline/seed_<s>/`), así que `run_campaign()` lo
  re-entrenaría desde cero si no se importa antes. Se agregó
  `fno_co2.experiments.campaign_runner.seed_existing_run` +
  `scripts/import_existing_run.py`: copian los artefactos (`metrics_history.json`,
  `config.json`, `checkpoints/best.pt`) al layout de la campaña y escriben un `run.done`
  con la **firma de corrida real** (`build_run_signature`) — para que `run_campaign.py
  --resume` salte esas 3 seeds sin re-entrenar. No modifica ni mueve la corrida original,
  solo copia; no sobrescribe un `job_dir` que ya tenga contenido. Uso:
  ```bash
  python scripts/import_existing_run.py \
    --config configs/campaigns/fno_vs_unet_vs_attn.yaml \
    --variant baseline --source-root outputs/baseline
  python scripts/run_campaign.py \
    --config configs/campaigns/fno_vs_unet_vs_attn.yaml --yes --resume
  ```
  Ahorra las 3 corridas de `baseline` (ver estimado de tiempo abajo — no es poco).
  Verificado con 7 tests unitarios (`test_seed_existing_run_*`, `test_run_campaign.py`).

- **Estimado de tiempo revisado (medido, no histórico) — completo, 3/3 variantes.** El
  estimado de `spec-002-debt-001` ("~1.5-2h/seed baseline") **no contemplaba correctamente
  el costo de la calibración de incertidumbre MC-Dropout**. Medición real (GPU RTX 6000 Ada,
  1 época, seed 42, datos completos, 2026-07-16):

  | variante | `train` (1 época) | `calib` (MC-Dropout ×30) | `eval` (con incertidumbre) | **total** |
  |---|---|---|---|---|
  | `baseline` | 12.1 min (3905 batches, 5.48 it/s) | 32.0 min (995 batches) | 33.0 min | **77.4 min** (1.29 h) |
  | `unet_film` | 14.8 min (7809 batches, batch_size=2, 9.40 it/s) | 46.0 min (1989 batches) | 47.6 min | **108.6 min** (1.81 h) |
  | `fno_axial_attn` | 48.4 min (3905 batches, **1.35 it/s** — atención axial ~4× más cara por batch que baseline) | 106.2 min | 112.2 min | **267.0 min** (4.45 h) |

  **Hallazgo clave:** `calib`+`eval` con incertidumbre **no corren cada época** —
  `training/loop.py` los dispara solo cuando `epoch == cfg.epochs` (época final) o
  `epoch % uncertainty_eval_interval(=10) == 0`. La mayoría de las épocas de una corrida
  real son "baratas" (`train` + `eval` sin incertidumbre, sin medir con precisión — la
  atención axial de `fno_axial_attn` sola, sin MC-Dropout, ya explica ~4× el costo de
  `train` de `baseline`); 2-3 épocas por corrida (según cuántas caen en el intervalo de 10 +
  la final) pagan el costo caro completo medido arriba.

  **`fno_axial_attn` es, con mucho, el factor dominante** — no `unet_film` (que era la
  preocupación original del backlog `spec-002-debt-001`). Con el histórico real de
  `baseline` (mejores épocas en 12/19/14 con `patience=5` → corridas totales de ~17-24
  épocas, 2-3 de ellas caras) como referencia de forma de la curva (**no medido para las
  otras 2 variantes** — supuesto, no dato):

  | variante | estimado por seed | 3 seeds |
  |---|---|---|
  | `baseline` | ~6-9 h | ~19-27 h |
  | `unet_film` | ~8-12 h | ~24-35 h |
  | `fno_axial_attn` | ~23-33 h (~1-1.5 **días** por seed) | ~70-100 h |

  **Total Fase 7 (9 corridas): ~4.7 a ~6.7 días** de GPU secuencial en 1 GPU; **~3.9 a
  ~5.6 días** si se reutiliza `baseline` (§7.0, arriba) — estimado **antes** del ajuste de
  AMP de abajo; sigue dominado casi por completo por `fno_axial_attn`, no por evitar
  re-entrenar baseline.

  **Ajuste aplicado (2026-07-16, sin riesgo científico):** `use_amp: true` (bfloat16,
  `torch.autocast`, sin `GradScaler` — bf16 comparte el rango de exponente de fp32, riesgo
  numérico bajo) en `unet_film.yaml` y `fno_axial_attn.yaml` — no cambia arquitectura ni
  métricas, solo la precisión numérica del cómputo. `baseline.yaml` **no se tocó**: dice
  explícitamente "No editar" y sus resultados ya están congelados en fp32 (`baseline-v1`,
  a reutilizar vía §7.0, no a re-entrenar). Impacto en el estimado de arriba: **no medido
  todavía** (los timing probes de §7.0 se corrieron con `use_amp: false`, antes de este
  cambio) — re-medir antes de comprometerse a un tiempo final de Fase 7.

  **Ajuste descartado tras revisión de código (2026-07-16):** se consideró subir/deshabilitar
  `uncertainty_eval_interval` para saltar la costosa calibración MC-Dropout más seguido, pero
  `do_uncertainty` en `training/loop.py` compara contra `epoch == cfg.epochs` (el máximo
  **configurado**, p. ej. 100) — no contra la época real donde para el early stopping (que
  `cfg.epochs` nunca refleja). Con el histórico real de `baseline` deteniéndose en épocas
  ~12-24, esa rama prácticamente nunca se activa; deshabilitar el intervalo periódico
  eliminaría el diagnóstico de incertidumbre de casi toda corrida real, silenciosamente.
  **No se tocó** este parámetro. Arreglarlo de verdad requeriría que `training/loop.py`
  reconozca "la época donde el early stopping decide parar" como época final para efectos de
  incertidumbre — un cambio a `training/loop.py` (spec-000/spec-001), fuera del alcance de
  este spec; queda registrado aquí como deuda técnica a evaluar, no como ajuste de config.

  **Decisión pendiente del usuario:** no se lanzó la ejecución real — la Fase 7 queda
  planificada pero **no iniciada**, a retomar cuando el usuario decida cómo proceder
  (re-medir con AMP, decidir sobre `attn_num_blocks`/`uncertainty_passes`, o correr tal cual).

1. `python scripts/run_campaign.py --config configs/campaigns/fno_vs_unet_vs_attn.yaml
   --dry-run` → revisar la cola y el preflight.
2. Con confirmación del usuario: (opcional) importar `baseline-v1` con
   `import_existing_run.py` (§7.0) para ahorrar esas 3 corridas; luego correr sin
   `--dry-run` (`--yes`), ≥ 3 seeds por variante. **Antes de lanzar de verdad, decidir qué
   hacer con el costo de `fno_axial_attn`** (ver estimado arriba) — correrlo tal cual
   (~1-1.5 días/seed) o ajustar `uncertainty_eval_interval`/`uncertainty_passes` para esa
   variante primero.
3. Si se interrumpe: relanzar con `--resume`.
4. El agregador de campaña (§Fase 5) genera el reporte cross-arquitectura final.
5. **Cerrar backlog:** actualizar `docs/experiments.md` (tablas de `unet_film` y
   `fno_axial_attn` con números reales) y marcar `spec-002-debt-002` Fase 5.3 y `spec-003`
   Fase 5 como resueltos; promover `spec-003` a `[DONE]` si cumple su criterio.

**Verificación:** `docs/experiments.md` y `campaign_report.md` con todas las variantes ×
≥3 seeds vs. baseline, criterios predefinidos evaluados, y `reproducibility/` completo.

---

## 3. Archivos impactados (resumen)

| Archivo / carpeta | Fase | Naturaleza |
|---|---|---|
| `configs/campaigns/<name>.yaml` | 1 | **Nuevo** — declaración de la matriz arch × seeds |
| `src/fno_co2/experiments/campaign_config.py` | 1 | **Nuevo** — carga/validación del YAML + preflight |
| `src/fno_co2/experiments/reproducibility.py` | 2 | **Nuevo** — git/env/checksum/snapshots + manifiesto |
| `src/fno_co2/experiments/campaign_runner.py` | 3 | **Nuevo** — cola secuencial, resume, aislamiento de fallos, estado atómico |
| `scripts/run_campaign.py` | 1, 3 | **Nuevo** — CLI (preflight/dry-run F1; `--resume`/`--yes`/ejecución real F3) |
| `src/fno_co2/experiments/tracking.py` | 4 | **Nuevo** — `ExperimentTracker` + FileTracker + adaptador |
| `src/fno_co2/experiments/campaign_report.py` | 5 | **Nuevo** — evaluación de criterio estructurado + orquesta agregación por variante + reporte |
| `scripts/aggregate_campaign.py` | 5 | **Nuevo** — CLI delgado, carga `aggregate_experiments.py` por ruta (sin modificarlo) |
| `scripts/import_existing_run.py` | 7 (prep.) | **Nuevo** — importa una corrida ya entrenada (p. ej. `baseline-v1`) al layout de campaña |
| `outputs/campaigns/` | 3, 5 | Runtime — estado, corridas, reproducibility, reporte |
| `docs/experiments.md`, `campaign_report.md` | 5 | Append / generado |
| `pyproject.toml` | 4 | `mlflow` (o `wandb`) **solo si** se elige ese backend (**confirmar**) |
| `tests/unit/test_campaign_*.py` (incl. `test_campaign_reproducibility.py`), `test_tracking.py`, `test_run_campaign.py` | 1–6 | **Nuevos** |
| `scripts/train.py`, `run_experiment.py`, `models/*` | — | **NO se modifican** (se reutilizan; el orquestador vive por encima) |
| Git: rama `feature/campaign-orchestration` | 0 | Desde `development` |

---

## 4. Riesgos y precondiciones

- **~~Bloqueo por `spec-001`~~ (RESUELTO 2026-07-16):** `spec-001` está `[DONE]` y sus Fases
  1–5 existen en el árbol. Ya no es un bloqueo; queda como precondición **verificada** (Fase 0).
- **Costo de cómputo (M×N):** 3 variantes × 3 seeds = 9 entrenamientos completos
  secuenciales en 1 GPU. Estimación afinada con datos reales (`spec-002-debt-001`): `baseline`
  y `fno_axial_attn` a `hidden_dim=128`; `unet_film` a `hidden_dim=64` (17.7M params, más
  rápido que la estimación original de 128). Mitigación: resume reanudable, `--dry-run` para
  estimar antes, y priorizar variantes con hipótesis clara (`spec-001` Fase 6).
- **Cierre de pendientes del backlog:** esta campaña **es** la corrida multi-seed que cierra
  `spec-002-debt-002` Fase 5.3 (`unet_film`, hoy con tabla "pendiente re-run" en
  `docs/experiments.md`) y `spec-003` Fase 5 (`fno_axial_attn`). Al terminar, actualizar el
  backlog marcando esos pendientes como resueltos con los números reales.
- **Memoria GPU no es riesgo en ejecución secuencial (`spec-002-debt-001`, RESUELTO):** el OOM
  temido era artefacto de **seeds en paralelo**; secuencialmente `unet_film` pico ~3.89 GiB a
  `batch_size=2`. La cola secuencial de esta campaña ya evita ese modo de fallo por diseño; **no
  se necesita** gradient checkpointing ni reducir `unet_depth`.
- **Dependencia nueva (MLflow/W&B):** solo si se elige ese backend; **confirmar e instalar**
  (§Dependencias de `CLAUDE.md`). **W&B publica datos externamente** — no activarlo sin
  consentimiento explícito; MLflow local evita transmisión externa. El core no lo necesita.
- **Split inmutable:** la guarda de checksum aborta si `train_test_split_80_20.csv` cambió
  entre corridas de la misma campaña — cierra el riesgo que `spec-001` §2 dejaba a
  disciplina manual. Regenerar el split invalida la campaña por diseño.
- **Sobrescritura de checkpoints:** sin `--resume`, no debe pisar `best.pt` existentes sin
  respaldo (§Acciones prohibidas de `CLAUDE.md`); el runner exige `--resume` o directorio
  limpio.
- **Árbol de git sucio:** los resultados de un árbol `dirty` no son reproducibles desde un
  commit; se registra `is_dirty=True` y se avisa, pero no se bloquea (decisión del usuario).
- **`docs/experiments.md` en `.gitignore`** (`spec-001` §2): el registro vive local;
  `campaign_report.md` (bajo `outputs/`, también ignorado) igual. Versionarlos es decisión
  del usuario, fuera de alcance.

---

## 5. Criterios de aceptación

- [ ] `run_campaign.py --dry-run` valida configs, variantes registradas, datos, ≥3 seeds y
      criterios predefinidos, imprime la cola y **no** entrena.
- [ ] `run_campaign.py` corre la matriz `variants × seeds` **secuencialmente en 1 GPU**, con
      salida en `outputs/campaigns/<name>/<variant>/seed_<s>/`.
- [ ] Una corrida fallida **no** aborta la campaña; queda `failed` y `--resume` la
      re-ejecuta saltando las `run.done`.
- [ ] `campaign_state.json` se escribe atómicamente y refleja el progreso real.
- [ ] `reproducibility/` contiene git hash, entorno, checksum del split y snapshots de
      config copiados.
- [ ] La **guarda de checksum del split** aborta la campaña si el split cambió.
- [ ] `FileTracker` funciona **sin dependencias nuevas**; el adaptador MLflow/W&B es opcional
      y degrada a `file` si falta, con warning.
- [ ] El reporte cross-arquitectura agrega mean±std por variante, compara vs. baseline con
      test no paramétrico + tamaño de efecto + valores por seed (**reutilizando**
      `aggregate_experiments.py`, sin modificarlo), y marca cada `success_criterion`
      **estructurado** (`metric`/`op`/`threshold` + `guard`) como cumplido/no de forma mecánica.
- [ ] Ninguna variante se reporta "mejor" con < 3 seeds ni con rangos mean±std solapados
      (`spec-001` Fase 6).
- [ ] `scripts/train.py`, `aggregate_experiments.py` y `models/*` quedan **sin modificar**;
      `run_experiment.py` se reutiliza sin modificar (preferido) o, si se opta por el resume
      interno, se le añade **solo** un `--resume` aditivo que no altera su comportamiento
      actual y mantiene verdes sus tests de `spec-001`.
- [ ] `pytest tests/ -m "not slow"` completo pasa; el test slow de integración de campaña
      pasa bajo demanda.
- [ ] Correr la campaña real exige confirmación explícita (`--yes`) — no lanza GPU sola.
