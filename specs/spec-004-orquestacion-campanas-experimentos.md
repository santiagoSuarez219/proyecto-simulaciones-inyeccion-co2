# spec-004 — Orquestación de campañas de experimentos (matriz arquitectura × seeds)

> **Autor:** rol `@architect`
> **Fecha:** 2026-07-02 · **Actualizado:** 2026-07-16 (revisión contra el estado real del
> repo tras cerrar `spec-001`/`spec-002`/`spec-003`)
> **Estado:** PLANIFICADO — **DESBLOQUEADO**. La precondición dura (`spec-001` Fases 1–5)
> **ya está satisfecha**: `spec-001` está `[DONE]` y sus artefactos existen en el árbol
> (`scripts/run_experiment.py`, `scripts/aggregate_experiments.py`, loader YAML,
> `build_model` con discovery por convención, `docs/experiments.md`). Listo para iniciar la
> Fase 0 → 1. Requiere confirmación del usuario para arrancar (rama nueva) y para la Fase 7
> (GPU).
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

## Fase 1 — Esquema de campaña + preflight

**Dónde:** `configs/campaigns/` (nuevo), `scripts/run_campaign.py` (parcial),
`src/fno_co2/experiments/campaign_config.py` (nuevo).

1. Definir y cargar el YAML de campaña (§1.1), reutilizando el loader de `spec-001` F2 para
   las configs por-variante.
2. Implementar el preflight (§1.2), incluyendo la guarda de checksum del split y el rechazo
   de < 3 seeds / criterio ausente.
3. `configs/campaigns/fno_vs_unet_vs_attn.yaml`: primera campaña de ejemplo (baseline +
   las variantes que existan).

**Verificación:** `tests/unit/test_campaign_config.py` — un YAML válido carga y expande a la
cola esperada; < 3 seeds, criterio ausente, config inexistente y variante no registrada
fallan en preflight con error explícito.

---

## Fase 2 — Captura de reproducibilidad

**Dónde:** `src/fno_co2/experiments/reproducibility.py` (nuevo).

1. Funciones para capturar git hash + dirty, entorno (`pip freeze`, versiones), checksum del
   split, y copiar snapshots de config (§1.4).
2. Escribir `campaign_manifest.json` + `reproducibility/` atómicamente.

**Verificación:** `tests/unit/test_reproducibility.py` — con un repo/entorno de prueba, el
manifiesto contiene hash, checksum y snapshots; el checksum de un CSV conocido coincide con
el esperado; árbol sucio marca `is_dirty=True`.

---

## Fase 3 — Runner de campaña (cola secuencial + resume)

**Dónde:** `scripts/run_campaign.py`.

1. Expansión de la matriz, ejecución secuencial **reutilizando `scripts/run_experiment.py`
   por variante** (que ya orquesta las seeds vía subproceso a `train.py`; §1.3),
   aislamiento de fallos, marcadores `run.done`, `campaign_state.json` atómico, `--resume`,
   `--dry-run`, gate de confirmación (§1.3). Fijar en esta fase cómo se logra el resume por
   seed (comprobar `run.done` antes de lanzar vs. `--resume` mínimo en `run_experiment.py`) y
   la ruta de salida anidada (`experiment_name="campaigns/<name>/<variant>"`).
2. Directorios de salida `outputs/campaigns/<name>/<variant>/seed_<s>/`.
3. Logging estructurado con `get_logger`.

**Verificación:** `tests/unit/test_run_campaign.py` — con `train.py` **mockeado**
(subproceso simulado): la cola corre en orden; una corrida `failed` no aborta el resto;
`--resume` salta las `run.done`; el estado se escribe atómicamente. (Integración real:
Fase 7.)

---

## Fase 4 — Abstracción de tracking

**Dónde:** `src/fno_co2/experiments/tracking.py` (nuevo).

1. Interfaz `ExperimentTracker` + `FileTracker` (siempre) + adaptador opcional
   `MlflowTracker`/`WandbTracker` (§1.5). Selección por `campaign.tracking.backend`.
2. Degradación grácil a `file` si el backend no está instalado (warning, no aborta).
3. **Confirmar e instalar** MLflow/W&B solo si el usuario elige ese backend (dependencia
   nueva).

**Verificación:** `tests/unit/test_tracking.py` — `FileTracker` consolida rutas sin deps;
seleccionar un backend ausente degrada a `file` con warning (mockeando el import faltante).

---

## Fase 5 — Agregación y reporte cross-arquitectura

**Dónde:** un envoltorio a nivel campaña (p. ej. `scripts/aggregate_campaign.py` **nuevo**, o
una función que itere sobre `aggregate_experiments.py` **sin modificarlo** para no romper sus
tests), `outputs/campaigns/<name>/campaign_report.md` (generado).

1. **[reutiliza]** Agregar seeds por variante y comparar cada variante vs. baseline con test
   no paramétrico, tamaño de efecto y valores crudos (§1.6.1–2) — **llamando** a
   `aggregate_experiment`/`compare_groups` existentes, no reimplementándolos.
2. **[nuevo]** Veredicto por `success_criterion` **estructurado** (`metric`/`op`/`threshold` +
   `guard`, §1.1): evaluación mecánica cumplido/no-cumplido. Sin conclusiones con < 3 seeds ni
   rangos mean±std solapados (reutiliza el guardas de `compute_verdict`, `spec-001` Fase 6).
3. Generar `campaign_report.md` autocontenido (tabla multi-variante vs. baseline + enlaces a
   `reproducibility/`) + *upsert* por variante en `docs/experiments.md` (mecanismo de
   marcadores ya existente).

**Verificación:** `tests/unit/test_campaign_report.py` — con `metrics_history.json`
sintéticos de 3 variantes × 3 seeds, el reporte reproduce mean±std manuales, aplica el test
estadístico y marca correctamente cumplido/no-cumplido según el criterio predefinido.

---

## Fase 6 — Suite de tests y no-regresión

1. Todos los tests de Fases 1–5 pasan; `pytest tests/ -m "not slow"` completo sigue verde
   (no rompe `spec-000`..`spec-003`).
2. Un test de integración corto (`@pytest.mark.slow`): campaña mínima con `baseline`,
   2 seeds, 1 época, `--overfit-sample-idx 0` → produce `campaign_state.json` consistente,
   `run.done` por corrida, manifiesto de reproducibilidad y un `campaign_report.md`.

**Verificación:** `pytest tests/ -m "not slow"` verde; el test slow de integración pasa bajo
demanda.

---

## Fase 7 — Ejecución real de la campaña

> **⚠️ Requiere GPU + datos post-C1 + baseline congelada + confirmación explícita.**

1. `python scripts/run_campaign.py --config configs/campaigns/fno_vs_unet_vs_attn.yaml
   --dry-run` → revisar la cola y el preflight.
2. Con confirmación del usuario: correr sin `--dry-run` (`--yes`), ≥ 3 seeds por variante.
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
| `scripts/run_campaign.py` | 1, 3 | **Nuevo** — orquestador (cola secuencial, resume, dry-run) |
| `src/fno_co2/experiments/tracking.py` | 4 | **Nuevo** — `ExperimentTracker` + FileTracker + adaptador |
| `scripts/aggregate_campaign.py` (o wrapper) | 5 | **Nuevo** — itera `aggregate_experiments.py` (sin modificarlo) + veredicto por criterio estructurado + reporte multi-variante |
| `outputs/campaigns/` | 3, 5 | Runtime — estado, corridas, reproducibility, reporte |
| `docs/experiments.md`, `campaign_report.md` | 5 | Append / generado |
| `pyproject.toml` | 4 | `mlflow` (o `wandb`) **solo si** se elige ese backend (**confirmar**) |
| `tests/unit/test_campaign_*.py`, `test_reproducibility.py`, `test_tracking.py`, `test_run_campaign.py` | 1–6 | **Nuevos** |
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
