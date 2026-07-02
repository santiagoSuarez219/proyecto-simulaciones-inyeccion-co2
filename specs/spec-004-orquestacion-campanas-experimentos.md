# spec-004 — Orquestación de campañas de experimentos (matriz arquitectura × seeds)

> **Autor:** rol `@architect`
> **Fecha:** 2026-07-02
> **Estado:** PLANIFICADO — **bloqueado** por `spec-001` (requiere sus Fases 1–5
> implementadas; ver Fase 0). No iniciar antes.
> **Depende de:** `spec-001` (framework de experimentación: `--model-variant`,
> `build_model`, loader YAML, `run_experiment.py` multi-seed, `aggregate_experiments.py`,
> `docs/experiments.md`). Consume las variantes de `spec-002` (U-Net) y `spec-003`
> (FNO+atención) **si existen**, pero puede correr con las que haya (incluida solo la
> línea base).
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
    success_criterion: "no degradar val_sf_r2 mean >2% ni val_vd_r2"
  - name: fno_axial_attn
    config: configs/experiments/fno_axial_attn.yaml
    success_criterion: "reducir val_sf_rmse mean >=5% sin degradar val_vd_r2"
tracking:
  backend: file                     # file | mlflow | wandb  (ver 1.5)
  mlflow_tracking_uri: null         # local por defecto si backend=mlflow
epochs_override: null               # opcional; si no, usa el de cada config
```

- **`success_criterion` obligatorio y predefinido** por variante (`spec-001` Fase 6.3): se
  fija **antes** de correr, evita p-hacking. El preflight (1.2) **rechaza** una variante
  sin criterio (salvo la línea base).
- **`seeds` ≥ 3** (o `n_seeds` ≥ 3 con derivación determinística `base+i`). El preflight
  rechaza < 3.

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
- **Ejecución secuencial en 1 GPU:** cada trabajo invoca `scripts/train.py --config
  <snapshot> --seed <s> --model-variant <v> --experiment-name <v>` **como subproceso**
  (aislamiento total: un `RuntimeError` — p. ej. guardas NaN/Inf M6 — de una corrida **no**
  aborta la campaña; se marca `failed` y sigue). **No reimplementa** el loop; reutiliza el
  camino probado por los tests de `spec-000`/`spec-001`.
- **Salida por trabajo:** `outputs/campaigns/<campaign_name>/<variant>/seed_<s>/`
  (`metrics_history.json`, `best.pt`, `config.json`, logs).
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

`scripts/aggregate_experiments.py` (de `spec-001` F5) se **extiende** (o se envuelve) para
operar a nivel campaña:

1. Por variante, agrega sus seeds (mean±std de `val_sf_r2`, `val_vd_r2`, `val_sf_rmse`,
   `val_vd_rmse` en la época de `best.pt`) — exactamente como `spec-001` F5.
2. **Comparación cross-arquitectura:** cada variante vs. la línea base con test no
   paramétrico apropiado para n pequeño (Mann-Whitney U / Wilcoxon pareado por seed),
   **tamaño de efecto** y **valores crudos por seed** (nada se oculta; `spec-001` F5.3).
3. **Veredicto por criterio predefinido:** para cada variante, marca si cumplió su
   `success_criterion` (1.1). Ninguna variante se declara "mejor" con < 3 seeds ni con
   rangos mean±std solapados (`spec-001` Fase 6).
4. Genera/actualiza `docs/experiments.md` (append) **y** un
   `outputs/campaigns/<name>/campaign_report.md` autocontenido: tabla de todas las variantes
   de la campaña vs. baseline, con enlaces a los manifiestos de reproducibilidad.

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

## Fase 0 — Precondiciones (bloqueantes)

1. **`spec-001` Fases 1–5 implementadas:** `--model-variant`/`--seed`/`--experiment-name`/
   `--config`, `build_model`, loader YAML, `run_experiment.py`, `aggregate_experiments.py`,
   `docs/experiments.md`. Sin esto, la campaña no tiene sobre qué orquestar.
2. **Variantes a incluir registradas** (`spec-002` U-Net y/o `spec-003` FNO+atención) — o
   correr la campaña solo con `baseline` si aún no existen.
3. **Datos `data/processed/` regenerados post-C1** y `train_test_split_80_20.csv` estable
   (`spec-001` Fase 0). La línea base **congelada** (`baseline-v1`) antes de reportar
   comparaciones.
4. **Decisión de tracking confirmada:** `file` (sin deps) o `mlflow`/`wandb` (**confirmar e
   instalar** la dependencia nueva, §Dependencias de `CLAUDE.md`). Recomendado MLflow local.
5. Rama `feature/campaign-orchestration` desde `development` (es infraestructura, no un
   experimento de arquitectura → prefijo `feature/`, `CLAUDE.md` §Git).

**Verificación:** `run_campaign.py --help` disponible; `build_model` despacha las variantes
de la campaña; split checksum calculable.

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

1. Expansión de la matriz, ejecución secuencial vía subproceso a `scripts/train.py`,
   aislamiento de fallos, marcadores `run.done`, `campaign_state.json` atómico, `--resume`,
   `--dry-run`, gate de confirmación (§1.3).
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

**Dónde:** `scripts/aggregate_experiments.py` (extender, de `spec-001` F5),
`outputs/campaigns/<name>/campaign_report.md` (generado).

1. Agregar seeds por variante y comparar cada variante vs. baseline con test no paramétrico,
   tamaño de efecto y valores crudos (§1.6), reutilizando la lógica de `spec-001` F5.
2. Veredicto por `success_criterion` predefinido; sin conclusiones con < 3 seeds ni rangos
   solapados (`spec-001` Fase 6).
3. Generar `campaign_report.md` autocontenido + append a `docs/experiments.md`.

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
4. `aggregate_experiments.py` genera el reporte cross-arquitectura final.

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
| `scripts/aggregate_experiments.py` | 5 | Extiende (de `spec-001` F5) a nivel campaña |
| `outputs/campaigns/` | 3, 5 | Runtime — estado, corridas, reproducibility, reporte |
| `docs/experiments.md`, `campaign_report.md` | 5 | Append / generado |
| `pyproject.toml` | 4 | `mlflow` (o `wandb`) **solo si** se elige ese backend (**confirmar**) |
| `tests/unit/test_campaign_*.py`, `test_reproducibility.py`, `test_tracking.py`, `test_run_campaign.py` | 1–6 | **Nuevos** |
| `scripts/train.py`, `run_experiment.py`, `models/*` | — | **NO se modifican** (se reutilizan; el orquestador vive por encima) |
| Git: rama `feature/campaign-orchestration` | 0 | Desde `development` |

---

## 4. Riesgos y precondiciones

- **Bloqueo por `spec-001`:** la campaña orquesta piezas que `spec-001` debe entregar
  (Fases 1–5). Sin ellas no hay nada que orquestar. Fase 0 es dura.
- **Costo de cómputo (M×N):** 3 variantes × 3 seeds = 9 entrenamientos completos
  secuenciales en 1 GPU — puede ser días. Mitigación: `--resume` (reanudable), `--dry-run`
  para estimar antes, y priorizar variantes con hipótesis clara (`spec-001` Fase 6).
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
      test no paramétrico + tamaño de efecto + valores por seed, y marca cada
      `success_criterion` como cumplido/no según lo predefinido.
- [ ] Ninguna variante se reporta "mejor" con < 3 seeds ni con rangos mean±std solapados
      (`spec-001` Fase 6).
- [ ] `scripts/train.py`, `run_experiment.py` y `models/*` quedan **sin modificar**.
- [ ] `pytest tests/ -m "not slow"` completo pasa; el test slow de integración de campaña
      pasa bajo demanda.
- [ ] Correr la campaña real exige confirmación explícita (`--yes`) — no lanza GPU sola.
