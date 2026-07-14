# spec-001 — Framework de experimentación y comparación de arquitecturas [DONE]

> **Autor:** revisión de código (rol `@architect`)
> **Fecha:** 2026-07-02
> **Estado:** `[DONE]` — Fases 0-6 implementadas en `feature/framework-experimentacion-arquitecturas`
> (2026-07-10): línea base entrenada y taggeada (`baseline-v1` en `ce9cbfa`, checkpoint
> respaldado fuera del árbol de trabajo), CLI reproducible + YAML + registry de variantes +
> runner multi-seed + agregación estadística, todos con test unitario y verificación manual
> real. Fase 6 (rigor de mínimo 3 seeds) es disciplina de proceso, reforzada en código por
> `compute_verdict()`. (2026-07-14): `@reviewer` aprobó el framework (8/8 criterios de
> aceptación cubiertos, 124 tests pasando) — ver hallazgo 🟠 registrado como deuda técnica
> `EXP-baseline-n1` en `specs/backlog.md` (la línea base tiene n=1 seed, pendiente
> re-entrenar a ≥3 antes de reportar cualquier comparación). Pendiente: la primera variante
> real (spec-002/003) para ejercitar el framework end-to-end con más de un experimento.
> **Depende de:** `spec-000` (migración + correcciones), completado en `[DONE]` (cierre
> 2026-07-08, C1/M2/M3 verificados en GPU real), fusionado a `main`/`development` en el
> commit `9dcf4bd` (rename `modelo_itm` → `fno_co2`).
> **Objetivo:** el modelo actual (`PhysicalFNOArchitecture`, paquete `fno_co2`) queda
> fijado como **línea base**. Este spec define el proceso — no una arquitectura nueva
> específica — para modificar la arquitectura y comparar variantes contra la línea base
> con reproducibilidad y rigor académico: mismos datos, seeds múltiples, métricas
> globales ya corregidas por `spec-000` (C3), y un registro trazable de cada experimento.

---

## 0. Contexto y decisiones ya tomadas

- La línea base es el estado de `main` en el commit `9dcf4bd` (`PhysicalFNOArchitecture`
  con dropout real, scheduler coseno, AMP opt-in, métricas globales R²/RMSE — ver
  `docs/arquitectura-y-correcciones-spec-000.md`).
- El split train/test es el que produce `scripts/etl/make_split.py`
  (`train_test_split_80_20.csv`, 80/20 estratificado) — **debe reutilizarse sin
  regenerar** entre experimentos; regenerarlo invalida cualquier comparación (ver
  Fase 2).
- Hoy `scripts/train.py` solo acepta overrides por **flags CLI** sobre `Config()`; no hay
  carga de archivos YAML (`configs/default.yaml` es documentación, no está conectado a
  ningún loader) ni flag `--seed`. Ambos son requisitos de este spec (Fases 1–2).
- `training/checkpoint.py::build_run_signature` ya invalida `--auto-resume` cuando cambia
  la arquitectura o los hiperparámetros relevantes, pero usa un `model_name` hardcodeado
  (`"PhysicalFNOArchitectureRealInjection"`) — no distingue variantes (Fase 1).
- No existen checkpoints ni corridas reales todavía en este repo; la línea base se
  congela **cuando exista una corrida real completa** (Fase 0), no antes.

**Principio rector:** una variante de arquitectura es *comparable* contra la línea base
solo si difiere en **una cosa a la vez** (la arquitectura), con todo lo demás controlado:
mismos datos, mismo split, mismas seeds evaluadas, mismo número de épocas/criterio de
paro, mismas métricas calculadas de la misma forma.

---

## Fase 0 — Congelar la línea base

> **⚠️ Requiere una corrida de entrenamiento real completa (GPU + datos procesados).**
> Esta fase no se ejecuta con datos sintéticos. La precondición que la bloqueaba (C1/M2/M3
> de `spec-000`) **ya está resuelta**: se verificó en GPU real (RTX 6000 Ada) al cerrar
> `spec-000` el 2026-07-08. Falta únicamente correr la línea base con la **config default de
> `Config()`** — la corrida de verificación de `spec-000` usó `batch 16 / lr 1.6e-3`, que no
> son los defaults, por lo que no sirve como línea base congelada de este framework.

1. Confirmar que `data/processed/{train,test}/` está regenerado con la corrección **C1**
   aplicada (test normalizado con stats de train) antes de correr la línea base — si no,
   los resultados de la línea base quedarían contaminados por el mismo bug que ya se
   corrigió en código.
2. Entrenar la línea base con la config default de `Config()` hasta agotar
   `early_stopping_patience` o `epochs`, con `--seed 42` (default).
3. Respaldar `outputs/checkpoints/best.pt` fuera del árbol de trabajo (ver §Acciones
   prohibidas de `CLAUDE.md` — no sobrescribir sin respaldo).
4. Tag de git **⚠️ requiere confirmación explícita**:
   ```bash
   git tag -a baseline-v1 -m "Linea base PhysicalFNOArchitecture (spec-000 completo)"
   git push origin baseline-v1
   ```
5. Registrar en `docs/experiments.md` (creado en la Fase 5) la fila `baseline` con su
   `run_signature` completo, métricas finales y el hash del tag.

**Verificación:** `git tag` lista `baseline-v1`; `outputs/checkpoints/best.pt` de esa
corrida respaldado; `docs/experiments.md` tiene la fila `baseline`.

---

## Fase 1 — Completar el CLI para experimentación reproducible

**Dónde:** `scripts/train.py`, `src/fno_co2/config.py`,
`src/fno_co2/training/checkpoint.py`.

1. Exponer `--seed` en `scripts/train.py` (hoy falta; `Config.seed` solo tiene el
   default `42` y no es overrideable desde CLI). Sin esto, correr multi-seed requiere
   editar `config.py` por corrida — inaceptable para reproducibilidad.
2. Añadir `--experiment-name` (str, default `"baseline"`) y `--model-variant` (str,
   default `"fno_baseline"`) a `Config`/CLI. `build_run_signature` deja de hardcodear
   `model_name` y usa `cfg.model_variant` — así un checkpoint de una variante nunca se
   confunde con otra ni con la línea base al intentar `--auto-resume`.
3. `main()` (`training/loop.py`) deriva el `output_dir` efectivo como
   `outputs/<experiment_name>/seed_<seed>/` cuando `--experiment-name` se pasa
   explícitamente (si no, se mantiene el comportamiento actual con `--output-dir`
   literal, sin romper corridas existentes).

**Verificación:** `tests/unit/test_config_cli.py` (nuevo) — `--seed 123` cambia
`cfg.seed`; `run_signature["model_name"]` refleja `--model-variant`; dos corridas con
`--experiment-name exp_a --seed 1` y `--experiment-name exp_a --seed 2` escriben en
directorios de salida distintos sin colisionar.

---

## Fase 2 — Config-driven: cargador de YAML para experimentos

**Dónde:** nuevo `src/fno_co2/config.py::load_config_from_yaml`, `scripts/train.py`.

1. Añadir `pyyaml` a las dependencias core de `pyproject.toml` (**mencionar y confirmar**
   antes de instalar, según §Dependencias de `CLAUDE.md`).
2. `scripts/train.py` gana `--config <path.yaml>`: si se pasa, carga un `Config()` base
   desde el YAML (mismos campos que `configs/default.yaml`, ahora sí conectado); los
   flags CLI explícitos siguen teniendo prioridad sobre el YAML (permite overrides
   puntuales sin editar el archivo).
3. Estructura de configs de experimentos: `configs/experiments/<experiment_name>.yaml` —
   cada uno es la config completa de una variante (no un diff), para que el archivo sea
   autocontenido y reproducible por sí solo sin depender del default vigente en el
   momento de correrlo.
4. `configs/experiments/baseline.yaml` es el primer archivo de este directorio: copia
   exacta de los defaults de `Config()` en el momento del tag `baseline-v1`.

**Por qué config y no solo CLI:** un `.yaml` versionado en git es el artefacto que
documenta exactamente qué se corrió — los flags CLI se pierden si no se registran a mano
en algún lado; el archivo no.

**Verificación:** `tests/unit/test_config_yaml.py` (nuevo) — round-trip: cargar
`configs/experiments/baseline.yaml` produce un `Config()` idéntico a los defaults del
dataclass; un flag CLI (`--lr 1e-3`) sobrescribe el valor del YAML.

---

## Fase 3 — Convención de código para variantes de arquitectura

**Dónde:** `src/fno_co2/models/`.

1. La línea base (`fno.py::PhysicalFNOArchitecture`) **no se modifica in-place** para
   probar variantes estructurales (nuevo bloque, otra topología, otro mecanismo de
   condicionamiento). Cambios que sean *hiperparámetros* del modelo existente
   (`dropout_p`, `use_group_norm`, `hidden_dim`, `spectral_modes`) siguen resolviéndose
   por `Config`, sin nuevo módulo (ya es el patrón actual).
2. Una variante estructural nueva vive en su propio módulo:
   `src/fno_co2/models/variants/<nombre>.py`, con su propia clase (p. ej.
   `FNOWithAttentionBlock`). No hereda de `PhysicalFNOArchitecture` salvo que
   genuinamente comparta implementación — copiar y modificar es preferible a una
   jerarquía que acopla la línea base a los experimentos.
3. `src/fno_co2/models/registry.py` (nuevo): `build_model(cfg: Config) -> nn.Module`
   despacha por `cfg.model_variant` (`"fno_baseline"` → `PhysicalFNOArchitecture`;
   cualquier otro string → busca en `variants/`). `training/loop.py::main()` usa
   `build_model(cfg)` en vez de instanciar `PhysicalFNOArchitecture` directamente.
4. Cada variante nueva se desarrolla en su propia rama `exp/<nombre>` desde
   `development` (convención ya establecida en `CLAUDE.md` §Git), con su propio spec
   corto si el cambio es no trivial (referenciando este spec-001 como framework).

**Verificación:** `tests/unit/test_registry.py` (nuevo) — `build_model` con
`model_variant="fno_baseline"` devuelve `PhysicalFNOArchitecture`; variante desconocida
lanza `ValueError` explícito (evita silenciosamente entrenar la arquitectura equivocada).

---

## Fase 4 — Runner multi-seed

**Dónde:** nuevo `scripts/run_experiment.py`.

1. Recibe `--config configs/experiments/<name>.yaml` y `--seeds 1,2,3` (o `--n-seeds 3`
   con seeds derivadas determinísticamente, p. ej. `42, 43, 44`).
2. Por cada seed, invoca `scripts/train.py --config <path> --seed <s>
   --experiment-name <name>` como subproceso (aislamiento total entre corridas: si una
   falla — p. ej. `RuntimeError` de las guardas NaN/Inf de M6 — no aborta las demás).
3. Escribe `outputs/<name>/run_manifest.json`: lista de seeds, estado de cada una
   (`completed`/`failed`), timestamps, y el path exacto del `.yaml` usado (copiado, no
   solo referenciado, por si el archivo cambia después).
4. No reimplementa el loop de entrenamiento — reutiliza `scripts/train.py` tal cual, para
   no divergir del camino de código ya probado por los 108 tests de `spec-000`.

**Verificación:** correr con un dataset sintético mínimo (2 seeds, 1 época,
`--overfit-sample-idx 0`) produce 2 subdirectorios `outputs/<name>/seed_<s>/` con
`metrics_history.json` cada uno y un `run_manifest.json` consistente.

---

## Fase 5 — Agregación de resultados y comparación estadística

**Dónde:** nuevo `scripts/aggregate_experiments.py`, nuevo `docs/experiments.md`.

1. Por experimento, lee `metrics_history.json` de cada seed en
   `outputs/<name>/seed_*/`, toma la época del `best.pt` (mínimo `val_loss`, criterio ya
   usado por early stopping) y extrae `val_sf_r2`, `val_vd_r2`, `val_sf_rmse`,
   `val_vd_rmse`.
2. Agrega **mean ± std sobre las seeds** (mínimo 3 seeds por experimento; ver Fase 6)
   — nunca reporta una sola corrida como resultado de una variante.
3. Compara variante vs. línea base con un test no paramétrico apropiado para n pequeño
   (Mann-Whitney U o, si el número de seeds coincide entre ambos grupos, Wilcoxon
   pareado por seed) **y**, dado que con 3–5 seeds el poder estadístico es limitado,
   reporta también el tamaño de efecto (diferencia de medias en unidades de RMSE/R²) y
   los valores crudos por seed — la significancia formal es secundaria frente a la
   magnitud del efecto con n tan chico; ambas cosas se muestran, ninguna se oculta.
4. Genera `docs/experiments.md`: una tabla por experimento (`nombre`, `qué cambia vs.
   línea base`, `commit/rama`, `seeds`, `val_sf_r2 mean±std`, `val_vd_r2 mean±std`,
   `val_sf_rmse mean±std`, `val_vd_rmse mean±std`, `¿supera la línea base?`,
   `conclusión`). Se actualiza (no se sobrescribe) cada vez que se corre un experimento
   nuevo — es el registro append-only del proceso experimental.

**Verificación:** `tests/unit/test_aggregate.py` (nuevo) — con `metrics_history.json`
sintéticos de 3 seeds, la agregación reproduce mean/std calculados manualmente; el test
estadístico no falla con datos idénticos entre grupos (p-valor no significativo
esperado) y sí detecta diferencia con datos claramente separados.

---

## Fase 6 — Rigor: número mínimo de seeds y criterio de decisión

1. **Mínimo 3 seeds por variante** (línea base incluida) antes de reportar cualquier
   comparación; 5 si el resultado está cerca del umbral de decisión (mejora marginal).
2. Ninguna variante se declara "mejor" solo por una media más alta si los rangos de
   mean±std se solapan sustancialmente con la línea base — se documenta como
   "inconcluso con n=<k> seeds", no como mejora.
3. El criterio de éxito para una variante debe fijarse **antes** de correrla (p. ej.
   "reduce `val_sf_rmse` mean en ≥5% sin degradar `val_vd_r2`"), registrado en la fila de
   `docs/experiments.md` de esa variante desde que se planifica, no reescrito después de
   ver el resultado (evita *p-hacking* informal).

**Verificación:** no es automatizable — es una revisión de proceso; se aplica al llenar
`docs/experiments.md` en cada experimento nuevo.

---

## 1. Archivos impactados (resumen)

| Archivo / carpeta | Fase | Naturaleza |
|---|---|---|
| `scripts/train.py` | 1, 2 | `--seed`, `--experiment-name`, `--model-variant`, `--config` |
| `src/fno_co2/config.py` | 1, 2 | Campos nuevos + `load_config_from_yaml` |
| `src/fno_co2/training/checkpoint.py` | 1 | `model_name` dinámico en `run_signature` |
| `src/fno_co2/training/loop.py` | 1, 3 | `output_dir` derivado; usa `build_model(cfg)` |
| `src/fno_co2/models/registry.py` | 3 | Nuevo — despacho de variantes |
| `src/fno_co2/models/variants/` | 3 | Nuevo — futuras arquitecturas no-baseline |
| `configs/experiments/baseline.yaml` | 2 | Nuevo — config congelada de la línea base |
| `scripts/run_experiment.py` | 4 | Nuevo — runner multi-seed |
| `scripts/aggregate_experiments.py` | 5 | Nuevo — agregación + comparación estadística |
| `docs/experiments.md` | 5, 6 | Nuevo — registro append-only de experimentos |
| `pyproject.toml` | 2 | Añade `pyyaml` — confirmado por el usuario 2026-07-10 |
| `tests/unit/test_config_cli.py`, `test_config_yaml.py`, `test_registry.py`, `test_aggregate.py`, `conftest.py` | 1, 2, 3, 5 | Nuevos |
| `.gitignore` | — | Extendido: `outputs/<experiment_name>/` (seeds, manifest) ignorado igual que el resto de `outputs/` |
| Git: tag `baseline-v1`, ramas `exp/<variante>` | 0, 3 | Tag creado 2026-07-09 (commit `ce9cbfa`, confirmado explícitamente). Ramas `exp/<variante>` pendientes hasta spec-002/003 |

---

## 2. Riesgos y precondiciones

- **Costo de cómputo:** multi-seed multiplica el tiempo de entrenamiento por N (mínimo
  3×). Con GPU limitada, priorizar variantes con hipótesis clara sobre exploración
  amplia sin criterio (Fase 6, punto 3).
- **La línea base debe congelarse con datos reales post-C1**, no antes — si se fija la
  línea base con datos de test sin normalizar (bug pre-C1), toda comparación posterior
  hereda esa distorsión.
- **`--auto-resume` entre variantes:** `run_signature` ya protege contra resumir un
  checkpoint incompatible (aborta con motivo listado), pero **no** contra sobrescribir
  `outputs/<experiment_name>/` de una variante con otra si se reutiliza el mismo
  `experiment_name` por error — Fase 1 nombra el directorio por
  `experiment_name/seed_N`, pero la disciplina de nombres únicos sigue siendo manual.
- **Split train/test inmutable entre experimentos:** si `make_split.py` se vuelve a
  correr con otra semilla/proporción entre dos experimentos que se van a comparar, la
  comparación queda inválida (no es la arquitectura lo que cambió, son los datos). No hay
  guarda automática para esto todavía — se documenta como responsabilidad del proceso
  (Fase 6) hasta que se justifique automatizarlo.
- **`pyyaml` es una dependencia nueva** (Fase 2) — mencionada y confirmada por el usuario
  antes de instalar (2026-07-10); ya estaba presente en la imagen Docker (v6.0.3) pero sin
  declarar en `pyproject.toml`, se agregó a `[project.dependencies]`.
- ~~`docs/` está en `.gitignore`~~ — **corregido**: verificado contra el repo real
  (2026-07-10), `docs/` **no** está gitignoreado (`git ls-files docs/` lo confirma
  tracked); la nota original del spec era incorrecta. `docs/experiments.md` vive
  versionado en `development` como el resto de `docs/` (se excluye de `main` vía
  `scripts/promote-to-main.sh`, no vía `.gitignore`) — sí es visible para
  colaboradores en el repo remoto sin necesidad de reconsiderar nada.
- **`.gitignore` extendido para la nueva estructura de outputs:** `outputs/<experiment_name>/`
  (incluye `seed_<seed>/`, `run_manifest.json`, checkpoints, logs) se ignora igual que el
  resto de `outputs/` — regla `outputs/*/` con negación explícita de `checkpoints/`,
  `logs/`, `figures/` para no romper las reglas granulares ya existentes de esos tres
  subdirectorios.

---

## 3. Criterios de aceptación

- [x] `--seed`, `--experiment-name`, `--model-variant`, `--config` funcionan en
      `scripts/train.py` sin romper invocaciones existentes (flags nuevos con default
      `None`/valor actual). Verificado en `tests/unit/test_config_cli.py`.
- [x] `configs/experiments/baseline.yaml` existe y `load_config_from_yaml` lo carga
      produciendo un `Config()` idéntico a los defaults del dataclass. Verificado en
      `tests/unit/test_config_yaml.py::test_baseline_yaml_matches_config_defaults`.
- [x] `build_model(cfg)` despacha correctamente `"fno_baseline"` →
      `PhysicalFNOArchitecture`; variante desconocida falla explícito. Verificado en
      `tests/unit/test_registry.py`.
- [x] `scripts/run_experiment.py` corre N seeds de una config y produce
      `run_manifest.json` + un `metrics_history.json` por seed. Verificado con una corrida
      real (2 seeds, 1 época, `--overfit-sample-idx 0`, 2026-07-10; artefactos de la
      prueba eliminados tras verificar).
- [x] `scripts/aggregate_experiments.py` calcula mean±std sobre seeds y un test
      estadístico apropiado para n pequeño (Wilcoxon pareado si las seeds coinciden
      exactamente entre grupos, si no Mann-Whitney U), sin ocultar los valores crudos por
      seed. Verificado en `tests/unit/test_aggregate.py` y con una corrida real end-to-end.
- [x] `docs/experiments.md` tiene la fila `baseline` completa (Fase 0) antes de registrar
      cualquier variante — generada corriendo `aggregate_experiments.py` sobre la corrida
      real de Fase 0 (seed 42, best en epoch 12, `val_loss=0.012696`).
- [x] Toda la suite existente (`pytest tests/ -m "not slow"`) sigue pasando — 124 passed,
      11 deselected (2026-07-10, dentro del contenedor `fno-co2:dev`).
- [x] Ninguna variante se reporta como "mejor" en `docs/experiments.md` con menos de 3
      seeds, ni sin el criterio de éxito predefinido antes de correrla (Fase 6) —
      `compute_verdict()` fuerza el texto "inconcluso — n=<k> seeds" cuando `n_seeds < 3`,
      sin importar el resultado de la comparación; verificado en
      `test_aggregate.py::test_compute_verdict_inconclusive_below_min_seeds`.
