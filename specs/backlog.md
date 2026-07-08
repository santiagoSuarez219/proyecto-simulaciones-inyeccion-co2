# Backlog — deuda técnica y hallazgos

> Registro de deuda técnica (`# DEBT:` en código) y hallazgos fuera del scope inmediato.
> Los ítems resueltos se conservan como historial, marcados con ✅.

---

## ✅ M2-AMP — `GradScaler.unscale_` incompatible con parámetros `ComplexFloat` del FNO

- **Estado:** RESUELTO (2026-07-06). Detectado en el preflight de entrenamiento y corregido
  en la misma sesión.
- **Origen:** spec-000, hallazgo **M2** (mixed precision). El spec lo marcó como "no
  ejercido en hardware real" porque no había GPU en las sesiones de implementación.
- **Síntoma:** con `--use-amp true` en CUDA, `run_one_epoch` crasheaba en
  `scaler.unscale_(optimizer)` con
  `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'ComplexFloat'`.
- **Causa raíz:** los parámetros espectrales de `FiLMSpectralBlock` son `ComplexFloat`.
  El path AMP original usaba `float16` + `GradScaler`, cuyo `unscale_` recorre todos los
  gradientes del optimizador (incluidos los complejos) y no soporta ese dtype.
- **Fix aplicado (`src/fno_co2/training/loop.py`):** la ruta AMP usa **bfloat16**
  (`_AMP_DTYPE = torch.bfloat16`) **sin `GradScaler`**. bf16 tiene el mismo rango
  dinámico que float32, así que no requiere loss scaling y evita por completo el
  `unscale_` problemático. El `GradScaler` queda como no-op transparente (`enabled=False`)
  para no alterar el flujo de guardas NaN/Inf (M6). La FFT ya se fuerza a float32 dentro
  de `FiLMSpectralBlock`, independiente del dtype de autocast.
- **Verificación:** `tests/unit/test_amp.py` — `test_amp_dtype_is_bfloat16` (CPU) y
  `test_run_one_epoch_with_amp_on_cuda_handles_complex_params` (marcado `slow`+skipif-CUDA,
  regresión del bug). Ejecutado en **RTX 6000 Ada**: `run_one_epoch` con `use_amp=True`
  completa con loss finita. Suite completa: 110 passed.

---

## ✅ A3-bis — Incertidumbre MC-Dropout acoplada al loop hacía cada época ~2.3h

- **Estado:** RESUELTO (2026-07-06). Detectado al cronometrar la primera época real.
- **Origen:** extensión de spec-000 **A3**. A3 puso la calibración tras `model_has_dropout`,
  pero con dropout activo (`dropout_p=0.1`) la incertidumbre corría **completa cada época**:
  calibración inicial (~68 min) + `calibrate_uncertainty` por época (~68 min) + las
  `uncertainty_passes=30` pasadas internas de `evaluate_epoch` (~68 min) → **~2.3 h/época**,
  inviable para 100 épocas.
- **Fix aplicado (`src/fno_co2/training/loop.py`, `config.py`, `scripts/train.py`):**
  - `evaluate_epoch` computa `val_loss`/R²/RMSE **siempre con un forward determinista**
    (dropout off), así la selección de `best.pt` es consistente entre épocas y no depende
    del MC estocástico. Nuevo parámetro `compute_uncertainty` controla si además se corren
    las pasadas MC.
  - La incertidumbre (calibración + resumen) pasa a ser **diagnóstico periódico**: solo
    cada `Config.uncertainty_eval_interval` épocas (default 10) y en la época final.
  - Calibración inicial **perezosa** (se carga de disco si existe; ya no se calibra al
    arrancar).
  - Nuevo flag `--uncertainty-eval-interval`.
- **Verificación:** `tests/unit/test_training_loop.py` — `val_loss`/R²/RMSE **idénticos**
  con `compute_uncertainty` True/False (selección consistente); incertidumbre trivial
  (0.0/1.0) cuando no se computa. Suite: 112 passed.
- **Nota:** las épocas de incertidumbre siguen costando ~2.3h (30 pasadas × val, dos veces).
  Para corridas donde la incertidumbre no es el foco (p. ej. validar C1), usar
  `--uncertainty-eval-interval 0` (solo la época final) o subir el intervalo.

---

## Hardware / entorno de entrenamiento (notas del preflight, 2026-07-06)

- **`data/` es un symlink a disco externo** (`/media/.../DATA3/...`). `docker/run.sh` monta
  su destino real automáticamente (fix aplicado). Tenerlo presente si se cambia de máquina.
- **Bug preexistente en `docker/run.sh`:** el índice de GPU se leía de `$2` (que es la
  sesión) en vez de `$3`. Corregido a `${3:-all}`.
- **Techo de VRAM medido** (batch, sin AMP, step completo, RTX 6000 Ada 48 GB): batch 16 =
  70%, batch 24 = 93%, batch 32 = OOM. Batch seguro recomendado: **16**.
- **Grilla real 50×50** y **362 timesteps mensuales** (~30 años); el modelo usa los
  primeros 61 (~5 años). La tabla de dimensiones de `CLAUDE.md` (100×100, NZ=20) está
  desactualizada respecto a los datos actuales.
</content>
