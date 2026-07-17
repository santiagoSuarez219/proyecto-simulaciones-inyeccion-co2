# Backlog de Deuda Técnica

## ⬜ [spec-004-debt-001] `do_uncertainty` compara contra `cfg.epochs`, no contra la época real de early stopping

**Prioridad:** BAJA (no bloquea Fase 7; el `val_sf_unc`/`val_vd_unc` final puede quedar en 0
si el intervalo nunca coincide, pero eso ya pasa hoy — no es una regresión de spec-004).
**Componente:** `src/fno_co2/training/loop.py::main`, línea `do_uncertainty = ... epoch ==
cfg.epochs ...`.

### Problema

`do_uncertainty` (dispara la calibración/evaluación cara con MC-Dropout, spec-000 M/C2)
decide "época final" comparando `epoch == cfg.epochs` — el máximo **configurado** (p. ej.
100), no la época donde el early stopping realmente detiene el entrenamiento (`cfg.epochs`
nunca se reasigna a ese valor). Con `early_stopping_patience=5` y el histórico real de
`baseline` (mejores épocas 12/19/14, corridas totales ~17-24), esa rama casi nunca se
activa: el único disparo real viene del término periódico
(`epoch % uncertainty_eval_interval == 0`), que puede o no coincidir con la última época de
la corrida. Esto significa que el `val_sf_unc`/`val_vd_unc` de la fila final de
`metrics_history.json` no necesariamente refleja la incertidumbre calibrada en la mejor
época — puede ser 0.0 (nunca calculada) si el intervalo no coincidió con ninguna época de
la corrida.

**Descubierto durante:** `spec-004` Fase 7, evaluando si `uncertainty_eval_interval` podía
subirse/deshabilitarse para acelerar los timing probes (ver `spec-004` §7.0, "Ajuste
descartado"). No se tocó ningún config por este hallazgo.

### Solución propuesta (no implementada)

Que el loop reconozca "esta es la última época que se va a correr" (por early stopping o por
alcanzar `cfg.epochs`) como caso especial de `do_uncertainty=True`, en vez de comparar contra
el máximo configurado. Requiere restructurar el loop para saber de antemano (o detectar en el
momento) que el próximo `break` por early stopping va a ocurrir, y forzar el cálculo de
incertidumbre en esa época antes de salir.

### Verificación pendiente

- [ ] Repro: correr con `uncertainty_eval_interval` que no divida ninguna época del rango
      real de una corrida corta (overfit o real) y confirmar `val_sf_unc=0.0` en la fila
      final de `metrics_history.json`.
- [ ] Fix + test de regresión: la fila final de una corrida con early stopping siempre tiene
      incertidumbre calculada (`val_sf_unc > 0`), sin importar el valor de
      `uncertainty_eval_interval`.

---

## ✅ [spec-002-debt-001] Optimización GPU de U-Net temporal

**Estado:** RESUELTO (2026-07-15). El OOM era un artefacto de correr **seeds en
paralelo** (varios contenedores compartiendo la GPU + fragmentación), no un
problema por-seed. `scripts/run_experiment.py` corre las seeds **secuencialmente**,
así que el problema no aplica a la campaña real.

**Medición (sonda forward+backward, datos reales 50×50, `hidden_dim=64`, RTX 6000
Ada 48 GB):**
- `batch_size=2`: pico **3.89 GiB** (17.7M params)
- `batch_size=4`: pico **7.55 GiB**

A `batch_size=2` (el valor del YAML) hay holgura enorme; **no se necesita gradient
checkpointing** ni reducir `unet_depth`. Se corrige de paso un `batch_size`
duplicado en `configs/experiments/unet_film.yaml` (líneas 14 y 32; en YAML gana el
último → 2, pero era un bug latente). Nota: la estimación de ~6-8 h/seed del reporte
original era con `hidden_dim=128` (69.9M params); el YAML usa `hidden_dim=64`
(17.7M), sensiblemente más rápido.

<details><summary>Reporte original (histórico)</summary>

**Prioridad:** MEDIA  
**Estimado:** 4-8 horas  
**Componente:** `src/fno_co2/models/variants/unet_film.py`

### Problema
La arquitectura U-Net con expansión temporal (skips expandidos sobre T=61 timesteps) es correcta pero computacionalmente costosa:
- Entrenamiento ~6-8 horas por seed (vs. ~1.5-2 horas para baseline)
- GPU memory fragmentation con múltiples seeds paralelos
- Gradientes lentos a través del decoder

### Raíz Causa
Al expandir los skips de (B, C, H, W) a (B*T, C, H, W), se duplica el uso de memoria en el decoder. Cada UpBlock concatena y procesa esta representación expandida, incrementando el costo computacional sin beneficio de paralelización.

### Soluciones Propuestas

1. **Gradient Checkpointing (recomendado)**
   - Aplicar `torch.utils.checkpoint.checkpoint()` en UpBlock.forward()
   - Trade-off: 20-30% más lento pero 40-50% menos memoria
   - Permite batch_size=4 nuevamente

2. **Refactor de skip connections**
   - No expandir skips, procesarlos per-timestep en el decoder
   - Requiere reescribir el loop del decoder
   - Potencial mejora: 3-4x más rápido

3. **Reducir profundidad de U-Net**
   - Cambiar `unet_depth=3` a `unet_depth=2`
   - Pierde capacidad receptiva pero más rápido
   - No recomendado: arquitectura menos capaz

### Verificación
- [ ] Implementar gradient checkpointing
- [ ] Benchmark: memoria, velocidad vs. baseline
- [ ] Re-entrenar multi-seed (3 seeds, ~2 horas total)
- [ ] Verificar que las métricas no degraden

### Notas
- Arquitectura es estructuralmente correcta (tests pasan 135/135)
- Problema es de rendimiento, no de correción
- Solución #2 (refactor) es la más elegante pero requiere más trabajo

</details>

---

## ✅ [spec-002-debt-002] Investigar convergencia deficiente

**Estado:** RESUELTO — mecanismo (2026-07-15). Diagnóstico con overfit de 1 muestra
(gate de Fase 5.1) + fix aplicado + humo en verde. Falta solo la confirmación
multi-seed real con GPU (Fase 5.3, gated por el usuario).

### Causa raíz (dos bugs independientes)

1. **Explosión en la inicialización.** El `_init_weights` previo (commit `e1a4091`,
   introducido *para* arreglar la convergencia y que la empeoró) aplicaba
   `kaiming_normal_(fan_out, relu)` a **todo**, incluidos los `gamma`/`beta` de FiLM.
   FiLM es `x·(1+γ)+β`; con γ,β aleatorios grandes la modulación amplifica la señal,
   y eso se **compone a través de los 3 UpBlock** → el forward explota ya en la
   inicialización (overfit: `train_loss` E1 = **986**, `sf_rmse` = 860 con targets en
   `[0,1]`). Luego el 1er paso de AdamW dispara la loss a millones.
2. **LR demasiado alto.** La U-Net (sin normalización, profunda) es inestable al
   `lr=8e-4` del baseline FNO: AdamW normaliza el update por-parámetro, así que el
   1er paso mueve cada peso ~`±lr` sin importar `grad_clip`. Con el init arreglado
   pero `lr=8e-4` seguía divergiendo; a `lr=1e-4` oscilaba; a **`lr=3e-5`** converge
   suave.

> El `val_sf_r2=0.116` original es de **datos completos, 1 época** (gradientes que
> varían por batch promedian y evitan la explosión que sí aparece en el overfit de 1
> muestra), y de **antes** del `_init_weights` Kaiming. Con init por defecto
> sub-ajustaba; el "fix" Kaiming lo volvió divergente. Ambos quedan corregidos.

### Fix aplicado
- **`src/fno_co2/models/variants/unet_film.py::_init_weights`:** Kaiming `fan_in`
  en convs/linears activos; **FiLM `gamma`/`beta` a cero** (modulación identidad al
  inicio); **última conv de cada `ResBlock` escalada a 0.1** (rama residual
  casi-identidad, variante de Fixup; **no** exactamente cero para mantener vivo el
  `Dropout2d` intermedio → MC Dropout sigue activo, contrato §2.4).
- **`configs/experiments/unet_film.yaml`:** `lr: 8e-4 → 3e-5` (hiperparámetro
  por-variante; datos/split/seeds/loss/métricas siguen idénticos al baseline).

### Verificación
- [x] Overfit de 1 muestra baja la loss de forma monótona y estable, con la config
      real del YAML (`h_dim=64`, `lr=3e-5`): `train_loss` E1=5.26 → E40=0.55,
      `sf_rmse` 0.31 → 0.036. Antes: divergía a millones.
- [x] `pytest tests/ -m "not slow"` en verde (135 passed), incl. `test_mc_dropout_active`.
- [ ] **Pendiente (Fase 5.3, GPU + confirmación del usuario):** corrida multi-seed
      real ≥3 seeds y comprobar `val_sf_r2` cercano al baseline en datos completos.

<details><summary>Reporte original (histórico)</summary>

**Prioridad:** MEDIA  
**Estimado:** 2-4 horas  
**Componente:** `src/fno_co2/models/variants/unet_film.py` + training

### Problema
Incluso con inicialización Kaiming y arquitectura correcta, el modelo converge lentamente:
- Seed 42 (128 hidden): `val_sf_r2 = 0.116` después de 1 época
- Seed 42 (64 hidden): `val_sf_r2 = 0.085`
- Comparación: baseline consigue `val_sf_r2 = 0.994`

### Raíz Causa Probable
Uno de:
1. FiLM modulation aplicada en lugar subóptimo
2. Escalado de gradientes en el decoder
3. Skip connections no siendo aprovechadas
4. Loss landscape más compleja que el FNO baseline

### Verificación
- [ ] Visualizar activaciones
- [ ] Probar FiLM en diferentes puntos
- [ ] Comparar gradientes entre U-Net y baseline
- [ ] Intentar arquitectura simplificada

</details>

