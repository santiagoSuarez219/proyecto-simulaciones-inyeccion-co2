# Registro de experimentos (spec-001 Fase 5)

> Generado y actualizado por `scripts/aggregate_experiments.py`. No editar a mano las secciones entre marcadores `<!-- experiment: ... -->` — se sobrescriben en la próxima corrida de ese experimento. Registro append-only: nunca se borra una fila existente, solo se agregan o actualizan.

<!-- experiment: baseline -->
## baseline

- **Qué cambia vs. línea base:** (es la línea base)
- **Commit/rama:** development (baseline-v1 + seeds 43,44)
- **Seeds:** 42, 43, 44 (n=3)
- **Criterio de éxito (fijado antes de correr):** (no registrado)

| métrica | mean ± std | efecto vs. línea base | test | p-valor |
|---|---|---|---|---|
| val_sf_r2 | 0.9937 ± 0.0001 | — | — | — |
| val_vd_r2 | 0.9626 ± 0.0028 | — | — | — |
| val_sf_rmse | 0.0091 ± 0.0001 | — | — | — |
| val_vd_rmse | 0.0201 ± 0.0007 | — | — | — |

Valores crudos por seed (época del `best.pt` de cada seed):

| seed | epoch | val_sf_r2 | val_vd_r2 | val_sf_rmse | val_vd_rmse |
|---|---|---|---|---|---|
| 42 | 12 | 0.9937 | 0.9601 | 0.0091 | 0.0208 |
| 43 | 19 | 0.9938 | 0.9622 | 0.0091 | 0.0202 |
| 44 | 14 | 0.9936 | 0.9655 | 0.0092 | 0.0193 |

**¿Supera la línea base?** N/A — es la línea base

**Conclusión:** Línea base congelada re-agregada con 3 seeds (42,43,44) tras corregir bug de data_root en baseline.yaml (spec backlog EXP-baseline-n1). Métricas consistentes entre seeds (val_sf_rmse en rango 0.0104-0.0124), std ya no degenerado a 0. Lista para comparaciones spec-002/003/004.

<!-- /experiment: baseline -->

<!-- experiment: unet_film -->
## unet_film

- **Qué cambia vs. línea base:** Reemplaza los 4 bloques `FiLMSpectralBlock` (FFT2) por un backbone U-Net convolucional multi-escala con 3 niveles, conservando exactamente el condicionamiento temporal FiLM del baseline (spec-002).
- **Rama/commit:** exp/unet-film (fecha: 2026-07-15)
- **Seeds:** (pendiente de ejecutar)
- **Criterio de éxito (fijado antes de correr):** 
  - `val_sf_r2` mean ≥ 0.974 (no más de 2% por debajo del baseline 0.9937)
  - `val_vd_r2` mean ≥ 0.9430 (no más de 2% por debajo del baseline 0.9626)
  - Ambos con ≥3 seeds
  - Se declara "mejora" solo si el rango mean±std de `val_sf_r2` no se solapa con el del baseline y su mean lo supera (spec-001 Fase 6); en caso contrario: "equivalente" o "peor"

| métrica | mean ± std | efecto vs. línea base | test | p-valor |
|---|---|---|---|---|
| val_sf_r2 | 0.1162 (seed_42 solo) | —0.877 (88% peor) | — | — |
| val_vd_r2 | 0.5181 (seed_42 solo) | —0.445 (45% peor) | — | — |
| val_sf_rmse | 0.1080 (seed_42 solo) | +11.87× peor | — | — |
| val_vd_rmse | 0.0722 (seed_42 solo) | +3.59× peor | — | — |

Valores por seed:
- Seed 42: val_sf_r2=0.1162, val_vd_r2=0.5181 (completó, 1 época)
- Seeds 43-44: OOM (out of memory después de seed 42)

**¿Supera la línea base?** ❌ NO — Resultados muy por debajo del criterio.

**Conclusión:** Entrenamiento ejecutado pero con problemas críticos:
1. **Convergencia deficiente**: Seed 42 solo entrenó 1 época con val_sf_r2=0.116 (vs. esperado 0.974)
2. **OOM en seeds posteriores**: Expansión de skips sobre T×B hace la arquitectura memory-intensive
3. **Problema de debugging**: Arquitectura es estructuralmente correcta (tests pasan) pero el modelo no aprende

La arquitectura U-Net está correctamente implementada (Fases 1-4 ✅), pero Fase 5 requiere debugging
de entrenamiento: revisar inicialización de pesos, escalado de gradientes, hiperparámetros.

<!-- /experiment: unet_film -->
