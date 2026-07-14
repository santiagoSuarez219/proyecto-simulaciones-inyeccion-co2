# Registro de experimentos (spec-001 Fase 5)

> Generado y actualizado por `scripts/aggregate_experiments.py`. No editar a mano las secciones entre marcadores `<!-- experiment: ... -->` — se sobrescriben en la próxima corrida de ese experimento. Registro append-only: nunca se borra una fila existente, solo se agregan o actualizan.

<!-- experiment: baseline -->
## baseline

- **Qué cambia vs. línea base:** (es la línea base)
- **Commit/rama:** ce9cbfa (tag baseline-v1)
- **Seeds:** 42 (n=1)
- **Criterio de éxito (fijado antes de correr):** (no registrado)

| métrica | mean ± std | efecto vs. línea base | test | p-valor |
|---|---|---|---|---|
| val_sf_r2 | 0.9937 ± 0.0000 | — | — | — |
| val_vd_r2 | 0.9601 ± 0.0000 | — | — | — |
| val_sf_rmse | 0.0091 ± 0.0000 | — | — | — |
| val_vd_rmse | 0.0208 ± 0.0000 | — | — | — |

Valores crudos por seed (época del `best.pt` de cada seed):

| seed | epoch | val_sf_r2 | val_vd_r2 | val_sf_rmse | val_vd_rmse |
|---|---|---|---|---|---|
| 42 | 12 | 0.9937 | 0.9601 | 0.0091 | 0.0208 |

**¿Supera la línea base?** N/A — es la línea base

**Conclusión:** Línea base congelada (spec-001 Fase 0). Entrenada con config default de Config(), seed 42, hasta early stopping en epoch 17 (best en epoch 12).

<!-- /experiment: baseline -->
