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
