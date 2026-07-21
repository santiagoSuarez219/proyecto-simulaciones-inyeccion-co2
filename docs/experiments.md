# Registro de experimentos (spec-001 Fase 5)

> Generado y actualizado por `scripts/aggregate_experiments.py`. No editar a mano las secciones entre marcadores `<!-- experiment: ... -->` — se sobrescriben en la próxima corrida de ese experimento. Registro append-only: nunca se borra una fila existente, solo se agregan o actualizan.

<!-- experiment: baseline -->
## baseline

- **Qué cambia vs. línea base:** (es la línea base)
- **Commit/rama:** fno_vs_unet_vs_attn
- **Seeds:** 42, 43, 44 (n=3)
- **Criterio de éxito (fijado antes de correr):** referencia (linea base); no se evalua contra si misma

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

**Conclusión:** (pendiente)

<!-- /experiment: baseline -->

<!-- experiment: unet_film -->
## unet_film

- **Qué cambia vs. línea base:** (pendiente de documentar)
- **Commit/rama:** fno_vs_unet_vs_attn
- **Seeds:** 42, 43, 44 (n=3)
- **Criterio de éxito (fijado antes de correr):** val_sf_r2 >= 0.974 (guard: val_vd_r2 >= 0.943)

| métrica | mean ± std | efecto vs. línea base | test | p-valor |
|---|---|---|---|---|
| val_sf_r2 | 0.9920 ± 0.0002 | -0.0017 | wilcoxon | 0.2500 |
| val_vd_r2 | 0.9650 ± 0.0010 | +0.0024 | wilcoxon | 0.5000 |
| val_sf_rmse | 0.0103 ± 0.0001 | +0.0012 | wilcoxon | 0.2500 |
| val_vd_rmse | 0.0195 ± 0.0003 | -0.0007 | wilcoxon | 0.5000 |

Valores crudos por seed (época del `best.pt` de cada seed):

| seed | epoch | val_sf_r2 | val_vd_r2 | val_sf_rmse | val_vd_rmse |
|---|---|---|---|---|---|
| 42 | 19 | 0.9922 | 0.9640 | 0.0101 | 0.0197 |
| 43 | 14 | 0.9919 | 0.9661 | 0.0103 | 0.0192 |
| 44 | 20 | 0.9918 | 0.9649 | 0.0104 | 0.0195 |

**¿Supera la línea base?** cumplido

**Conclusión:** (pendiente)

<!-- /experiment: unet_film -->

<!-- experiment: fno_axial_attn -->
## fno_axial_attn

- **Qué cambia vs. línea base:** (pendiente de documentar)
- **Commit/rama:** fno_vs_unet_vs_attn
- **Seeds:** 42, 43, 44 (n=3)
- **Criterio de éxito (fijado antes de correr):** val_sf_rmse <= 0.00864 (guard: val_vd_r2 >= 0.9598)

| métrica | mean ± std | efecto vs. línea base | test | p-valor |
|---|---|---|---|---|
| val_sf_r2 | 0.9874 ± 0.0042 | -0.0063 | wilcoxon | 0.2500 |
| val_vd_r2 | 0.9498 ± 0.0141 | -0.0128 | wilcoxon | 0.2500 |
| val_sf_rmse | 0.0128 ± 0.0021 | +0.0037 | wilcoxon | 0.2500 |
| val_vd_rmse | 0.0232 ± 0.0032 | +0.0031 | wilcoxon | 0.2500 |

Valores crudos por seed (época del `best.pt` de cada seed):

| seed | epoch | val_sf_r2 | val_vd_r2 | val_sf_rmse | val_vd_rmse |
|---|---|---|---|---|---|
| 42 | 6 | 0.9910 | 0.9335 | 0.0109 | 0.0268 |
| 43 | 2 | 0.9828 | 0.9591 | 0.0151 | 0.0210 |
| 44 | 6 | 0.9883 | 0.9568 | 0.0124 | 0.0216 |

**¿Supera la línea base?** no cumplido (val_sf_rmse no cumple <= 0.00864; guard val_vd_r2 no cumple >= 0.9598)

**Conclusión:** (pendiente)

<!-- /experiment: fno_axial_attn -->
