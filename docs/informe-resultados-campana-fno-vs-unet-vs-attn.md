# Informe de resultados — Campaña `fno_vs_unet_vs_attn`

> **Campaña:** `fno_vs_unet_vs_attn` · **Commit evaluado:** `0e2b03f` (limpio) · **Fecha del
> informe:** 2026-07-21 · **Spec:** `spec-006` (este documento) sobre resultados producidos
> por `spec-004` (orquestación de campañas).
>
> Este es un informe **curado**, pensado para lectura de principio a fin por investigadores
> de geomecánica y CCS. Complementa —no reemplaza— el reporte de máquina
> `outputs/campaigns/fno_vs_unet_vs_attn/campaign_report.md` y el registro detallado
> `docs/experiments.md`, de donde provienen las cifras citadas aquí (ver §10,
> Reproducibilidad).

---

## 1. Resumen ejecutivo

Se compararon tres arquitecturas para predecir la evolución espacio-temporal del Factor de
Seguridad (SF) y la Deformación Volumétrica (VD) bajo inyección de CO₂: la línea base
**FNO+FiLM** (`baseline`), una variante **U-Net con condicionamiento FiLM temporal**
(`unet_film`) y una variante **FNO + atención espacial axial** (`fno_axial_attn`). Cada
arquitectura se entrenó con 3 semillas independientes sobre el mismo split de datos.

**Resultado en una frase:** la línea base FNO+FiLM sigue siendo la referencia de mejor
desempeño; `unet_film` alcanza una **paridad** de desempeño (no una mejora, pero tampoco una
degradación relevante) con una arquitectura estructuralmente distinta; `fno_axial_attn`
**no** cumplió su criterio de éxito predefinido y mostró mayor varianza entre semillas.

| Variante | val_sf_r2 | val_vd_r2 | Veredicto |
|---|---|---|---|
| `baseline` | 0.9937 ± 0.0001 | 0.9626 ± 0.0028 | referencia |
| `unet_film` | 0.9920 ± 0.0002 | 0.9650 ± 0.0010 | equivalente a la línea base |
| `fno_axial_attn` | 0.9874 ± 0.0042 | 0.9498 ± 0.0141 | no cumple el criterio predefinido |

---

## 2. Contexto del problema

El modelo predice la evolución temporal del **Factor de Seguridad (SF)** y la
**Deformación Volumétrica (VD)** en una grilla 2D por capa de un reservorio depletado
sometido a **inyección de CO₂** (almacenamiento geológico de carbono, CCS). SF y VD son
indicadores geomecánicos: SF cuantifica el margen de estabilidad frente a falla del
reservorio (valores más bajos, mayor riesgo) y VD cuantifica la deformación volumétrica
inducida por el cambio de presión de poro.

La entrada del modelo combina **propiedades estáticas** del reservorio por capa
(permeabilidad, porosidad, cohesión, AFI, profundidad) con **series temporales de
inyección** de los pozos TENE-1 y TENE-2, y produce la evolución de SF y VD a lo largo de
61 pasos temporales. Este tipo de modelo sustituye simulaciones numéricas costosas (CMG)
por una inferencia rápida, útil para exploración de escenarios de inyección y evaluación
preliminar de riesgo geomecánico.

---

## 3. Arquitecturas comparadas

Las tres variantes comparten el mismo condicionamiento temporal (embedding de paso
temporal + MLP sobre `[inyección TENE-1, inyección TENE-2, profundidad]` inyectado vía
FiLM) y el mismo régimen de datos/split/pérdida — **solo cambia el núcleo de la
arquitectura**, para que la comparación aísle ese único factor.

### `baseline` — FNO + FiLM (`PhysicalFNOArchitecture`)

Encoder convolucional → 4 bloques `FiLMSpectralBlock` (FFT2 → multiplicación espectral con
modos de Fourier truncados → iFFT2, más una rama convolucional local 1×1 y modulación FiLM)
→ decoder convolucional. El núcleo espectral captura acoplamiento **global** de baja
frecuencia entre puntos de la grilla vía los modos de Fourier truncados
(`spectral_modes=16`).

### `unet_film` — U-Net con FiLM temporal

Reemplaza el núcleo espectral por un backbone **U-Net convolucional** (encoder-decoder con
*skip connections* multi-escala), conservando el mismo mecanismo de condicionamiento FiLM.
**Hipótesis motivadora:** materializar la arquitectura que da nombre al paper en redacción
(*"A U-Net Approach..."*) y medir si un backbone convolucional multi-escala iguala o supera
al núcleo espectral del baseline. Requirió un ajuste de `lr` (8e-4 → 3e-5): sin
normalización de capas, la U-Net (~70 M de parámetros) es inestable al `lr` del baseline
(diagnóstico completo en `specs/backlog.md`, `spec-002-debt-002`, ya resuelto).

### `fno_axial_attn` — FNO + atención espacial axial

Conserva los `FiLMSpectralBlock` del baseline **y añade** bloques de atención espacial
axial (self-attention sobre la grilla H×W, factorizada en filas y columnas para evitar el
costo O(N²) de la atención densa sobre 100×100 = 10 000 posiciones). **Hipótesis
motivadora:** los modos espectrales truncados descartan las frecuencias altas; la atención
ofrece un mecanismo de acoplamiento global complementario y adaptativo que podría capturar
lo que el truncamiento espectral pierde. Es un cambio **aditivo** sobre el baseline (no un
reemplazo), pensado para aislar el efecto de "añadir atención".

---

## 4. Dataset y metodología

- **Split:** 90/10 estratificado train/test (`scripts/etl/make_split.py`), fijo para las
  tres variantes — checksum `f51dfe25…529c95d` (§10) garantiza que las tres entrenaron y
  validaron sobre exactamente el mismo split.
- **Normalización:** min-max global `[0,1]` calculada **solo** sobre `train/` y aplicada
  también a `test/`, sin fuga de datos (corrección C1 de `spec-000`).
- **Semillas:** 3 por variante (42, 43, 44), entrenamiento independiente de principio a
  fin cada una.
- **Criterio de éxito predefinido por variante (fijado antes de correr, anti p-hacking):**
  - `unet_film`: `val_sf_r2 ≥ 0.974` (guard `val_vd_r2 ≥ 0.943`).
  - `fno_axial_attn`: `val_sf_rmse ≤ 0.00864` (guard `val_vd_r2 ≥ 0.9598`).
  - `baseline` es la referencia; no se evalúa contra sí misma.
- **Métricas:** R² y RMSE de validación para SF y VD, en la época del `best.pt` de cada
  semilla (menor `val_loss`).
- **Comparación estadística:** test no paramétrico (Wilcoxon/Mann-Whitney) por métrica,
  `unet_film`/`fno_axial_attn` vs. `baseline`, sobre los 3 valores por semilla.
- **Umbral de evidencia:** con menos de 3 semillas ningún veredicto se declara "cumplido"
  (`MIN_SEEDS_FOR_VERDICT=3`); las tres variantes completaron sus 3 semillas.

---

## 5. Resultados: métricas de rendimiento

*(Fuente: `outputs/campaigns/fno_vs_unet_vs_attn/campaign_report.md`, §Resumen — campaña del
commit `0e2b03f`.)*

| variante | n_seeds | val_sf_r2 | val_vd_r2 | val_sf_rmse | val_vd_rmse | criterio predefinido | veredicto |
|---|---|---|---|---|---|---|---|
| baseline | 3 | 0.9937 ± 0.0001 | 0.9626 ± 0.0028 | 0.0091 ± 0.0001 | 0.0201 ± 0.0007 | referencia (línea base) | N/A — es la línea base |
| unet_film | 3 | 0.9920 ± 0.0002 | 0.9650 ± 0.0010 | 0.0103 ± 0.0001 | 0.0195 ± 0.0003 | val_sf_r2 ≥ 0.974 (guard: val_vd_r2 ≥ 0.943) | cumplido |
| fno_axial_attn | 3 | 0.9874 ± 0.0042 | 0.9498 ± 0.0141 | 0.0128 ± 0.0021 | 0.0232 ± 0.0032 | val_sf_rmse ≤ 0.00864 (guard: val_vd_r2 ≥ 0.9598) | no cumplido |

### Curvas de convergencia comparadas

![Curvas de convergencia comparadas: val_loss, val_sf_r2 y val_vd_r2 por época, las tres variantes superpuestas con banda ±std entre semillas](figures/campana-fno-vs-unet-vs-attn/campaign_convergence_curves.png)

*Media entre semillas por época, con banda ±std. Las semillas paran en épocas distintas
por early stopping (`baseline`: 17/19/24; `unet_film`: 19/22/25; `fno_axial_attn`: 7/8/11),
así que la banda se calcula solo sobre las semillas presentes en cada época — más allá de
la época 8, `fno_axial_attn` queda representada por una sola semilla (sin banda visible).
El eje Y usa un rango robusto por percentil: `fno_axial_attn/seed_44` tuvo un pico de
inestabilidad de una sola época (época 3, recuperado en la época 4) que de otro modo
aplastaría la escala completa; el pico no se recorta como dato, solo queda fuera del marco
visible.*

### Métricas finales por variante

![Barras de métricas finales (mean±std) por variante: val_sf_r2, val_vd_r2, val_sf_rmse, val_vd_rmse](figures/campana-fno-vs-unet-vs-attn/campaign_final_metrics.png)

*Mismos valores de la tabla anterior, en la época del `best.pt` de cada semilla.*

---

## 6. Análisis estadístico vs. línea base

*(Fuente: `campaign_report.md`, §Comparación estadística — test de Wilcoxon, n=3 por
grupo.)*

| variante | métrica | efecto | test | p-valor |
|---|---|---|---|---|
| unet_film | val_sf_r2 | −0.0017 | wilcoxon | 0.2500 |
| unet_film | val_vd_r2 | +0.0024 | wilcoxon | 0.5000 |
| unet_film | val_sf_rmse | +0.0012 | wilcoxon | 0.2500 |
| unet_film | val_vd_rmse | −0.0007 | wilcoxon | 0.5000 |
| fno_axial_attn | val_sf_r2 | −0.0063 | wilcoxon | 0.2500 |
| fno_axial_attn | val_vd_r2 | −0.0128 | wilcoxon | 0.2500 |
| fno_axial_attn | val_sf_rmse | +0.0037 | wilcoxon | 0.2500 |
| fno_axial_attn | val_vd_rmse | +0.0031 | wilcoxon | 0.2500 |

Con n=3 semillas por grupo, el mínimo p-valor posible de un test de rangos con signo es
0.25 — **ningún p-valor de esta tabla puede alcanzar significancia estadística
convencional** (p<0.05) con este tamaño de muestra. Los p-valores no son concluyentes por
sí solos y deben leerse junto con el tamaño de efecto y el criterio de
no-solapamiento de rangos mean±std aplicado en §7.

---

## 7. Interpretación y discusión

**`unet_film` — paridad, no mejora.** El criterio predefinido (`val_sf_r2 ≥ 0.974`) se
cumple con holgura amplia (0.9920 vs. umbral 0.974), y el veredicto mecánico es
"cumplido". Sin embargo, comparado directamente con `baseline`, el rango mean±std de
`val_sf_r2` de `unet_film` ([0.9918, 0.9922]) **no se solapa** con el de `baseline`
([0.9936, 0.9938]) y su media queda **por debajo**. Por la regla de no-solapamiento
adoptada en `spec-002`, la lectura correcta es que `unet_film` es una **alternativa
arquitectónica viable, en paridad de desempeño con la línea base — no una mejora**. En
`val_vd_r2` sí hay una ligera ventaja de `unet_film` (0.9650 vs. 0.9626), pero el efecto es
pequeño y el p-valor (0.5000) no aporta evidencia adicional.

**`fno_axial_attn` — no cumple, y con más varianza.** El criterio predefinido
(`val_sf_rmse ≤ 0.00864`) no se cumple (0.0128 ± 0.0021, muy por encima del umbral) y el
guard tampoco (`val_vd_r2 ≥ 0.9598` vs. 0.9498 ± 0.0141 observado). La hipótesis de que la
atención espacial axial complementaría los modos espectrales truncados **no se confirma**
con esta configuración: el desempeño es inferior al baseline en las cuatro métricas, y la
dispersión entre semillas (`std` de `val_sf_rmse` = 0.0021, más del doble que `unet_film` o
`baseline`) sugiere una arquitectura menos estable con los mismos hiperparámetros
heredados del baseline (`lr=8e-4`).

**Observación adicional: early stopping muy temprano en `fno_axial_attn`.** Sus tres
semillas pararon en las épocas 7, 8 y 11 — sensiblemente antes que `baseline` (17-24) y
`unet_film` (19-25). Esto es consistente con una arquitectura más sensible al `lr`
heredado del baseline (a diferencia de `unet_film`, que sí requirió y recibió un ajuste de
`lr` propio — §3). El pico de inestabilidad de una sola época observado en
`fno_axial_attn/seed_44` (§5, nota de la figura) refuerza esta lectura. **Línea de
investigación futura, no ejecutada en esta campaña:** repetir `fno_axial_attn` con un
`lr` más bajo, análogo al ajuste que ya benefició a `unet_film`, antes de descartar la
hipótesis de la atención axial por completo.

---

## 8. Limitaciones

- **n=3 semillas por variante.** Suficiente para cruzar el umbral mínimo de evidencia del
  framework (`MIN_SEEDS_FOR_VERDICT=3`), pero insuficiente para significancia estadística
  convencional (§6) — las conclusiones se apoyan en tamaño de efecto y no-solapamiento de
  rangos, no en p-valores.
- **Una sola campaña, un solo split.** Los resultados no se han replicado con un split de
  datos distinto ni con datos adicionales.
- **Datos sintéticos.** El dataset proviene de simulaciones CMG, no de mediciones de campo;
  las conclusiones son válidas dentro del régimen de esos escenarios sintéticos.
- **`fno_axial_attn` no probó un `lr` propio.** A diferencia de `unet_film`, se evaluó con
  el `lr` del baseline sin ajuste — el veredicto "no cumple" refleja esa configuración
  concreta, no descarta la arquitectura bajo otro régimen de entrenamiento (§7).

### Salvedad sobre la incertidumbre MC-Dropout (no evaluada en este informe)

Este informe **no reporta cifras de incertidumbre MC-Dropout** (`val_sf_uncertainty_mean`
/ `val_vd_uncertainty_mean`) porque, por decisión explícita, la incertidumbre es
**secundaria** frente a las métricas de desempeño (R²/RMSE) para este informe, y además
existe una deuda técnica conocida que afecta su cálculo:

- `do_uncertainty` en `training/loop.py` solo dispara el cálculo de incertidumbre en
  épocas múltiplo de `uncertainty_eval_interval=10`. Como las 9 corridas de esta campaña
  paran por early stopping antes de alcanzar un múltiplo exacto en varios casos, **la fila
  final de `metrics_history.json` de las 9 corridas tiene incertidumbre = 0.0 — es un
  artefacto de esta deuda, no una medición** (`spec-004-debt-001`, backlog, prioridad
  BAJA — sigue abierta).
- Auditoría realizada sobre los datos reales de esta campaña (tomando la última época con
  `val_sf_uncertainty_mean > 0` de cada semilla, no la fila final): `baseline` ≈ 0.30-0.32;
  `unet_film` ≈ 0.34-0.35. **`fno_axial_attn` solo tiene un dato válido** (`seed_44` ≈
  0.33); `seed_42` y `seed_43` pararon (épocas 8 y 7) antes de alcanzar la primera época
  calibrada (10), así que no hay muestra suficiente (n=1) para comparar la incertidumbre
  de esta variante con las otras dos.
- **Esta deuda no afecta a ninguna cifra de este informe:** las métricas de R²/RMSE (§5-6)
  se calculan y registran en cada época, no dependen de `uncertainty_eval_interval`, y son
  plenamente fiables.

---

## 9. Conclusiones

1. **`baseline` (FNO+FiLM) sigue siendo la arquitectura de referencia**: mejor `val_sf_r2`
   y menor dispersión entre semillas que las dos alternativas evaluadas.
2. **`unet_film` es una alternativa arquitectónica viable**, en paridad de desempeño con
   el baseline (no una mejora), relevante porque materializa la arquitectura U-Net que da
   nombre al paper en redacción sin sacrificar precisión frente al FNO espectral.
3. **`fno_axial_attn`, en su configuración actual, no justifica el costo de añadir
   atención espacial**: no cumple su criterio predefinido, tiene mayor varianza entre
   semillas y para de entrenar sensiblemente antes que las otras dos variantes — un
   indicio de sensibilidad al `lr` heredado del baseline, no evaluado con un `lr` propio en
   esta campaña.

---

## 10. Reproducibilidad

- **Commit evaluado:** `0e2b03f0b1a84bf1d205f5caf6373be2c8d00ede` (árbol limpio).
- **Split checksum:** `f51dfe25cd14120e5624aefdc4b212b88b16610b71a11aaf143d3b8fa529c95d`
  — garantiza que las tres variantes entrenaron/validaron sobre el mismo split.
- **Snapshots de config y entorno:**
  `outputs/campaigns/fno_vs_unet_vs_attn/reproducibility/` (no versionado en git, generado
  por la campaña).
- **Fuentes de las cifras de este informe:**
  - Tabla de métricas (§5) y tabla estadística (§6): copiadas de
    `outputs/campaigns/fno_vs_unet_vs_attn/campaign_report.md`.
  - Valores por semilla: `docs/experiments.md` (secciones `baseline`, `unet_film`,
    `fno_axial_attn`).
  - Figuras (§5): generadas por `scripts/plot_campaign_comparison.py` a partir de los 9
    `metrics_history.json` de la campaña.
- **Regenerar las figuras:**
  ```bash
  python scripts/plot_campaign_comparison.py \
    --campaign-dir outputs/campaigns/fno_vs_unet_vs_attn
  ```
  Escribe en `outputs/campaigns/fno_vs_unet_vs_attn/comparison_figures/`; el subconjunto
  curado embebido en este informe vive además en
  `docs/figures/campana-fno-vs-unet-vs-attn/` (copia versionada, para que el documento
  renderice sin necesidad de `outputs/`, que no está bajo control de versiones).
- **Este informe no re-ejecuta la campaña ni entrena modelos**: consume exclusivamente
  artefactos ya producidos y verificados por `spec-004`.
