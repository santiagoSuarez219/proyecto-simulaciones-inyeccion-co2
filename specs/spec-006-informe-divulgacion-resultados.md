# spec-006 — Informe de divulgación de resultados (campaña fno_vs_unet_vs_attn) [DONE]

> **Autor:** rol `@architect`
> **Fecha:** 2026-07-21
> **Estado:** **`[DONE]`** — aprobado por el usuario 2026-07-21, implementado Fases 0-4,
> revisado por `@reviewer` el mismo día con veredicto **APROBADO** (223 tests en verde,
> incluidos los 19 nuevos; scope intacto; cifras del informe trazables y verificadas
> carácter a carácter contra `campaign_report.md` y los `metrics_history.json` reales).
> **Depende de:** `spec-004` (`[DONE]`, orquestación de campañas) — la campaña real
> `fno_vs_unet_vs_attn` ya corrió de punta a punta (3 variantes × 3 seeds), generó
> `outputs/campaigns/fno_vs_unet_vs_attn/campaign_report.md`, actualizó `docs/experiments.md`
> y dejó `reproducibility/` completo. **Este spec no re-ejecuta nada**: consume esos
> artefactos ya producidos.
> **Objetivo:** producir un **informe de divulgación curado, legible por humanos**
> (investigadores de geomecánica/CCS) en **Markdown**, que reporte y compare el rendimiento
> de las 3 variantes de arquitectura, con métricas, análisis estadístico, figuras
> comparativas y toda la información necesaria para interpretar y reproducir la comparación.
> Es un artefacto **narrativo y curado**, no una tabla autogenerada — complementa (no
> reemplaza) al `campaign_report.md` de máquina.

---

## 0. Contexto y delimitación

`spec-004` dejó dos artefactos de reporte, ambos **autogenerados** por
`src/fno_co2/experiments/campaign_report.py`:

- `outputs/campaigns/fno_vs_unet_vs_attn/campaign_report.md` — tabla resumen (mean±std de
  `val_sf_r2`/`val_vd_r2`/`val_sf_rmse`/`val_vd_rmse`), veredicto por criterio predefinido,
  tabla de comparación estadística vs. baseline (Wilcoxon, p-valores) y bloque de
  reproducibilidad (commit, split checksum).
- `docs/experiments.md` — valores crudos por seed.

Lo que **falta** —y es el alcance de este spec— es un **informe de divulgación**: un
documento **curado a mano**, con narrativa científica (contexto del problema, descripción de
las arquitecturas, discusión, limitaciones, conclusiones) que un investigador pueda leer de
principio a fin sin necesidad de interpretar tablas de máquina, y con **figuras comparativas
agregadas** de las que hoy no existe ninguna (las curvas actuales son por-seed y dispersas).

**Delimitación estricta (no duplicar `spec-004`):**

| Nivel | Artefacto | De quién |
|---|---|---|
| Reporte de máquina (tabla + estadística + repro) | `campaign_report.md` | `spec-004` F5 |
| Valores crudos por seed | `docs/experiments.md` | `spec-001` F5 / `spec-004` F5 |
| Curvas de entrenamiento **por-seed** | `training_curves.png` / `uncertainty_curves.png` | `spec-000` / `training/loop.py` |
| **Informe de divulgación curado (narrativa + figuras agregadas)** | **`docs/informe-...md`** | **este spec (F2)** |
| **Figuras comparativas agregadas (3 variantes superpuestas)** | **helper + script nuevos** | **este spec (F1)** |

**Principio rector:** el informe **no recalcula ciencia** ni re-entrena. Toma las cifras ya
producidas y verificadas por `spec-004` y las **presenta**. Toda cifra del informe debe ser
**trazable** a una fuente existente (`campaign_report.md`, `metrics_history.json`,
`campaign_manifest.json`), y la Fase 3 lo verifica mecánicamente para que el documento no
diverja de sus fuentes por un error de transcripción.

---

## 1. Diseño

### 1.1 Naturaleza del informe: narrativa manual + tablas/figuras derivadas (decisión clave)

El informe es un **híbrido explícito**, no un `.md` autogenerado ni uno enteramente a mano:

- **Manual (curado una vez, contenido narrativo):** resumen ejecutivo, contexto del problema
  (predicción espacio-temporal de SF y VD bajo inyección de CO₂), descripción de las 3
  arquitecturas comparadas, metodología en prosa, interpretación/discusión (qué gana, qué no,
  por qué), limitaciones, conclusiones. Es juicio experto: no se autogenera.
- **Derivado de artefactos existentes (reproducible, verificable):** la tabla de métricas
  consolidada y la tabla de comparación estadística se **copian de `campaign_report.md`**; el
  bloque de reproducibilidad, de `campaign_manifest.json`; las figuras, del script de la
  Fase 1 sobre los `metrics_history.json`. La Fase 3 verifica que estas cifras coinciden con
  su fuente.

> **Por qué híbrido y no 100% generado:** el valor del informe está en la **interpretación**
> (por qué `unet_film` es paridad y no mejora; por qué `fno_axial_attn` no cumple y por qué
> paró temprano), que ningún renderizador produce. Y **por qué no 100% manual:** las cifras
> deben poder re-verificarse contra las fuentes para que el documento sea reproducible y no
> se pudra silenciosamente si alguien re-corre la campaña.

Cada sección del informe declarará implícitamente su naturaleza por convención: las tablas
derivadas llevan una nota de procedencia (p. ej. "Fuente: `campaign_report.md`, campaña del
commit `0e2b03f`").

### 1.2 Ubicación y estructura del informe

**Ubicación propuesta:** `docs/informe-resultados-campana-fno-vs-unet-vs-attn.md`.

**Justificación:** `docs/` es el hogar de la documentación técnica curada del proyecto (p. ej.
`docs/arquitectura-y-correcciones-spec-000.md`, `docs/database_schema.md`), versionada **solo
en `development`** (§`main` limpio de `docs/`). Colocarlo ahí lo distingue nítidamente del
`campaign_report.md` **autogenerado y gitignored** bajo `outputs/`. El nombre es descriptivo y
apunta a la campaña concreta, dejando espacio a futuros informes de otras campañas.

**Estructura de secciones (orden propuesto):**

1. **Resumen ejecutivo** — 1 párrafo: qué se comparó, resultado en una frase (baseline sigue
   siendo la referencia; `unet_film` alcanza paridad; `fno_axial_attn` no cumple).
2. **Contexto del problema** — predicción espacio-temporal de SF y VD en reservorios
   depletados bajo inyección de CO₂ (CCS); qué mapea el modelo (propiedades estáticas +
   inyección TENE-1/TENE-2 → SF y VD por capa); para quién es útil.
3. **Arquitecturas comparadas** — descripción legible de las 3: `baseline` (FNO+FiLM),
   `unet_film` (U-Net con FiLM temporal), `fno_axial_attn` (FNO + atención espacial axial);
   qué hipótesis motivaba cada alternativa.
4. **Dataset y metodología** — split 90/10 estratificado (checksum del split como garantía de
   comparabilidad), normalización **global de train** sin fuga de datos, 3 seeds (42/43/44),
   criterio de éxito **predefinido** por variante (anti p-hacking), métricas evaluadas.
5. **Resultados: métricas de rendimiento** — tabla consolidada (mean±std de las 4 métricas
   por variante) + figuras comparativas agregadas (§1.3).
6. **Análisis estadístico vs. baseline** — tabla de efecto/test/p-valor (Wilcoxon/Mann-Whitney)
   con lectura en prosa (n=3 → p-valores no concluyentes por sí solos; se leen junto al
   tamaño de efecto y al no-solapamiento de rangos).
7. **Interpretación y discusión** — por qué `unet_film` es **equivalente, no mejora** (regla
   anti-solapamiento; mean por debajo del baseline); por qué `fno_axial_attn` **empeora** y su
   mayor varianza; observación del early stopping temprano (épocas 7/8/11) como línea de
   investigación (posible interacción LR/atención).
8. **Limitaciones** — incluida **explícitamente** la salvedad de incertidumbre (§1.4); n=3
   seeds; una sola campaña; datos sintéticos.
9. **Conclusiones** — recomendación práctica (baseline como referencia productiva; U-Net FiLM
   como alternativa viable de paridad; atención axial no justifica su costo).
10. **Reproducibilidad** — commit, split checksum, ruta a `reproducibility/`, cómo regenerar
    las figuras y verificar las cifras.

### 1.3 Figuras comparativas agregadas — generar nuevas (decisión con justificación)

**Decisión: generar un conjunto reducido de figuras comparativas nuevas**, no reutilizar las
por-seed existentes. Dos razones concretas:

1. Las figuras actuales son **por-seed y por-variante** (`<variante>/seed_<s>/training_curves.png`)
   — no permiten comparar las 3 arquitecturas de un vistazo, que es justo lo que un informe de
   divulgación necesita.
2. **Cobertura incompleta verificada:** los PNG por-seed **solo existen para `unet_film` y
   `fno_axial_attn`**; `baseline` **no tiene ninguno** (se importó vía `import_existing_run.py`,
   que copia `metrics_history.json`/`config.json`/`best.pt` pero **no** los PNG). Referenciar
   las figuras existentes daría un informe asimétrico. En cambio, **`metrics_history.json`
   existe para las 9 corridas**, así que una figura derivada de él es completa y consistente.

**Figuras propuestas (pocas y curadas):**

- **Curvas de convergencia comparativas:** `val_loss`, `val_sf_r2` y `val_vd_r2` vs. época,
  **las 3 variantes superpuestas**, con línea = media entre seeds y **banda ±std**.
- **Barras de métricas finales:** mean±std por variante de las 4 métricas (espejo visual de la
  tabla resumen), con marca del criterio predefinido.

**Sutileza técnica que el helper debe manejar (verificada en los datos):** las seeds paran por
early stopping en **distinto número de épocas** (baseline 17/24/19; unet_film 22/19/25;
fno_axial_attn **8/7/11**). Al agregar por índice de época, la banda ±std debe calcularse
**solo sobre las seeds que alcanzaron esa época** (n decreciente); más allá de la época 8,
`fno_axial_attn` queda con n=1 (sin banda). El helper no debe fallar ni inventar; el informe
debe **anotar** esta asimetría en el pie de figura.

**Dónde viven las figuras:** el script las escribe en
`outputs/campaigns/fno_vs_unet_vs_attn/comparison_figures/` (canónico, reproducible, bajo
`outputs/` gitignored). Como el informe vive en `docs/` (versionado) y debe embeber imágenes
por ruta relativa, el subconjunto curado se **copia** a `docs/figures/campana-fno-vs-unet-vs-attn/`
para que el `.md` renderice con o sin `outputs/`. Ambas rutas se documentan en el informe.

### 1.4 Salvedad de incertidumbre MC-Dropout (declarar, no corregir)

El informe **debe declarar** la deuda técnica `spec-004-debt-001` (backlog, prioridad BAJA) y
tratarla con cuidado. Estado **auditado y verificado** en los `metrics_history.json` reales:

- `val_sf_uncertainty_mean`/`val_vd_uncertainty_mean` solo se calculan en épocas múltiplo de
  `uncertainty_eval_interval=10`. Como las 9 corridas paran por early stopping (épocas 7–25),
  **la fila FINAL de las 9 tiene incertidumbre = 0.0 — es un artefacto**, no una medición.
- Valores **reales** en la última época donde SÍ se calculó (verificados):
  - `baseline`: sf_unc ≈ 0.30–0.32 (ep 10/20/10); vd_unc ≈ 0.26–0.27.
  - `unet_film`: sf_unc ≈ 0.34–0.35 (ep 20/10/20); vd_unc ≈ 0.26–0.27.
  - `fno_axial_attn`: **solo `seed_44`** tiene dato (sf_unc ≈ 0.33, ep 10); **`seed_42` y
    `seed_43` NUNCA calcularon** (pararon en época 8 y 7, antes de la época 10) → **n=1** para
    incertidumbre en esta variante.

**Reglas que el informe debe cumplir (decisión del usuario: la incertidumbre es SECUNDARIA):**

- (a) **No usar la fila final** de `metrics_history.json` para incertidumbre (es 0.0 artefacto).
- (b) Si se reportan cifras de incertidumbre, usar la **última época calibrada** (última fila
  con `val_sf_uncertainty_mean > 0`), no la fila `-1`.
- (c) **Declarar explícitamente la salvedad** y que `fno_axial_attn` tiene **n=1** para
  incertidumbre (comparación no representativa para esa variante).
- (d) Dejar claro que **las métricas de desempeño (R²/RMSE) NO están afectadas** por esta
  deuda y son plenamente fiables.

Las **figuras de incertidumbre no son prioritarias**; si se incluyen, deben respetar (a)–(c).
Este spec **no corrige** `spec-004-debt-001` (queda en el backlog).

### 1.5 Fuentes de datos (reutilizar, no recalcular)

| Dato del informe | Fuente | Naturaleza |
|---|---|---|
| Tabla de métricas (mean±std) | `campaign_report.md` (§Resumen) | copiar + nota de procedencia |
| Tabla estadística vs. baseline | `campaign_report.md` (§Comparación estadística) | copiar |
| Valores por seed (referencia) | `docs/experiments.md` | enlazar |
| Commit / split checksum | `campaign_manifest.json` + `reproducibility/` | copiar |
| Curvas de convergencia | `metrics_history.json` (9 corridas) → helper F1 | generado |
| Incertidumbre real (última época calibrada) | `metrics_history.json` (filtrar `>0`) | generado/verificado |

---

## Fase 0 — Precondiciones (bloqueantes)

**No entra a implementación hasta cumplirlas.**

1. **`spec-004` `[DONE]`** con la campaña `fno_vs_unet_vs_attn` corrida: verificar presencia de
   `campaign_report.md`, `docs/experiments.md` (3 secciones), `campaign_manifest.json`,
   `reproducibility/` y los 9 `metrics_history.json`. **(Verificado 2026-07-21.)**
2. **Salvedad de incertidumbre auditada** (§1.4): valores reales y n=1 de `fno_axial_attn`
   confirmados en los `metrics_history.json`. **(Verificado 2026-07-21.)**
3. **Rama nueva desde `development`:** `feature/informe-divulgacion-resultados`.
4. **Aprobación del usuario de este spec** (naturaleza híbrida, ubicación, alcance de figuras).

**Criterios de aceptación (Fase 0):**
- [x] Los 6 artefactos fuente existen y son legibles.
- [x] Rama `feature/informe-divulgacion-resultados` creada desde `development`.
- [x] Spec aprobado por el usuario y promovido a `[IN PROGRESS]`.

---

## Fase 1 — Figuras comparativas agregadas (helper + script CLI)

**Dónde:** `src/fno_co2/visualization/plots.py` (extender con una función pura nueva, sin tocar
`save_history_plots`/`save_epoch_visuals`), `scripts/plot_campaign_comparison.py` (nuevo, CLI
delgado).

1. Función pura nueva en `plots.py` (p. ej. `save_campaign_comparison_plots`) que recibe las
   rutas de los `metrics_history.json` por variante/seed y un directorio de salida, y produce:
   curvas comparativas (`val_loss`, `val_sf_r2`, `val_vd_r2` vs. época, 3 variantes, media +
   banda ±std) y barras de métricas finales por variante.
2. **Manejo de seeds de distinta longitud (§1.3):** la banda ±std se calcula solo sobre las
   seeds presentes en cada época (n decreciente); n=1 → sin banda, sin fallo. Las métricas
   "finales" por seed se toman de la **época del `best.pt`** (coherente con la agregación de
   `spec-004`), no de la última fila.
3. **Incertidumbre (si se grafica):** usar la última fila con `val_sf_uncertainty_mean > 0`
   por seed; excluir explícitamente las filas 0.0 (§1.4-a/b). `fno_axial_attn` se marca n=1.
4. CLI `scripts/plot_campaign_comparison.py`: descubre las corridas bajo
   `outputs/campaigns/<name>/`, invoca el helper, escribe en
   `outputs/campaigns/<name>/comparison_figures/`. Logging vía `fno_co2.utils.get_logger`
   (no `print`). Sin dependencias nuevas (usa `matplotlib`/`numpy` ya presentes).

**Criterios de aceptación (Fase 1):**
- [x] El helper genera las figuras para las 9 corridas sin error, con seeds de distinta longitud.
- [x] La banda ±std usa n decreciente; el caso n=1 (`fno_axial_attn` tras la época 8) no falla.
- [x] Ninguna figura de incertidumbre usa la fila final 0.0. **(N/A: por decisión del usuario
      este spec no genera figura de incertidumbre — solo curvas de convergencia + barras de
      métricas finales.)**
- [x] El script escribe en `comparison_figures/` y loguea con `get_logger`, sin dependencias nuevas.
- [x] `plots.py` existente (`save_history_plots`/`save_epoch_visuals`) queda **sin modificar**.

**Nota de implementación (no prevista en el diseño original):** las curvas de convergencia
reales muestran un pico de inestabilidad de una sola época en `fno_axial_attn/seed_44`
(época 3: `val_sf_r2=-100.7`, se recupera en la época 4) que, con límites de eje por
min/max, aplastaba la escala completa y hacía ilegible la comparación entre variantes. Se
añadió `_percentile_ylim` (rango de eje robusto por percentil 2-98 sobre los valores
crudos, no la media) para que el pico siga siendo visible sin dominar el marco — no se
recorta ningún dato, solo el viewport. Genérico, no hardcodea la variante/seed/época
concreta.

---

## Fase 2 — Redacción del informe curado

**Dónde:** `docs/informe-resultados-campana-fno-vs-unet-vs-attn.md` (nuevo),
`docs/figures/campana-fno-vs-unet-vs-attn/` (subconjunto curado de figuras copiado desde
`outputs/.../comparison_figures/`).

1. Redactar las 10 secciones de §1.2 en **español**, tono divulgativo para investigadores de
   geomecánica/CCS. Contenido narrativo a mano; tablas y figuras derivadas de las fuentes (§1.5).
2. Tablas derivadas **copiadas** de `campaign_report.md` con **nota de procedencia** (campaña,
   commit `0e2b03f`). No re-derivar a mano las cifras.
3. Embeber el subconjunto curado de figuras por ruta relativa (`docs/figures/...`).
4. Sección de **limitaciones** con la salvedad de incertidumbre (§1.4) redactada explícitamente:
   fila final = artefacto; cifras reales de la última época calibrada; `fno_axial_attn` n=1;
   R²/RMSE no afectadas.
5. Sección de **reproducibilidad**: commit, split checksum, ruta a `reproducibility/`, y el
   comando exacto para regenerar las figuras (`scripts/plot_campaign_comparison.py`).

**Criterios de aceptación (Fase 2):**
- [x] El informe existe en `docs/`, en español, con las 10 secciones y narrativa curada.
- [x] Las tablas coinciden con `campaign_report.md` y llevan nota de procedencia.
- [x] Las figuras de la Fase 1 se embeben y renderizan por ruta relativa.
- [x] La salvedad de incertidumbre está declarada explícitamente en Limitaciones, con (a)–(d).
- [x] Ninguna cifra de incertidumbre proviene de la fila final (0.0) — el informe no cita
      cifras de incertidumbre en absoluto (decisión del usuario: secundaria), solo declara
      la salvedad y los valores auditados de la última época calibrada en prosa.

---

## Fase 3 — Verificación (tests + coherencia numérica + guarda de incertidumbre)

**Dónde:** `tests/unit/test_campaign_comparison_plots.py` (nuevo),
`tests/unit/test_informe_coherencia.py` (nuevo). Framework `pytest`; tests que necesiten leer
las figuras reales o los `metrics_history.json` completos se marcan `@pytest.mark.slow`.

1. **Test del helper de figuras (unit, rápido):** con `metrics_history.json` sintéticos de 3
   variantes × 3 seeds de **longitudes distintas** (incluido un caso que emula
   `fno_axial_attn`: 2 seeds sin ninguna fila de incertidumbre y todas parando antes de la
   época 10), verificar que las figuras se generan, la banda maneja n decreciente y n=1, y no
   se usa la fila final para incertidumbre.
2. **Test de coherencia numérica (guarda anti-divergencia):** recomputar mean±std de las 4
   métricas desde los `metrics_history.json` (época del `best.pt`) y comparar con las cifras
   de `campaign_report.md` (la fuente que el informe copia) — si la campaña se re-corre y los
   números cambian, este test avisa de que el informe curado quedó desactualizado.
3. **Guarda de incertidumbre:** verificar que el selector de "última época calibrada" devuelve
   una fila con `val_sf_uncertainty_mean > 0` (no la fila `-1`) y que marca `fno_axial_attn`
   como n=1.
4. **No-regresión:** `pytest tests/ -m "not slow"` completo sigue verde (este spec no toca
   modelo, ETL ni training; solo añade un helper de plotting y tests).

**Criterios de aceptación (Fase 3):**
- [x] El test del helper cubre seeds de distinta longitud, n=1 y exclusión de la fila final
      (`tests/unit/test_campaign_comparison_plots.py`, 9 tests).
- [x] El test de coherencia detecta divergencia entre el informe y `campaign_report.md`
      (`tests/unit/test_informe_coherencia.py`: mecanismo validado con datos sintéticos,
      portable/CI-safe; más una verificación contra la campaña real que hace `pytest.skip`
      si `outputs/campaigns/...` no está presente localmente — no está versionado en git).
- [x] La guarda de incertidumbre confirma que no se cita la fila final 0.0
      (`test_last_calibrated_uncertainty_row_*` + `test_informe_uncertainty_caveat_never_cites_final_row_as_measurement`,
      esta última confirma también contra datos reales que `fno_axial_attn` tiene n=1 y las
      otras dos variantes n=3 semillas calibradas).
- [x] `pytest tests/ -m "not slow"` completo pasa (223 passed; 213 previos + 10 nuevos).

**Desviación respecto al diseño original:** las pruebas contra la campaña real usan
`pytest.skip` si `outputs/campaigns/fno_vs_unet_vs_attn/` no existe, en vez de asumir su
presencia — `outputs/` está gitignored (CLAUDE.md), así que ni `metrics_history.json` ni
`campaign_report.md` están versionados. La parte portable de la guarda (parseo de tablas +
recomputación mean/std) se valida con datos sintéticos siguiendo el patrón de
`test_campaign_report.py`.

---

## Fase 4 — Cierre y referencias cruzadas (ligera)

**Dónde:** `specs/spec-006-informe-divulgacion-resultados.md` (este archivo), `README.md`
(opcional), `specs/backlog.md` (referencia, no cierre).

1. Marcar este spec `[DONE]` tras verificación.
2. Enlazar el informe desde `README.md` como entregable de divulgación (opcional, a criterio
   del usuario).
3. **No** cerrar `spec-004-debt-001`: el informe **declara** la salvedad de incertidumbre pero
   **no corrige** la deuda; añadir una nota en el backlog de que el informe la documenta.

**Criterios de aceptación (Fase 4):**
- [x] Spec en `[DONE]`; informe enlazado en `README.md` §Documentación (aprobado por el
      usuario 2026-07-21). `@reviewer` emitió **APROBADO** el 2026-07-21 (sin hallazgos
      bloqueantes; dos sugerencias menores no bloqueantes, ver informe de revisión).
- [x] `spec-004-debt-001` sigue abierta, con nota de que el informe la documenta
      (`specs/backlog.md`).

---

## 2. Archivos impactados (resumen)

| Archivo / carpeta | Fase | Naturaleza |
|---|---|---|
| `docs/informe-resultados-campana-fno-vs-unet-vs-attn.md` | 2 | **Nuevo** — informe curado |
| `docs/figures/campana-fno-vs-unet-vs-attn/` | 2 | **Nuevo** — subconjunto curado de figuras |
| `src/fno_co2/visualization/plots.py` | 1 | Extender — helper de figuras comparativas (sin tocar lo existente) |
| `scripts/plot_campaign_comparison.py` | 1 | **Nuevo** — CLI que genera las figuras agregadas |
| `outputs/campaigns/fno_vs_unet_vs_attn/comparison_figures/` | 1 | Runtime — figuras generadas (gitignored) |
| `tests/unit/test_campaign_comparison_plots.py` | 3 | **Nuevo** |
| `tests/unit/test_informe_coherencia.py` | 3 | **Nuevo** |
| `README.md` | 4 | Enlace opcional al informe |
| `specs/backlog.md` | 4 | Nota (la deuda de incertidumbre sigue abierta) |
| `campaign_report.py`, `campaign_report.md`, `docs/experiments.md`, `train.py`, `models/*`, `training/*` | — | **NO se modifican** (solo se leen como fuente) |
| Git: rama `feature/informe-divulgacion-resultados` | 0 | Desde `development` |

---

## 3. Reproducibilidad

- El informe es **reproducible por construcción**: sus tablas provienen de `campaign_report.md`
  (con commit y split checksum) y sus figuras del script de la Fase 1 sobre los
  `metrics_history.json` versionados por la campaña. Cualquiera puede regenerar las figuras con
  un comando y re-verificar las cifras con el test de coherencia (Fase 3).
- **Trazabilidad de cada número:** nota de procedencia en cada tabla (campaña, commit `0e2b03f`,
  split checksum `f51dfe25…529c95d`) — el informe nunca introduce cifras sin fuente.
- **No re-ejecución:** este spec **no** entrena, no toca GPU, no regenera la campaña. Solo lee
  artefactos ya producidos y verificados por `spec-004`.
- `docs/` y `specs/` viven **solo en `development`** (§`main` limpio); el informe y este spec no
  deben llegar a `main` (usar `scripts/promote-to-main.sh`).

---

## 4. Riesgos y notas

- **Divergencia informe ↔ fuentes:** si la campaña se re-corre con otros números, el informe
  curado quedaría desactualizado. Mitigación: el test de coherencia (Fase 3.2) lo detecta; la
  nota de procedencia fija el commit de la campaña reportada.
- **Incertidumbre engañosa:** el riesgo principal de contenido es citar la fila final 0.0 como
  si fuera una medición. Mitigación: reglas (a)–(d) de §1.4 + guarda de test (Fase 3.3).
- **Asimetría de figuras por early stopping:** `fno_axial_attn` para muy temprano (7/8/11) →
  bandas con n=1 más allá de la época 8. No es un bug; se **anota** en el pie de figura y en la
  discusión (posible interacción LR/atención, ya registrada en `spec-003`).
- **Cobertura de figuras por-seed:** `baseline` no tiene PNG por-seed (importado sin copiar
  figuras). Por eso se generan figuras nuevas desde `metrics_history.json` (completo para las 9
  corridas), no se referencian las existentes.
- **Sin dependencias nuevas:** todo con `matplotlib`/`numpy` ya en el entorno; no se instala nada.
- **Alcance:** un informe para **una** campaña. Si a futuro se quiere un informe por campaña
  arbitraria, el helper de figuras (Fase 1) ya es reutilizable; solo la narrativa es específica.

---

## 5. Criterios de aceptación (globales)

- [x] Existe `docs/informe-resultados-campana-fno-vs-unet-vs-attn.md`, curado, en español, con
      las 10 secciones (resumen, contexto, arquitecturas, metodología, resultados, estadística,
      discusión, limitaciones, conclusiones, reproducibilidad).
- [x] El informe reporta y compara las 3 variantes con métricas (mean±std), análisis estadístico
      vs. baseline y figuras comparativas agregadas de convergencia.
- [x] Las tablas coinciden **exactamente** con `campaign_report.md` (verificado por test) y
      llevan nota de procedencia.
- [x] Las figuras agregadas se generan con un script reproducible desde los `metrics_history.json`
      y se embeben en el informe; el helper vive en `visualization/plots.py` sin romper lo existente.
- [x] La salvedad de incertidumbre está declarada: fila final = artefacto (no se usa),
      cifras reales de la última época calibrada mencionadas en prosa, `fno_axial_attn` n=1,
      R²/RMSE no afectadas.
- [x] Ninguna cifra de incertidumbre del informe proviene de la fila final 0.0 (el informe no
      cita cifras de incertidumbre en tablas/figuras — decisión del usuario, secundaria).
- [x] `campaign_report.py`, `train.py`, `models/*` y `training/*` quedan **sin modificar**
      (verificado: `git diff --stat` solo toca `plots.py`, script nuevo, tests, docs, spec).
- [x] `pytest tests/ -m "not slow"` completo pasa (223 passed); no hay tests `slow` nuevos en
      este spec.
- [x] El spec **no** re-ejecuta la campaña ni toca GPU.
