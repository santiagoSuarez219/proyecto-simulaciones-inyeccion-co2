# spec-002 — Variante de arquitectura U-Net con condicionamiento FiLM temporal [DONE]

> **Autor:** rol `@architect`
> **Fecha:** 2026-07-02 · **Actualizado:** 2026-07-19 (Fase 5 cerrada con corrida multi-seed
> real vía la campaña de `spec-004`)
> **Estado:** [DONE] — Fases 1-5 completas, incluida la corrida multi-seed real (Fase 5.3,
> antes pendiente): 3 seeds, datos completos, GPU real. `val_sf_r2 = 0.9920 ± 0.0002`,
> `val_vd_r2 = 0.9650 ± 0.0010` — dentro del criterio de no-degradación (`≥0.974`/`≥0.9430`).
> Por la regla anti-solapamiento del propio criterio (§Fase 5), el veredicto es
> **"equivalente" a baseline, no "mejora"** (mean de `val_sf_r2` levemente por debajo del de
> baseline, rangos sin solapar). Detalle en `specs/backlog.md` (`spec-002-debt-002`) y
> `docs/experiments.md`.
> **Depende de:** `spec-001` (framework de experimentación y comparación de
> arquitecturas). Este spec es la **primera variante estructural** que consume ese
> framework (spec-001 §Fase 3): vive en `models/variants/`, se registra en
> `build_model(cfg)` y se compara contra la línea base con el mismo proceso multi-seed.
> **Objetivo:** implementar una arquitectura **U-Net** (encoder-decoder con *skip
> connections*) que **reemplace los `FiLMSpectralBlock` (FFT)** del baseline, **conservando
> el condicionamiento temporal** (embedding de timestep + MLP de inyección/profundidad
> inyectados vía FiLM en el decoder), como variante seleccionable por `cfg.model_variant`.
> No modifica la línea base (`PhysicalFNOArchitecture`) in-place.

---

## 0. Contexto y motivación

- **Alineación con el paper.** El paper en redacción se titula literalmente *"A U-Net
  Approach for Safety Factor Prediction in Depleted Reservoirs Using Synthetic Data"*,
  pero la implementación actual (`PhysicalFNOArchitecture`) es un **FNO** con bloques
  espectrales FiLM. Esta variante materializa la arquitectura que el título anuncia y
  permite reportar la comparación FNO vs. U-Net con datos propios — que es exactamente lo
  que el framework de `spec-001` está diseñado para producir con rigor (mismos datos,
  mismo split, multi-seed, mean±std, tamaño de efecto).
- **Qué NO es esta variante.** No es una U-Net "de libro" estática por timestep: se
  conserva el mecanismo de condicionamiento actual (embedding de paso temporal +
  `cond_mlp` sobre `[inj_t1, inj_t2, depth]`), porque ese condicionamiento **no es parte
  del debate "FNO vs U-Net"** sino de la física del problema (la evolución de SF/VD depende
  de la serie de inyección y la profundidad de la capa). Se cambia **una cosa a la vez**
  (el núcleo espectral → un backbone U-Net convolucional multi-escala), respetando el
  principio rector de `spec-001` §0.
- **Qué se conserva del baseline (contrato duro).** La variante debe ser un *drop-in*: se
  instancia desde `build_model(cfg)`, entrena con el mismo `training/loop.py`, la misma
  loss, el mismo `build_param_groups`, y es compatible con MC Dropout
  (`inference/uncertainty.py`). Ver §2 (Contrato de interfaz).

---

## 1. Diseño de la arquitectura `UNetFiLMTemporal`

### 1.1 Idea general

Igual que el baseline, se **codifica una sola vez** el estado estático (propiedades +
profundidad) y se **decodifica por cada uno de los `T` timesteps** aplicando FiLM. La
diferencia es el backbone entre encoder y decoder:

| | Línea base (`PhysicalFNOArchitecture`) | Variante (`UNetFiLMTemporal`) |
|---|---|---|
| Núcleo | 4× `FiLMSpectralBlock` (FFT2 → modos truncados → iFFT2) a resolución fija | Camino contractivo/expansivo multi-escala con *skip connections* |
| Receptivo global | Vía transformada de Fourier (global por construcción) | Vía *downsampling* (crece con la profundidad de la U-Net) |
| Condicionamiento | FiLM por timestep dentro de cada bloque espectral | FiLM por timestep en cada bloque del decoder |
| Skips | No | Sí (features estáticos del encoder → decoder) |

### 1.2 Flujo (`forward(x, d, inj)`)

```
Entrada (idéntica al baseline):
  x   : (B, 4, H, W)   — propiedades estáticas (AFI/COH/PERM/PORO)
  d   : (B, 1)          — profundidad normalizada de la capa
  inj : (B, T, 2)       — series de inyección TENE-1 y TENE-2

1. depth_map = d.view(B,1,1,1).expand(B,1,H,W)
   z0 = concat([x, depth_map], dim=1)            → (B, 5, H, W)   (in_c=5, igual que baseline)

2. ENCODER (estático, una sola vez):
   stem: Conv2d(5 → C0)
   Down_1 → skip s1 (C0,  H,   W  ), baja a (C1, H/2,  W/2)
   Down_2 → skip s2 (C1,  H/2, W/2), baja a (C2, H/4,  W/4)
   Down_3 → skip s3 (C2,  H/4, W/4), baja a (C3, H/8,  W/8)  = bottleneck
   (profundidad configurable, ver 1.4; se guardan los tamaños exactos de cada nivel)

3. CONDICIONAMIENTO TEMPORAL (reutiliza EXACTAMENTE la lógica del baseline):
   - padding/truncado de inj a T pasos (misma guarda que fno.py:43-49)
   - t_emb   = Embedding(T, cond_dim)[arange(T)]           → (B, T, cond_dim)
   - depth_seq = d expandida a T
   - cond_seq = t_emb + cond_mlp(concat([inj, depth_seq])) → (B, T, cond_dim)

4. Expandir bottleneck sobre T:
   z_bt   = bottleneck expandido a (B*T, C3, H/8, W/8)
   cond_bt = cond_seq.reshape(B*T, cond_dim)
   (skips s1..s3 se expanden sobre T al concatenarlos en cada Up)

5. DECODER (por timestep, con FiLM):
   Up_3: upsample(z_bt) al tamaño de s3 → concat(s3⊗T) → ResBlock → FiLM(cond_bt)
   Up_2: upsample        al tamaño de s2 → concat(s2⊗T) → ResBlock → FiLM(cond_bt)
   Up_1: upsample        al tamaño de s1 → concat(s1⊗T) → ResBlock → FiLM(cond_bt)
   head: Conv2d(C0 → 2)

6. Salida:
   (B*T, 2, H, W).view(B, T, 2, H, W)             — SF y VD para todos los T pasos
```

### 1.3 Bloques nuevos (en `models/variants/unet_film.py` o helpers propios)

- **`FiLMModulation(cond_dim, c)`** — modulación afín `y·(1+γ(cond)) + β(cond)`, con
  submódulos nombrados `.gamma` y `.beta` (`nn.Linear`) **para que `build_param_groups`
  los detecte como `no_decay`** (misma convención que `FiLMSpectralBlock`; ver
  `training/optim.py`). Este es el único mecanismo por el que la inyección/profundidad/
  timestep entran al decoder.
- **`DownBlock(c_in, c_out, dropout_p, use_group_norm)`** — `ResBlock` (reutilizado de
  `models/blocks.py`, ya trae `Dropout2d` + `GroupNorm` opcional) seguido de reducción de
  resolución (Conv2d stride-2 o MaxPool2d; recomendado stride-2 con `padding_mode="replicate"`
  por consistencia con el resto del modelo).
- **`UpBlock(c_in, c_skip, c_out, cond_dim, dropout_p, use_group_norm)`** — *upsample* +
  concat del skip + `ResBlock` + `FiLMModulation`.
- **Reutilización:** `ResBlock` y el patrón `padding_mode="replicate"` se reutilizan del
  baseline; **no** se copia `FiLMSpectralBlock` (esta variante no usa FFT).

### 1.4 Hiperparámetros de la variante

- **Profundidad de la U-Net:** recomendado **3 niveles** (`unet_depth=3`): con la grilla
  100×100 el bottleneck queda en ~13×13, receptivo suficientemente global sin explotar
  memoria. Configurable vía `Config.unet_depth` (ver Fase 3), documentado como **solo
  aplica a la variante `unet_film`**.
- **Canales:** parten de `cfg.hidden_dim` (C0) y duplican por nivel
  (`C0, 2·C0, 4·C0, …`). Reutiliza `hidden_dim`, no introduce campo nuevo para esto.
- **Dropout / GroupNorm:** vía los `ResBlock` internos, gobernados por `cfg.dropout_p` y
  `cfg.use_group_norm` (mismos campos que el baseline; MC Dropout funciona sin cambios).

### 1.5 Reconciliación de tamaños en grillas no potencia-de-2 (⚠️ punto crítico)

100 no es divisible limpiamente por 8 (100 → 50 → 25 → 13 con redondeo). El *upsample* del
decoder **debe** restaurar el tamaño exacto del skip correspondiente, no asumir ×2. Diseño
obligatorio:

- El encoder **guarda el tamaño espacial `(h_i, w_i)` de cada nivel** al descender.
- Cada `UpBlock` hace *upsample* **al tamaño del skip** (`F.interpolate(..., size=skip
  spatial)` o `ConvTranspose2d` + pad/crop), nunca por factor fijo ×2.
- Los tests (Fase 4) deben cubrir **al menos dos grillas**: 100×100 (real) y una pequeña
  no-cuadrada/impar (p. ej. 30×26) para forzar el camino de reconciliación.

---

## 2. Contrato de interfaz (debe cumplirse para ser *drop-in*)

Cualquier discrepancia aquí rompe el reuso de `training/loop.py` y `inference/`:

1. **Firma:** `forward(x, d, inj) -> Tensor` con salida exacta `(B, T, 2, H, W)`
   (verificado contra el shape que consume la loss en `training/loop.py:89-90`
   —`pred = model(x, d, inj)` → `compute_loss_terms(pred, y, cfg)`— y la
   incertidumbre en `inference/uncertainty.py`).
2. **Entrada:** `x=(B,4,H,W)`, `d=(B,1)`, `inj=(B,T,2)`; concatena `depth_map` internamente
   (in_c=5). Misma guarda de padding/truncado de `inj` a `T` que el baseline.
3. **`time_steps`:** atributo `self.time_steps` presente (lo usan resume/firma y el
   expandido temporal).
4. **MC Dropout:** contiene capas `nn.Dropout2d(cfg.dropout_p)` (vía `ResBlock`) para que
   `predict_with_uncertainty` produzca incertidumbre real (>0) al forzar modo train.
5. **Param groups:** submódulos FiLM nombrados `.gamma`/`.beta` y embeddings como
   `nn.Embedding` → `build_param_groups` los clasifica en `no_decay` sin cambios.
6. **Constructor:** instanciable desde `build_model(cfg)` a través de la función
   `build(cfg) -> nn.Module` del módulo `variants/unet_film.py` (mecanismo de import
   dinámico del registry, ver §Fase 3), leyendo campos de `Config`. `in_c=5` y
   `cond_dim=128` se fijan dentro de `build()`, **igual que `_build_baseline` en
   `registry.py`** (no son campos de `Config`; este es el patrón vigente, no una excepción).

---

## Fase 0 — Precondiciones

1. **✅ CUMPLIDA — `spec-001` Fase 1:** `--model-variant` en CLI ([`scripts/train.py:46`])
   y `Config.model_variant` (default `"fno_baseline"`, `config.py:20`);
   `build_run_signature` ya expone `cfg.model_variant` (bajo la clave `"model_name"` de la
   firma, `checkpoint.py:34` — el valor es `cfg.model_variant`, no un literal hardcodeado).
2. **✅ CUMPLIDA — `spec-001` Fase 3:** existe `src/fno_co2/models/registry.py::build_model`
   y el directorio `src/fno_co2/models/variants/` (con `__init__.py`); `training/loop.py`
   usa `build_model(cfg)` en vez de instanciar `PhysicalFNOArchitecture` directamente
   (hoy en `loop.py:239`). ⚠️ **Ojo con el mecanismo real:** `build_model` **no** mantiene
   un diccionario de variantes; para cualquier `model_variant != "fno_baseline"` hace
   **import dinámico** de `fno_co2.models.variants.<variant>` y llama a su función
   `build(cfg) -> nn.Module`. Esto redefine la Fase 3 (ver allí): **no se toca `registry.py`**.
3. **⏳ PENDIENTE — Rama `exp/unet-film`** creada **desde `development`** (convención
   `CLAUDE.md` §Git; `spec-001` §Fase 3.4). **No** trabajar sobre `main`/`development`.
   Único paso de la Fase 0 que falta antes de escribir código.

**Verificación:** `build_model(Config(model_variant="fno_baseline"))` devuelve
`PhysicalFNOArchitecture` (test de `spec-001`, ya en verde); `--model-variant` aceptado por
`scripts/train.py` sin error. Ambas ya se cumplen hoy en `development`.

---

## Fase 1 — Bloques U-Net y modulación FiLM

**Dónde:** `src/fno_co2/models/variants/unet_film.py` (nuevo); posible reuso desde
`src/fno_co2/models/blocks.py`.

1. Implementar `FiLMModulation`, `DownBlock`, `UpBlock` (ver §1.3), reutilizando `ResBlock`
   de `blocks.py`. **No** modificar `blocks.py::FiLMSpectralBlock` ni `fno.py`.
2. Respetar `padding_mode="replicate"`, `dropout_p`, `use_group_norm` como en el baseline.
3. Nombrar los submódulos de FiLM `gamma`/`beta` (contrato §2.5).

**Verificación:** test unitario de shapes por bloque (un `DownBlock` reduce H,W a la mitad
—o al tamaño esperado— y un `UpBlock` restaura al tamaño del skip provisto), incluido en
Fase 4.

---

## Fase 2 — Clase `UNetFiLMTemporal`

**Dónde:** `src/fno_co2/models/variants/unet_film.py`.

1. Implementar `UNetFiLMTemporal(nn.Module)` con el `forward` de §1.2 y el contrato de §2.
2. Reutilizar **verbatim** el bloque de condicionamiento temporal del baseline (padding de
   `inj`, `t_embed`, `cond_mlp` sobre `[inj, depth_seq]`) — mismo `cond_dim=128`.
3. Implementar la reconciliación de tamaños de §1.5 (guardar sizes en el encoder, *upsample*
   al size del skip en el decoder).
4. Exponer `self.time_steps` y firmar `forward(x, d, inj)`.

**Verificación (parte en Fase 4):** `forward` con tensores dummy `(B=2, 4, 100, 100)`,
`d=(2,1)`, `inj=(2,T,2)` devuelve `(2, T, 2, 100, 100)`; y con grilla `(2,4,30,26)` devuelve
`(2, T, 2, 30, 26)` sin error de shape.

---

## Fase 3 — Integración: Config, función `build()` de la variante y config de experimento

**Dónde:** `src/fno_co2/config.py`, `src/fno_co2/models/variants/unet_film.py` (la misma
función `build`), `configs/experiments/unet_film.yaml`.
**NO se toca `registry.py`** (ver abajo).

1. `Config`: añadir `unet_depth: int = 3` (documentado: **solo afecta a `unet_film`**; el
   baseline lo ignora). No añadir campo de canales (se deriva de `hidden_dim`, §1.4).
   Al ser un campo nuevo del dataclass, el loader `load_config_from_yaml` lo acepta
   automáticamente y `baseline.yaml` (que no lo declara) sigue cargando con el default.
2. **Auto-registro por import dinámico (NO editar `registry.py`):** el `build_model` real
   (`registry.py:23`) despacha cualquier `model_variant != "fno_baseline"` importando
   `fno_co2.models.variants.<model_variant>` y llamando a su función
   **`build(cfg) -> nn.Module`**. Por tanto, para registrar `"unet_film"` **basta con que
   `variants/unet_film.py` exponga una función `build(cfg)`** que construya
   `UNetFiLMTemporal(...)` leyendo `time_steps`, `hidden_dim`, `dropout_p`,
   `use_group_norm`, `unet_depth` de `cfg` (y fijando `in_c=5`, `cond_dim=128` como
   `_build_baseline`). La entrada `"fno_baseline"` queda intacta; una variante cuyo módulo
   no exista o no defina `build` ya lanza `ValueError` explícito (código existente).
3. `configs/experiments/unet_film.yaml`: config **completa y autocontenida** (misma
   estructura y claves que `configs/experiments/baseline.yaml`), idéntica al baseline salvo
   `model_variant: unet_film`, `experiment_name: unet_film` y, si aplica, `unet_depth`.
   Es el artefacto reproducible del experimento.

**Verificación:** `build_model(Config(model_variant="unet_film"))` devuelve
`UNetFiLMTemporal` (vía el import dinámico, sin tocar `registry.py`); cargar
`configs/experiments/unet_film.yaml` con `load_config_from_yaml` produce un `Config` con
`model_variant="unet_film"` (round-trip del loader de `spec-001` Fase 2).

---

## Fase 4 — Tests unitarios

**Dónde:** `tests/unit/test_unet_film.py` (nuevo).

Casos mínimos (todos rápidos, sin datos reales, sin `@pytest.mark.slow`):

1. **Shape feliz:** `forward` → `(B, T, 2, H, W)` con grilla 100×100.
2. **Reconciliación de tamaños:** misma prueba con grilla no potencia-de-2 (p. ej. 30×26)
   → shape correcto, sin error de concatenación de skips.
3. **Backward:** `loss.backward()` sobre salida dummy produce gradientes finitos (sin
   `NaN`/`Inf`) en todos los parámetros que requieren grad.
4. **MC Dropout real:** con `dropout_p>0` y `model.train()`, dos `forward` con la misma
   entrada difieren (dropout activo) → garantiza incertidumbre >0 en inferencia.
5. **Param groups:** `build_param_groups(model, wd)` coloca los `gamma`/`beta` de FiLM y los
   embeddings en `no_decay` (ninguno queda con weight decay por error).
6. **Registry:** `build_model(cfg)` con `model_variant="unet_film"` devuelve la clase
   correcta; string desconocido lanza `ValueError`.

**Verificación:** `pytest tests/unit/test_unet_film.py -v` pasa; `pytest tests/ -m "not
slow"` completo sigue verde (no se rompe `spec-000`/`spec-001`).

---

## Fase 5 — Humo de convergencia y experimento comparativo

> Esta fase **usa el framework de `spec-001`**; no reimplementa nada de entrenamiento.

1. **Humo (sin GPU obligatoria):** correr `scripts/train.py --model-variant unet_film
   --overfit-sample-idx 0` unas pocas épocas; verificar que la loss **baja** (la variante
   puede sobreajustar 1 muestra). Esto detecta errores de arquitectura antes de gastar GPU.
2. **Criterio de éxito predefinido (⚠️ fijar ANTES de correr, `spec-001` Fase 6):**
   escribir la fila de la variante en `docs/experiments.md` con la hipótesis y el umbral
   *antes* de ver resultados. Referencia de la línea base ya congelada (3 seeds 42/43/44,
   `docs/experiments.md`): **`val_sf_r2 = 0.9937 ± 0.0001`**, **`val_vd_r2 = 0.9626 ± 0.0028`**,
   `val_sf_rmse = 0.0091 ± 0.0001`, `val_vd_rmse = 0.0201 ± 0.0007`. Criterio fijado:
   > *"`unet_film` iguala o supera a la línea base sin degradar la métrica principal:*
   > *`val_sf_r2` mean **≥ 0.974** (no más de 2% por debajo del baseline 0.9937) **y***
   > *`val_vd_r2` mean **≥ 0.9430** (no más de 2% por debajo del baseline 0.9626), ambos con*
   > *≥3 seeds. Se declara 'mejora' solo si el rango mean±std de `val_sf_r2` de `unet_film`*
   > *no se solapa con el del baseline y su mean lo supera (regla anti-solapamiento,*
   > *`spec-001` Fase 6); en caso contrario se reporta 'equivalente' o 'peor', nunca por un*
   > *único seed."*
3. **Corrida real (requiere GPU + datos post-C1, y la línea base ya congelada — `spec-001`
   Fase 0):** `scripts/run_experiment.py --config configs/experiments/unet_film.yaml
   --n-seeds 3` (mínimo 3 seeds). **⚠️ Confirmación explícita del usuario** antes de lanzar
   entrenamiento (§Despliegue de `CLAUDE.md`).
4. **Agregación y reporte:** `scripts/aggregate_experiments.py` (spec-001 Fase 5) añade la
   fila `unet_film` a `docs/experiments.md` con mean±std, tamaño de efecto y valores por
   seed vs. baseline. Ninguna conclusión de "mejor/peor" con <3 seeds ni con rangos
   mean±std solapados (spec-001 Fase 6).

**Verificación:** humo de overfit baja la loss; `docs/experiments.md` tiene la fila
`unet_film` con criterio predefinido y resultados multi-seed vs. baseline.

> ✅ **Cumplida (2026-07-19), vía la campaña `fno_vs_unet_vs_attn` de `spec-004`** (no con
> `run_experiment.py` directo como decía el paso 3 original — la campaña internamente
> reutiliza ese mismo camino de código, sin reimplementarlo). 3 seeds (42/43/44), 19-25
> épocas cada una (early stopping), `use_amp=true`: `val_sf_r2 = 0.9920 ± 0.0002`,
> `val_vd_r2 = 0.9650 ± 0.0010`, `val_sf_rmse = 0.0103 ± 0.0001`, `val_vd_rmse = 0.0195 ±
> 0.0003`. Cumple el umbral de no-degradación (`≥0.974`/`≥0.9430`) con holgura amplia. Por
> la regla anti-solapamiento del criterio: rango de `val_sf_r2` de `unet_film` ([0.9918,
> 0.9922]) no se solapa con el de `baseline` ([0.9936, 0.9938]) y su mean queda por debajo
> → veredicto **"equivalente", no "mejora"**. Valores crudos por seed en `docs/experiments.md`.

---

## 3. Archivos impactados (resumen)

| Archivo / carpeta | Fase | Naturaleza |
|---|---|---|
| `src/fno_co2/models/variants/unet_film.py` | 1, 2, 3 | **Nuevo** — bloques U-Net/FiLM + `UNetFiLMTemporal` + función `build(cfg)` |
| `src/fno_co2/config.py` | 3 | Añade `unet_depth` (solo afecta a la variante) |
| `configs/experiments/unet_film.yaml` | 3 | **Nuevo** — config autocontenida del experimento |
| `tests/unit/test_unet_film.py` | 4 | **Nuevo** — shapes, backward, MC Dropout, param groups, registry |
| `docs/experiments.md` | 5 | Fila `unet_film` (append; archivo ya versionado, creado por `spec-001` Fase 5) |
| `src/fno_co2/models/registry.py` | — | **NO se modifica** — el despacho es por import dinámico; la variante se auto-registra vía su `build(cfg)` |
| `src/fno_co2/models/fno.py`, `blocks.py` | — | **NO se modifican** (línea base intacta, `spec-001` Fase 3.1) |
| Git: rama `exp/unet-film` | 0 | Desde `development` |

---

## 4. Riesgos y precondiciones

- **Bloqueo por `spec-001`:** sin las Fases 1 y 3 de `spec-001`, no hay `--model-variant`
  ni `build_model`; esta variante no se puede seleccionar. Fase 0 es dura, no opcional.
- **Receptivo global:** el FNO ve todo el dominio vía FFT; la U-Net solo alcanza contexto
  global mediante *downsampling*. Si SF/VD dependen de patrones muy globales del campo de
  presión, una U-Net poco profunda podría rendir peor — de ahí `unet_depth=3` por defecto.
  Es una hipótesis a **medir** (Fase 5), no a asumir.
- **Reconciliación de tamaños (§1.5):** grilla 100×100 no divisible por 8; olvidar el
  *upsample* al tamaño del skip produce errores de concatenación. Cubierto por el test de
  grilla impar (Fase 4.2).
- **Memoria:** el decoder opera sobre `B*T` (p. ej. 4×61=244) a resolución completa — misma
  magnitud que el baseline (que también expande `z` sobre `T` a resolución completa), así
  que la memoria pico es **comparable**, no mayor por definición; verificar en la corrida
  real de todos modos.
- **Comparabilidad (spec-001 §0 y Fase 6):** mismos datos, mismo `train_test_split`
  (no regenerar), mismas seeds, misma loss y métricas. Cambiar cualquiera invalida la
  comparación FNO vs U-Net.
- **`docs/experiments.md` está versionado** (trackeado en git; la fila `baseline` con 3
  seeds ya está commiteada — commit `dc5e991`). La fila `unet_film` se añade por
  `aggregate_experiments.py` (append entre marcadores `<!-- experiment: ... -->`) y se
  commitea como parte del experimento. *(Nota: versiones previas de este spec y de
  `spec-001` §2 lo daban por `.gitignore`; ya no es así.)*
- **Sin dependencias nuevas:** la U-Net usa solo `torch.nn`/`F` ya presentes; **no**
  requiere instalar nada (a diferencia de `pyyaml`, que introduce `spec-001` Fase 2).

---

## 5. Criterios de aceptación

- [ ] `build_model(Config(model_variant="unet_film"))` devuelve `UNetFiLMTemporal`; la
      entrada `"fno_baseline"` sigue devolviendo `PhysicalFNOArchitecture`.
- [ ] `UNetFiLMTemporal.forward(x, d, inj)` devuelve `(B, T, 2, H, W)` en grilla 100×100 y
      en una grilla no potencia-de-2, sin error de shape.
- [ ] La variante entrena con `scripts/train.py --model-variant unet_film` sin tocar
      `training/loop.py` más allá de lo previsto por `spec-001` Fase 3 (uso de `build_model`).
- [ ] MC Dropout produce incertidumbre >0 con la variante (`predict_with_uncertainty`).
- [ ] `build_param_groups` clasifica correctamente `gamma`/`beta` y embeddings en `no_decay`.
- [ ] `fno.py` y `blocks.py` (línea base) quedan **sin modificar**.
- [ ] `configs/experiments/unet_film.yaml` es autocontenido y reproducible por el loader de
      `spec-001` Fase 2.
- [ ] `pytest tests/ -m "not slow"` completo pasa (no rompe `spec-000`/`spec-001`).
- [ ] Overfit de 1 muestra baja la loss (humo de convergencia) antes de cualquier corrida
      real con GPU.
- [ ] La comparación vs. baseline en `docs/experiments.md` cumple `spec-001` Fase 6
      (≥3 seeds, criterio predefinido, mean±std y valores crudos por seed).
