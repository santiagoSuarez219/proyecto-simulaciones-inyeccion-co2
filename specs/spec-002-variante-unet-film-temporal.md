# spec-002 — Variante de arquitectura U-Net con condicionamiento FiLM temporal

> **Autor:** rol `@architect`
> **Fecha:** 2026-07-02
> **Estado:** PLANIFICADO — **bloqueado** por `spec-001` (requiere sus Fases 1 y 3
> implementadas; ver Fase 0). No iniciar hasta que exista `models/registry.py` y el
> flag `--model-variant`.
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
   (verificado contra el shape que consume la loss en `training/loop.py:79` y la
   incertidumbre en `inference/uncertainty.py`).
2. **Entrada:** `x=(B,4,H,W)`, `d=(B,1)`, `inj=(B,T,2)`; concatena `depth_map` internamente
   (in_c=5). Misma guarda de padding/truncado de `inj` a `T` que el baseline.
3. **`time_steps`:** atributo `self.time_steps` presente (lo usan resume/firma y el
   expandido temporal).
4. **MC Dropout:** contiene capas `nn.Dropout2d(cfg.dropout_p)` (vía `ResBlock`) para que
   `predict_with_uncertainty` produzca incertidumbre real (>0) al forzar modo train.
5. **Param groups:** submódulos FiLM nombrados `.gamma`/`.beta` y embeddings como
   `nn.Embedding` → `build_param_groups` los clasifica en `no_decay` sin cambios.
6. **Constructor:** instanciable desde `build_model(cfg)` leyendo solo campos de `Config`
   (nada hardcodeado fuera de `Config`, §Convenciones de `CLAUDE.md`).

---

## Fase 0 — Precondiciones (bloqueantes)

1. **`spec-001` Fase 1 implementada:** `--model-variant` en CLI y `Config.model_variant`;
   `build_run_signature` usando `cfg.model_variant` (no el `model_name` hardcodeado).
2. **`spec-001` Fase 3 implementada:** existe `src/fno_co2/models/registry.py::build_model`
   y el directorio `src/fno_co2/models/variants/`; `training/loop.py` usa `build_model(cfg)`
   en vez de instanciar `PhysicalFNOArchitecture` directamente (hoy en `loop.py:211`).
3. Rama `exp/unet-film` creada **desde `development`** (convención `CLAUDE.md` §Git;
   `spec-001` §Fase 3.4). **No** trabajar sobre `main`/`development`.

**Verificación:** `build_model(Config(model_variant="fno_baseline"))` devuelve
`PhysicalFNOArchitecture` (test de `spec-001`); `--model-variant` aceptado por
`scripts/train.py` sin error.

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

## Fase 3 — Integración: Config, registry y config de experimento

**Dónde:** `src/fno_co2/config.py`, `src/fno_co2/models/registry.py`,
`configs/experiments/unet_film.yaml`.

1. `Config`: añadir `unet_depth: int = 3` (documentado: **solo afecta a `unet_film`**; el
   baseline lo ignora). No añadir campo de canales (se deriva de `hidden_dim`, §1.4).
2. `registry.py::build_model`: registrar `"unet_film"` → `UNetFiLMTemporal(...)`, leyendo
   `time_steps`, `hidden_dim`, `dropout_p`, `use_group_norm`, `unet_depth` de `cfg`. La
   entrada `"fno_baseline"` (de `spec-001`) queda intacta; variante desconocida sigue
   lanzando `ValueError` explícito.
3. `configs/experiments/unet_film.yaml`: config **completa y autocontenida** (misma
   estructura que `baseline.yaml` de `spec-001` Fase 2), idéntica al baseline salvo
   `model_variant: unet_film` y, si aplica, `unet_depth`. Es el artefacto reproducible del
   experimento.

**Verificación:** `build_model(Config(model_variant="unet_film"))` devuelve
`UNetFiLMTemporal`; cargar `configs/experiments/unet_film.yaml` produce un `Config` con
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
   *antes* de ver resultados. Propuesta a confirmar con el usuario:
   > *"`unet_film` iguala o supera a la línea base en `val_sf_r2` (criterio: no degradar
   > `val_sf_r2` mean en más de 2% y no degradar `val_vd_r2`), con ≥3 seeds."*
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

---

## 3. Archivos impactados (resumen)

| Archivo / carpeta | Fase | Naturaleza |
|---|---|---|
| `src/fno_co2/models/variants/unet_film.py` | 1, 2 | **Nuevo** — bloques U-Net/FiLM + `UNetFiLMTemporal` |
| `src/fno_co2/config.py` | 3 | Añade `unet_depth` (solo afecta a la variante) |
| `src/fno_co2/models/registry.py` | 3 | Registra `"unet_film"` (creado por `spec-001` Fase 3) |
| `configs/experiments/unet_film.yaml` | 3 | **Nuevo** — config autocontenida del experimento |
| `tests/unit/test_unet_film.py` | 4 | **Nuevo** — shapes, backward, MC Dropout, param groups, registry |
| `docs/experiments.md` | 5 | Fila `unet_film` (append; archivo creado por `spec-001` Fase 5) |
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
- **`docs/experiments.md` está en `.gitignore`** (decisión previa; ver `spec-001` §2): el
  registro vive local. Si el paper necesita el registro versionado, es decisión del usuario
  fuera del alcance de este spec.
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
