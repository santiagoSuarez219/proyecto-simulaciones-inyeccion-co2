# spec-003 — Variante de arquitectura FNO + atención espacial axial [TESTING]

> **Autor:** rol `@architect`
> **Fecha:** 2026-07-02 · **Actualizado:** 2026-07-15 (Fases 1–4 implementadas)
> **Estado:** [TESTING] — Fases 1–4 completas (AxialAttentionBlock, FNOAxialAttention, config,
> registry discovery, YAML; tests: 16/16 ✅). Fase 5 (humo + experimento multi-seed con GPU)
> pendiente de confirmación del usuario.
> **Depende de:** `spec-001` (framework de experimentación y comparación de
> arquitecturas). Es una **variante estructural** más dentro de ese framework
> (spec-001 §Fase 3): vive en `models/variants/`, se registra en `build_model(cfg)` y se
> compara contra la línea base con el proceso multi-seed. Paralela a `spec-002`
> (variante U-Net), independiente de ella.
> **Objetivo:** implementar una variante **híbrida** que **conserva los `FiLMSpectralBlock`
> (FFT) del baseline** y les **añade bloques de atención espacial axial** (self-attention
> sobre la grilla H×W, factorizada en filas y columnas), manteniendo intacto el
> condicionamiento temporal FiLM. Seleccionable por `cfg.model_variant`. No modifica la
> línea base (`PhysicalFNOArchitecture`) in-place.

---

## 0. Contexto y motivación

- **Hipótesis física.** El Factor de Seguridad y la deformación volumétrica dependen de un
  campo de presión que se propaga de forma **global** desde los pozos (TENE-1, TENE-2). El
  FNO ya captura globalidad vía FFT (modos espectrales truncados a `spectral_modes=16`), pero
  esa truncación **descarta los modos altos**. La atención espacial ofrece un mecanismo de
  acoplamiento global **complementario y adaptativo** (aprende qué posiciones se relacionan,
  no solo las de baja frecuencia). La hipótesis a **medir** es si añadir atención mejora la
  predicción de SF/VD sobre el FNO puro.
- **Cambio incremental (una cosa a la vez).** Se **conserva** el núcleo espectral y se
  **añade** atención intercalada — no se reemplaza. Así la comparación vs. baseline aísla el
  efecto de "añadir atención", respetando el principio rector de `spec-001` §0.
- **Por qué atención axial y no densa.** La grilla es 100×100 = **10 000 tokens**; la
  self-attention densa es O(N²) ≈ 10⁸ por mapa de features y por timestep — inviable en
  memoria. La **atención axial** factoriza en atención sobre filas (H) y sobre columnas (W),
  bajando a O(H·W·(H+W)) ≈ 2·10⁶: dos posiciones cualesquiera se comunican en **dos saltos**
  (una fila + una columna), conservando alcance global a costo tratable.
- **Qué se conserva del baseline (contrato duro).** Debe ser *drop-in*: se instancia desde
  `build_model(cfg)`, entrena con el mismo `training/loop.py`, la misma loss, el mismo
  `build_param_groups`, y es compatible con MC Dropout. Ver §2 (Contrato de interfaz).

---

## 1. Diseño de la arquitectura `FNOAxialAttention`

### 1.1 Idea general

Idéntica al baseline salvo el núcleo: se codifica el estado estático una vez, se expande
sobre `T` y se aplica FiLM por timestep. La diferencia es que, **intercalado** con cada
`FiLMSpectralBlock`, se inserta un **bloque de atención axial** que refina las features
espacialmente.

| | Línea base (`PhysicalFNOArchitecture`) | Variante (`FNOAxialAttention`) |
|---|---|---|
| Núcleo | 4× `FiLMSpectralBlock` (FFT2 → modos → iFFT2) | 4× [ `FiLMSpectralBlock` **+** `AxialAttentionBlock` ] |
| Acoplamiento global | Fourier (modos bajos, truncados) | Fourier **+** atención (adaptativa, dos saltos fila/col) |
| Condicionamiento | FiLM por timestep en cada bloque espectral | **Igual** — FiLM sigue en los `FiLMSpectralBlock` |
| Costo extra | — | Atención axial O(H·W·(H+W)) por bloque y timestep |

### 1.2 Flujo (`forward(x, d, inj)`)

```
Entrada (idéntica al baseline):
  x=(B,4,H,W)   d=(B,1)   inj=(B,T,2)

1. depth_map = d.view(B,1,1,1).expand(B,1,H,W)
   z = encoder(concat([x, depth_map]))            → (B, C, H, W)   (in_c=5, C=hidden_dim)

2. cond_seq = t_embed(arange(T)) + cond_mlp(concat([inj, depth_seq]))   → (B, T, cond_dim)
   (reutiliza VERBATIM el condicionamiento del baseline: padding/truncado de inj incluido)

3. z_bt   = z expandido a (B*T, C, H, W)
   cond_bt = cond_seq.reshape(B*T, cond_dim)

4. NÚCLEO (4 bloques, intercalado):
   for spectral, attn in zip(fno_blocks, attn_blocks):
       z_bt = spectral(z_bt, cond_bt)             # FFT + FiLM (condicionamiento temporal)
       z_bt = attn(z_bt)                          # atención axial (residual, sin condicionar)

5. z_bt = decoder(z_bt)                            → (B*T, 2, H, W)
   return z_bt.view(B, T, 2, H, W)                 — SF y VD para todos los T pasos
```

> **Decisión de condicionamiento:** el `AxialAttentionBlock` es un sublayer **residual no
> condicionado** (la información de inyección/profundidad/timestep sigue entrando por los
> `FiLMSpectralBlock`, sin cambios). Esto aísla el efecto "añadir atención" del efecto
> "cambiar el condicionamiento" — una cosa a la vez. (Alternativa documentada como deuda:
> modular también la atención con FiLM; fuera de alcance de este spec.)

### 1.3 Bloque nuevo: `AxialAttentionBlock`

En `src/fno_co2/models/variants/fno_axial_attn.py`:

- **Normalización:** `GroupNorm` sobre canales (consistente con la opción `use_group_norm`
  del baseline y estable sin depender del tamaño de batch, que aquí es `B*T`).
- **Positional encoding:** codificación posicional aprendida (o sinusoidal) **por eje**
  (filas y columnas) sumada antes de la atención — la atención es permutación-invariante y
  la posición espacial importa físicamente (distancia a los pozos). **Implementarlo como
  `nn.Embedding`** (una por eje) o como **buffer sinusoidal** (`register_buffer`, no es
  parámetro): así `build_param_groups` (name-based, **no se edita** — ver §2.5) lo deja
  fuera de weight decay automáticamente (los `nn.Embedding` caen en `no_decay`; un buffer no
  es parámetro). **Evitar un `nn.Parameter` crudo**, que caería en `decay` sin quererlo.
- **Atención por filas (eje H):** reordenar `(B*T, C, H, W)` para tratar `W` como parte del
  batch → secuencias de longitud `H`; MHSA multi-cabeza (`cfg.attn_heads`).
- **Atención por columnas (eje W):** análogo con `H` en el batch → secuencias de longitud `W`.
- **Residual + dropout:** `z = z + dropout(attn_axial(norm(z) + pos))`. El `nn.Dropout`
  aquí **también** alimenta MC Dropout (además del `Dropout2d` de los `ResBlock` del
  encoder/decoder).
- **Complejidad:** O(H·W·(H+W)·C) por bloque y timestep, frente a O((H·W)²·C) de la
  densa — este es el punto de usar atención axial.

### 1.4 Hiperparámetros de la variante

- **`attn_heads: int = 4`** — número de cabezas de la atención axial. `hidden_dim` debe ser
  divisible por `attn_heads` (validar en el constructor con error explícito).
- **`attn_num_blocks: int = 4`** — cuántos de los 4 bloques del núcleo llevan atención
  intercalada (default: todos). **Palanca de memoria/costo:** bajarlo (p. ej. a 2, aplicando
  atención solo en los últimos bloques) reduce cómputo si aparece OOM en la corrida real
  (ver §4 Riesgos). Reutiliza `hidden_dim`, `spectral_modes`, `dropout_p`, `use_group_norm`
  del baseline sin campos nuevos para esos.

---

## 2. Contrato de interfaz (debe cumplirse para ser *drop-in*)

Idéntico al del baseline y al de `spec-002` §2:

1. **Firma:** `forward(x, d, inj) -> Tensor` con salida exacta `(B, T, 2, H, W)`.
2. **Entrada:** `x=(B,4,H,W)`, `d=(B,1)`, `inj=(B,T,2)`; concatena `depth_map` (in_c=5);
   misma guarda de padding/truncado de `inj` a `T`.
3. **`time_steps`:** atributo `self.time_steps` presente.
4. **MC Dropout:** capas `nn.Dropout2d`/`nn.Dropout` con `cfg.dropout_p` para incertidumbre
   real (>0) con `predict_with_uncertainty`.
5. **Param groups:** los `gamma`/`beta` de los `FiLMSpectralBlock` reutilizados y los
   embeddings siguen cayendo en `no_decay` vía `build_param_groups` (name-based). Los pesos
   de la atención (`Linear` Q/K/V/proj) van a `decay` como cualquier `Linear` — es lo
   estándar; sus **bias** caen en `no_decay` por la regla `.bias`.
6. **Constructor / descubrimiento:** el módulo `fno_co2.models.variants.fno_axial_attn`
   expone una función `build(cfg) -> nn.Module` que instancia `FNOAxialAttention` leyendo
   solo campos de `Config`. `registry.build_model(cfg)` la descubre por `importlib` a partir
   de `cfg.model_variant` — **no se edita `registry.py`** (convención de `spec-001` Fase 3,
   confirmada por `spec-002` `[DONE]`).

---

## Fase 0 — Precondiciones

1. ✅ **`spec-001` Fase 1 (CUMPLIDA):** `--model-variant` / `Config.model_variant`
   (`config.py:20`) y `build_run_signature` usando `cfg.model_variant` ya existen.
2. ✅ **`spec-001` Fase 3 (CUMPLIDA):** existen `models/registry.py::build_model` y el
   directorio `models/variants/`; `training/loop.py:239` ya instancia vía `build_model(cfg)`
   (no `PhysicalFNOArchitecture` directo). El registry **despacha por `importlib`** al módulo
   `fno_co2.models.variants.<model_variant>` esperando una función `build(cfg)` — no hay que
   editarlo para añadir la variante.
3. ⬜ Rama `exp/attention-spatial` creada **desde `development`** (`CLAUDE.md` §Git). **No**
   trabajar sobre `main`/`development`. (Única precondición pendiente.)

**Verificación:** `--model-variant` aceptado por `scripts/train.py`; `build_model` despacha
`"fno_baseline"` → `PhysicalFNOArchitecture` (test de `spec-001`, ya en verde).

---

## Fase 1 — Bloque de atención axial

**Dónde:** `src/fno_co2/models/variants/fno_axial_attn.py` (nuevo).

1. Implementar `AxialAttentionBlock` (ver §1.3): norm + positional encoding por eje +
   atención por filas + atención por columnas + residual + dropout.
2. Validar `hidden_dim % attn_heads == 0` con error explícito.
3. **No** modificar `blocks.py` ni `fno.py`; **reutilizar** `FiLMSpectralBlock`/`ResBlock`
   importándolos (composición, no herencia).

**Verificación (parte en Fase 4):** un `AxialAttentionBlock` con entrada `(N, C, H, W)`
devuelve `(N, C, H, W)` (mismo shape), en grilla cuadrada y no cuadrada.

---

## Fase 2 — Clase `FNOAxialAttention`

**Dónde:** `src/fno_co2/models/variants/fno_axial_attn.py`.

1. Implementar `FNOAxialAttention(nn.Module)` con el `forward` de §1.2 y el contrato de §2.
2. Reutilizar **verbatim** el encoder, el decoder y el condicionamiento temporal del baseline
   (mismos `t_embed`, `cond_mlp`, `cond_dim=128`, guarda de `inj`).
3. Construir `fno_blocks` (4× `FiLMSpectralBlock`) y `attn_blocks`
   (`attn_num_blocks`× `AxialAttentionBlock`), intercalados en el bucle del núcleo. Si
   `attn_num_blocks < 4`, aplicar atención solo en los últimos bloques.
4. Exponer `self.time_steps` y firmar `forward(x, d, inj)`.

**Verificación (parte en Fase 4):** `forward` con `(B=2,4,100,100)`, `d=(2,1)`,
`inj=(2,T,2)` → `(2, T, 2, 100, 100)`; y con `(2,4,30,26)` → `(2, T, 2, 30, 26)`.

---

## Fase 3 — Integración: Config, función `build` y config de experimento

**Dónde:** `src/fno_co2/config.py`, `src/fno_co2/models/variants/fno_axial_attn.py`
(la función `build`), `configs/experiments/fno_axial_attn.yaml`.
**`registry.py` NO se toca.**

1. `Config`: añadir `attn_heads: int = 4` y `attn_num_blocks: int = 4` (documentados:
   **solo afectan a la variante `fno_axial_attn`**; el baseline los ignora).
2. **Función `build(cfg) -> nn.Module`** en `fno_axial_attn.py` (misma convención que
   `unet_film.py::build` de `spec-002`): construye `FNOAxialAttention` leyendo `time_steps`,
   `hidden_dim` (como `h_dim`), `spectral_modes` (como `modes`), `dropout_p`, `use_group_norm`,
   `attn_heads`, `attn_num_blocks` de `cfg`, con `in_c=5` y `cond_dim=128`. El registry la
   descubre por `importlib` con `model_variant="fno_axial_attn"` **sin editar `registry.py`**;
   `"fno_baseline"` sigue intacto y una variante desconocida sigue lanzando `ValueError`.
3. `configs/experiments/fno_axial_attn.yaml`: config **completa y autocontenida** (misma
   estructura que `baseline.yaml`), idéntica al baseline salvo `model_variant:
   fno_axial_attn` y `attn_heads`/`attn_num_blocks`.

**Verificación:** `build_model(Config(model_variant="fno_axial_attn"))` devuelve
`FNOAxialAttention`; el YAML round-trip por el loader de `spec-001` Fase 2 produce el
`Config` esperado.

---

## Fase 4 — Tests unitarios

**Dónde:** `tests/unit/test_fno_axial_attn.py` (nuevo). Todos rápidos, sin datos reales,
sin `@pytest.mark.slow`:

1. **Shape del bloque:** `AxialAttentionBlock` preserva `(N,C,H,W)` (cuadrada y no cuadrada).
2. **Shape del modelo:** `forward` → `(B, T, 2, H, W)` en 100×100 y en grilla no cuadrada
   (30×26).
3. **Backward:** `loss.backward()` produce gradientes finitos (sin `NaN`/`Inf`).
4. **MC Dropout real:** con `dropout_p>0` y `model.train()`, dos `forward` de la misma
   entrada difieren → incertidumbre >0.
5. **Param groups:** `build_param_groups` deja los `gamma`/`beta` de FiLM, los embeddings y
   el positional encoding (si es `nn.Embedding`) en `no_decay`; los pesos de atención en
   `decay` y sus bias en `no_decay`.
6. **Validación de heads:** `hidden_dim` no divisible por `attn_heads` lanza error explícito.
7. **Registry (sin editarlo):** `build_model(Config(model_variant="fno_axial_attn"))` devuelve
   la clase vía descubrimiento por `importlib`; string desconocido lanza `ValueError`.

**Verificación:** `pytest tests/unit/test_fno_axial_attn.py -v` pasa; `pytest tests/ -m
"not slow"` completo sigue verde (no rompe `spec-000`/`spec-001`/`spec-002`).

---

## Fase 5 — Humo de convergencia y experimento comparativo (Pendiente)

> Usa el framework de `spec-001`; no reimplementa entrenamiento.
> **Requisito:** confirmación explícita del usuario para ejecutar entrenamientos con GPU.

1. **Humo:** `scripts/train.py --model-variant fno_axial_attn --overfit-sample-idx 0 --epochs 5`
   ~épocas; la loss debe **bajar** (detecta errores de arquitectura antes de gastar GPU).
   ```bash
   source .venv/bin/activate
   python scripts/train.py --model-variant fno_axial_attn --overfit-sample-idx 0 --epochs 5
   ```

2. **Criterio de éxito predefinido (⚠️ fijar ANTES de correr, `spec-001` Fase 6):** escribir
   la fila en `docs/experiments.md` con hipótesis y umbral *antes* de ver resultados.
   Propuesta a confirmar:
   > *"`fno_axial_attn` reduce `val_sf_rmse` mean en ≥5% sin degradar `val_vd_r2`, con
   > ≥3 seeds; se acepta el costo extra de cómputo solo si la mejora supera ese umbral."*

3. **Corrida real (requiere GPU + datos post-C1 + baseline congelada):**
   ```bash
   python scripts/run_experiment.py --config configs/experiments/fno_axial_attn.yaml --n-seeds 3
   ```
   **🔴 REQUIERE CONFIRMACIÓN EXPLÍCITA DEL USUARIO** (§Despliegue de `CLAUDE.md`).

4. **Agregación:** `scripts/aggregate_experiments.py` añade la fila `fno_axial_attn` a
   `docs/experiments.md` con mean±std, tamaño de efecto y valores por seed vs. baseline.
   Sin conclusiones de "mejor/peor" con <3 seeds ni rangos mean±std solapados
   (`spec-001` Fase 6).

**Verificación:** el overfit baja la loss; `docs/experiments.md` tiene la fila
`fno_axial_attn` con criterio predefinido y resultados multi-seed vs. baseline.

---

## 3. Archivos impactados (resumen)

| Archivo / carpeta | Fase | Naturaleza |
|---|---|---|
| `src/fno_co2/models/variants/fno_axial_attn.py` | 1, 2, 3 | **Nuevo** — `AxialAttentionBlock` + `FNOAxialAttention` + función `build(cfg)` |
| `src/fno_co2/config.py` | 3 | Añade `attn_heads`, `attn_num_blocks` (solo afectan a la variante) |
| `src/fno_co2/models/registry.py` | — | **NO se modifica** — descubre la variante por `importlib`; la variante expone `build(cfg)` |
| `configs/experiments/fno_axial_attn.yaml` | 3 | **Nuevo** — config autocontenida del experimento |
| `tests/unit/test_fno_axial_attn.py` | 4 | **Nuevo** — shapes, backward, MC Dropout, param groups, heads, registry |
| `docs/experiments.md` | 5 | Fila `fno_axial_attn` (append; archivo de `spec-001` Fase 5) |
| `src/fno_co2/models/fno.py`, `blocks.py` | — | **NO se modifican** (línea base intacta; se reutilizan por import) |
| Git: rama `exp/attention-spatial` | 0 | Desde `development` |

---

## 4. Riesgos y precondiciones

- **Bloqueo por `spec-001`:** sin las Fases 1 y 3 de `spec-001` no hay `--model-variant` ni
  `build_model`; la variante no se puede seleccionar. Fase 0 es dura.
- **Memoria/cómputo (riesgo principal):** la atención axial se aplica sobre `B*T` mapas
  (p. ej. 4×61 = 244) a resolución completa 100×100. Aunque axial ≪ densa, sigue siendo el
  componente más caro del modelo. **Mitigación integrada:** `attn_num_blocks` permite
  reducir a 2 (o menos) bloques con atención, o `batch_size` puede bajarse; documentar el
  costo real medido en la primera corrida. Si aún hay OOM, evaluar aplicar atención en una
  versión *downsampled* de las features (fuera de alcance de este spec — deuda).
- **Duplicación de mecanismo global:** FNO (Fourier) y atención capturan globalidad; podrían
  ser **redundantes** y no aportar sobre el baseline. Ese es precisamente el resultado que
  el experimento (Fase 5) debe determinar — no se asume mejora.
- **Positional encoding imprescindible:** sin él la atención pierde la noción de posición
  (distancia a los pozos), físicamente relevante; su ausencia sería un bug silencioso.
  Cubierto por el diseño (§1.3) y verificado indirectamente por el humo de convergencia.
- **Comparabilidad (`spec-001` §0 y Fase 6):** mismos datos, split inmutable, mismas seeds,
  misma loss/métricas. Cambiar cualquiera invalida la comparación FNO vs FNO+atención.
- **`docs/experiments.md` en `.gitignore`** (`spec-001` §2): registro local; versionarlo es
  decisión del usuario, fuera de alcance.
- **Sin dependencias nuevas:** la atención usa `torch.nn.MultiheadAttention` o `Linear`
  propios ya disponibles; **no** requiere instalar nada.

---

## 5. Criterios de aceptación

- [ ] `build_model(Config(model_variant="fno_axial_attn"))` devuelve `FNOAxialAttention`;
      `"fno_baseline"` sigue devolviendo `PhysicalFNOArchitecture`.
- [ ] `FNOAxialAttention.forward(x, d, inj)` devuelve `(B, T, 2, H, W)` en 100×100 y en una
      grilla no cuadrada, sin error de shape.
- [ ] El `AxialAttentionBlock` preserva el shape espacial y usa atención axial (no densa).
- [ ] `hidden_dim % attn_heads != 0` falla con error explícito.
- [ ] Entrena con `scripts/train.py --model-variant fno_axial_attn` **sin tocar
      `training/loop.py`** (ya usa `build_model(cfg)` desde `spec-001` Fase 3, `loop.py:239`).
- [ ] MC Dropout produce incertidumbre >0 con la variante.
- [ ] `build_param_groups` clasifica FiLM/embeddings (incl. el positional encoding si es
      `nn.Embedding`) en `no_decay` y los pesos de atención en `decay` (bias en `no_decay`).
- [ ] `fno.py`, `blocks.py` y `registry.py` quedan **sin modificar** (la variante se descubre
      por `importlib` vía su función `build(cfg)`).
- [ ] `configs/experiments/fno_axial_attn.yaml` es autocontenido y reproducible por el loader
      de `spec-001` Fase 2.
- [ ] `pytest tests/ -m "not slow"` completo pasa (no rompe specs previos).
- [ ] Overfit de 1 muestra baja la loss antes de cualquier corrida real con GPU.
- [ ] La comparación vs. baseline en `docs/experiments.md` cumple `spec-001` Fase 6
      (≥3 seeds, criterio predefinido, mean±std y valores crudos por seed).
