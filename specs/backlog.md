# Backlog de Deuda Técnica

## [spec-002-debt-001] Optimización GPU de U-Net temporal

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

---

## [spec-002-debt-002] Investigar convergencia deficiente

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

