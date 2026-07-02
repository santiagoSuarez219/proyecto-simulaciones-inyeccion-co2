# Modelo-ITM

Modelo de deep learning (FNO + FiLM) para predicción espacio-temporal del Factor de
Seguridad (SF) y la Deformación Volumétrica (VD) en reservorios depletados bajo
inyección de CO₂.

## Estado

✅ **Fase 2 completada:** Migración de código a estructura modular en `src/modelo_itm/`.
El modelo está listo para entrenamiento. Próximo: Fase B (correcciones científicas).

## Instalación y uso

### Requisitos previos

- Python 3.12 (ver `CLAUDE.md` para instrucciones de instalación)
- Git
- Datos procesados de `cmg2tensor` en `../cmg2tensor/data/processed/`

### Setup

```bash
# Clonar el repositorio
cd 01-Modelo-ITM

# Crear e activar entorno virtual con Python 3.12
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate

# Instalar en modo editable con extras de desarrollo
pip install --upgrade pip
pip install -e ".[dev]"

# Ejecutar tests para verificar la instalación
pytest tests/unit -v
```

### Entrenar el modelo

```bash
python scripts/train.py \
  --data-root ../cmg2tensor/data/processed \
  --output-dir outputs/ \
  --device cuda \
  --epochs 100 \
  --batch-size 4 \
  --lr 8e-4
```

Ver `python scripts/train.py --help` para todos los argumentos disponibles.

## Documentación

- `CLAUDE.md`: Configuración del proyecto, stack tecnológico, convenciones
- `docs/`: Documentación técnica adicional (en desarrollo)
- `specs/`: Specs de funcionalidades y hallazgos técnicos
