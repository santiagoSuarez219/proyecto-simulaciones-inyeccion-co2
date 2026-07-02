# Modelo-ITM

Pipeline completo de deep learning (ETL + FNO/FiLM) para predicción espacio-temporal del
Factor de Seguridad (SF) y la Deformación Volumétrica (VD) en reservorios depletados bajo
inyección de CO₂: desde archivos crudos de simulación CMG hasta el modelo entrenado.

## Estado

✅ **Fase 2 completada:** código de entrenamiento migrado a estructura modular en
`src/modelo_itm/`.
✅ **Fase A2 completada:** ETL `cmg2tensor` integrado como `src/modelo_itm/etl/`. El
pipeline completo (ETL → modelo → entrenamiento) vive en un único paquete instalable.
Próximo: Parte B (correcciones científicas y del ETL).

## Instalación y uso

### Requisitos previos

- Python 3.12 (ver `CLAUDE.md` para instrucciones de instalación)
- Git
- Simulaciones CMG crudas (`.txt`) si vas a correr el ETL desde cero, o datos ya
  procesados (`data/processed/`) si solo vas a entrenar

### Setup

```bash
# Clonar el repositorio
cd 01-Modelo-ITM

# Crear e activar entorno virtual con Python 3.12
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate

# Instalar en modo editable con extras de desarrollo
pip install --upgrade pip
pip install -e ".[dev]"        # + ".[dev,db]" si vas a usar el ETL a MySQL/SQL Server

# Ejecutar tests para verificar la instalación
pytest tests/ -m "not slow" -v
```

### Procesar datos (ETL)

```bash
# Transformar simulaciones CMG crudas → tensores .pt
python -m modelo_itm.etl \
  --all-simulations --raw-root data/raw --output-dir data/processed \
  --nz 20 --nj 100 --ni 100 \
  --parallel --n-workers 8
```

Ver `python -m modelo_itm.etl --help` para todos los argumentos, y `CLAUDE.md` §Pipeline
de datos para el flujo completo (split, estadísticas globales, transformación).

### Entrenar el modelo

```bash
python scripts/train.py \
  --data-root data/processed \
  --output-dir outputs/ \
  --device cuda \
  --epochs 100 \
  --batch-size 4 \
  --lr 8e-4
```

Ver `python scripts/train.py --help` para todos los argumentos disponibles.

## Documentación

- `CLAUDE.md`: Configuración del proyecto, stack tecnológico, convenciones
- `docs/`: Documentación técnica adicional, incluido el esquema de BD (`database_schema.md`)
- `specs/`: Specs de funcionalidades y hallazgos técnicos
