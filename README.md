# FNO CO₂ Geomecánico

Pipeline completo de deep learning (ETL + modelo) para predicción espacio-temporal del
Factor de Seguridad (SF) y la Deformación Volumétrica (VD) en reservorios depletados bajo
inyección de CO₂ (almacenamiento geológico de carbono, CCS): desde archivos crudos de
simulación CMG hasta el modelo entrenado, pasando por comparación de arquitecturas
alternativas.

## Instalación y uso

### Requisitos previos

- Python 3.12
- Git
- Simulaciones CMG crudas (`.txt`) si vas a correr el ETL desde cero, o datos ya
  procesados (`data/processed/`) si solo vas a entrenar

### Setup

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd proyecto-simulaciones-inyeccion-co2

# Crear e activar entorno virtual con Python 3.12
# (ajusta la ruta del intérprete a tu instalación: pyenv, Homebrew, apt, etc.)
python3.12 -m venv .venv
source .venv/bin/activate

# Instalar en modo editable con extras de desarrollo
pip install --upgrade pip
pip install -e ".[dev]"        # + ".[dev,db]" si vas a usar el ETL a MySQL/SQL Server

# Ejecutar tests para verificar la instalación
pytest tests/ -m "not slow" -v
```

`pyproject.toml` es la fuente de verdad de las dependencias. Alternativamente, para instalar
solo las dependencias base sin modo editable (p. ej. en una imagen Docker o servidor),
usa `requirements.txt` (versiones fijadas a la combinación verificada del proyecto):

```bash
pip install -r requirements.txt
```

### Procesar datos (ETL)

```bash
# Transformar simulaciones CMG crudas → tensores .pt
python -m fno_co2.etl \
  --all-simulations --raw-root data/raw --output-dir data/processed \
  --nz 20 --nj 100 --ni 100 \
  --parallel --n-workers 8
```

Ver `python -m fno_co2.etl --help` para todos los argumentos disponibles. 

### Entrenar el modelo (una sola corrida)

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

### Campañas de experimentos

Para comparar arquitecturas alternativas (p. ej. baseline FNO+FiLM vs. U-Net con FiLM
vs. FNO+atención axial) con múltiples semillas por variante:

```bash
# Solo validar la configuración y ver la cola de corridas, sin entrenar
python scripts/run_campaign.py --config configs/campaigns/fno_vs_unet_vs_attn.yaml --dry-run

# Correr la campaña completa (--yes confirma explícitamente la ejecución real)
python scripts/run_campaign.py --config configs/campaigns/fno_vs_unet_vs_attn.yaml --yes

# Agregar métricas finales (actualiza docs/experiments.md) y generar figuras comparativas
python scripts/aggregate_campaign.py --config configs/campaigns/fno_vs_unet_vs_attn.yaml
python scripts/plot_campaign_comparison.py --campaign-dir outputs/campaigns/fno_vs_unet_vs_attn
```

Las arquitecturas alternativas viven en `src/fno_co2/models/variants/` (registradas en
`src/fno_co2/models/registry.py`); `run_campaign.py` corre cada variante × semilla de
forma independiente, y `aggregate_campaign.py` consolida los resultados en un reporte de
máquina (`campaign_report.md`) con comparación estadística contra la línea base.

## Documentación

- `resultados/`: informes de divulgación curados para lectura de principio a fin, incluido
  el [informe de resultados de la campaña `fno_vs_unet_vs_attn`](resultados/informe-resultados-campana-fno-vs-unet-vs-attn.md)
  (FNO baseline vs. U-Net FiLM vs. FNO+atención axial)
- **Checkpoints (`best.pt`)**: no se versionan en git (`.gitignore` — ver `outputs/checkpoints/`).
  Los 9 `best.pt` de la campaña `fno_vs_unet_vs_attn` (baseline/unet_film/fno_axial_attn ×
  3 semillas) están compartidos en Google Drive:
  [carpeta `fno_co2_checkpoints_fno_vs_unet_vs_attn`](https://drive.google.com/open?id=1iMZUkNlLJrKr23O6vLyhKuXc0gkA-lZ3)
