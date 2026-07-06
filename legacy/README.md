# legacy/ — Código heredado (solo referencia)

Snapshot del código previo al proyecto **fno_co2** (`01-Modelo-ITM`). Se conserva
como **referencia histórica**: NO se importa desde `src/`, NO se ejecuta como parte
del pipeline actual y NO se mantiene. Sirve para consultar decisiones y lógica del
modelo anterior mientras se construye el nuevo.

## Contenido

- **`cmg2tensor/`** — ETL antiguo (predecesor de `src/fno_co2/etl`). Transformaba
  las salidas CMG a tensores. Incluye scripts, notebooks, SQL y tests originales.
  Trae su propio `.gitignore` (se preservó tal cual, por eso el snapshot se añadió
  con `git add -f`).
- **`codigo-entrenamiento/`** — script de entrenamiento original (`train_dataset.py`),
  antecesor de `src/fno_co2/training`.

## Notas

- Copiado (no movido) desde la raíz del proyecto; se excluyeron caches
  (`__pycache__`, `.pytest_cache`, `.venv`).
- El código nuevo y soportado vive en `src/fno_co2/`. Ante cualquier discrepancia,
  manda el código nuevo.
