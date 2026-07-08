# Estructura de Base de Datos — Simulaciones Geomecánicas CO₂ (CCS)

## Contexto

Base de datos para gestionar resultados de **simulaciones geomecánicas de inyección de CO₂** (captura y almacenamiento de carbono). El modelo `CO2_951` representa escenarios de inyección en el campo TENE (pozos TENE-1 y TENE-2).

---

## Resumen

| Elemento | Detalle |
|---|---|
| Motor | MySQL 8.0 |
| Base de datos | `SimulacionesCO2` |
| Tablas | 6 |
| Corridas totales | 260 (208 train / 52 test) |
| Variables espaciales | AFI, COHESION, SF, VD |
| Propiedades de reservorio | POROSITY, PERMEABILITY |
| Pozos de inyección | TENE-1, TENE-2 |
| Malla 3D | 97 capas × 50 × 50 celdas |
| Timesteps dinámicos | 362 (mensual, ~30 años) |

---

## Diagrama de Relaciones

```
tblModelo (1)
├──── (N) tblCorrida
│              ├──── (N) tblResultadoVariables_Header
│              │              └──── (N) tblResultadoVariables_Detalle
│              └──── (N) tblTasaInyeccion
└──── (N) tblPropiedadesMalla
```

---

## Tablas

### 1. `tblModelo`
Catálogo de modelos geomecánicos. Define las dimensiones de la malla necesarias para deserializar los blobs de datos espaciales.

| Campo | Tipo | Clave | Descripción |
|---|---|---|---|
| Modelo | VARCHAR(50) | PK | Nombre del modelo — ej. `CO2_951` |
| DimK | INT | | Capas verticales (97) |
| DimJ | INT | | Filas de malla (50) |
| DimI | INT | | Columnas de malla (50) |
| Descripcion | VARCHAR(200) | | Descripción libre |

---

### 2. `tblCorrida`
Tabla maestra de corridas/simulaciones. Cada corrida representa una ejecución del modelo con parámetros de muestreo distintos.

| Campo | Tipo | Clave | Descripción |
|---|---|---|---|
| ID_Corrida | INT | PK | Identificador único (auto) |
| Modelo | VARCHAR(50) | FK → tblModelo | Modelo geomecánico |
| NombreCorrida | VARCHAR(100) | UQ | Nombre completo del directorio — ej. `003_normal` |
| CodigoCorrida | INT | | Prefijo numérico (no único por sí solo) |
| TipoMuestreo | VARCHAR(50) | | `normal`, `LatinHyperCube`, `high_to_low`, `low_to_high`, `inverse_normal` |
| Particion | VARCHAR(10) | | `train` / `test` |
| FechaModelo | DATE | | Fecha de referencia del modelo |
| FechaCarga | DATETIME | | Fecha y hora de carga al sistema |

**Volumen:** 260 filas

---

### 3. `tblPropiedadesMalla`
Propiedades estáticas del reservorio: **iguales en todas las corridas**, cargadas una sola vez por modelo. Cada fila almacena el grid completo de una capa K como blob binario float32.

| Campo | Tipo | Clave | Descripción |
|---|---|---|---|
| ID_Propiedad | INT | PK | Identificador único (auto) |
| Modelo | VARCHAR(50) | FK → tblModelo | Modelo asociado |
| Variable | VARCHAR(50) | | `POROSITY`, `PERMEABILITY` |
| K | INT | | Índice de capa (1..97) |
| CapaDatos | LONGBLOB | | Grid 50×50 en float32 (~10 KB) |

**Volumen:** 194 filas (2 variables × 97 capas)

> **Nota de diseño:** separadas de `tblResultadoVariables_Detalle` porque no varían entre corridas. Repetirlas 260 veces sería redundante.

---

### 4. `tblResultadoVariables_Header`
Encabezado de resultados por variable geomecánica, corrida y paso temporal. Actúa como índice para localizar los datos espaciales.

| Campo | Tipo | Clave | Descripción |
|---|---|---|---|
| ID_Header | INT | PK | Identificador único (auto) |
| ID_Corrida | INT | FK → tblCorrida | Corrida asociada |
| Variable | VARCHAR(50) | | `AFI`, `COHESION`, `SF`, `VD` |
| TimeStep | INT | | Índice temporal (0..361) |
| FechaResultado | DATE | | Fecha del resultado |
| Unidad | VARCHAR(20) | | `deg`, `psi`, `adim`, `ft` |
| ValorMin | DECIMAL(10,4) | | Mínimo en la malla (opcional) |
| ValorMax | DECIMAL(10,4) | | Máximo en la malla (opcional) |

**Volumen:** ~188,760 filas
`260 corridas × (2 vars estáticas × 1 ts + 2 vars dinámicas × 362 ts)`

---

### 5. `tblResultadoVariables_Detalle`
Detalle espacial de resultados por capa. Cada fila almacena el **grid completo 50×50** de una capa K como blob binario, en lugar de una fila por celda.

| Campo | Tipo | Clave | Descripción |
|---|---|---|---|
| ID_Detalle | INT | PK | Identificador único (auto) |
| ID_Header | INT | FK → tblResultadoVariables_Header | Header asociado |
| K | INT | | Índice de capa (1..97) |
| CapaDatos | LONGBLOB | | Grid 50×50 en float32 (~10 KB) |

**Volumen:** ~18.3 M filas / ~183 GB

> **Decisión de diseño clave:** almacenar el grid por capa como blob en lugar de una fila por celda (K, J, I, Valor) reduce de **~45,900 M filas** a **~18.3 M filas** (×2,500 menos).

#### Formato del blob `CapaDatos`
```
float32[DimJ × DimI]  —  orden row-major  (J=0..49, I=0..49)
Tamaño: 50 × 50 × 4 bytes = 10,000 bytes ≈ 10 KB
```
Para leer en Python:
```python
import numpy as np
grid = np.frombuffer(capa_datos, dtype=np.float32).reshape(50, 50)
```

---

### 6. `tblTasaInyeccion`
Series temporales de tasa de inyección de gas por pozo y corrida.

| Campo | Tipo | Clave | Descripción |
|---|---|---|---|
| ID_TasaInyeccion | INT | PK | Identificador único (auto) |
| ID_Corrida | INT | FK → tblCorrida | Corrida asociada |
| NombrePozo | VARCHAR(50) | | `TENE-1`, `TENE-2` |
| TimeStep | INT | | Índice temporal (0..361) |
| TimeDay | DECIMAL(18,9) | | Días desde inicio de simulación |
| Fecha | DATETIME | | Fecha y hora del registro |
| Parametro | VARCHAR(100) | | `Gas Rate SC - Monthly (ft3/day)` |
| Valor | DECIMAL(18,6) | | Valor de la tasa |
| FileModelo | VARCHAR(50) | | Archivo fuente — ej. `CO2_951_121` |

**Volumen:** ~188,240 filas
`260 corridas × 2 pozos × 362 timesteps`

---

## Variables registradas

| Variable | Tabla | Tipo | Unidad | Timesteps |
|---|---|---|---|---|
| AFI (FriccionAngle) | Detalle | Estática por corrida | deg | 1 |
| COHESION | Detalle | Estática por corrida | psi | 1 |
| SF (SafetyFactor) | Detalle | Dinámica | adim | 362 |
| VD (TopVerticalDisplacement) | Detalle | Dinámica | ft | 362 |
| POROSITY | PropiedadesMalla | Estática del reservorio | frac | 1 |
| PERMEABILITY | PropiedadesMalla | Estática del reservorio | md | 1 |

---

## Escala total

| Tabla | Filas | Tipo de dato voluminoso |
|---|---|---|
| tblModelo | 1 | — |
| tblCorrida | 260 | — |
| tblPropiedadesMalla | 194 | LONGBLOB × 194 |
| tblResultadoVariables_Header | ~188,760 | — |
| tblResultadoVariables_Detalle | ~18,300,000 | LONGBLOB × 18.3 M |
| tblTasaInyeccion | ~188,240 | — |
| **Total** | **~18.7 M** | **~183 GB** |

---

## Índices

| Índice | Tabla | Columnas | Propósito |
|---|---|---|---|
| IX_tblRVD_Header_K | Detalle | (ID_Header, K) | Leer todas las capas de un header |
| IX_tblRVH_Corrida_Variable | Header | (ID_Corrida, Variable) | Filtrar por corrida y variable |
| IX_tblRVH_Corrida_Variable_TS | Header | (ID_Corrida, Variable, TimeStep) | Snapshot en un timestep específico |
| IX_tblCorrida_Modelo_Tipo | Corrida | (Modelo, TipoMuestreo) | Filtrar por tipo de muestreo |
| IX_tblTI_Corrida_Pozo | TasaInyeccion | (ID_Corrida, NombrePozo) | Serie temporal completa por pozo |
| IX_tblPM_Modelo_Variable | PropiedadesMalla | (Modelo, Variable) | Cargar grilla estática |

---

## Scripts

| Archivo | Descripción |
|---|---|
| `sql/ddl_mysql.sql` | DDL completo — crea la BD y todas las tablas |
| `sql/ddl_sqlserver.sql` | DDL equivalente para SQL Server |
| `scripts/etl_mysql.py` | ETL que carga `data/processed/` → MySQL |

### Ejecutar carga completa
```bash
python scripts/etl_mysql.py --password TUPASS

# Solo prueba (2 corridas)
python scripts/etl_mysql.py --password TUPASS --max-simulations 2

# Primera vez (crea la BD y tablas)
python scripts/etl_mysql.py --password TUPASS --apply-schema
```
