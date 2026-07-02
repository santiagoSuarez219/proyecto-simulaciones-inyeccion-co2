-- =============================================================================
-- DDL  SQL Server  --  Simulaciones Geomecánicas CO2 (CCS)
-- =============================================================================
-- Tablas        : 6
-- Volumen clave : tblResultadoVariables_Detalle  ~18.3 M filas / ~183 GB
-- Estrategia    : detalle espacial como blob VARBINARY(MAX) por capa K
--                 (grid DimJ × DimI en float32, orden row-major J→I)
-- =============================================================================


-- =============================================================================
-- 1. tblModelo
--    Catálogo de modelos geomecánicos y dimensiones de malla.
--    Necesario para deserializar CapaDatos correctamente.
-- =============================================================================
CREATE TABLE tblModelo (
    Modelo       NVARCHAR(50)   NOT NULL,
    DimK         INT            NOT NULL,   -- capas verticales  (ej. 97)
    DimJ         INT            NOT NULL,   -- filas de malla    (ej. 50)
    DimI         INT            NOT NULL,   -- columnas de malla (ej. 50)
    Descripcion  NVARCHAR(200)  NULL,

    CONSTRAINT PK_tblModelo PRIMARY KEY (Modelo)
);


-- =============================================================================
-- 2. tblCorrida
--    Tabla maestra de corridas / simulaciones.
-- =============================================================================
CREATE TABLE tblCorrida (
    ID_Corrida     INT            NOT NULL IDENTITY(1,1),
    Modelo         NVARCHAR(50)   NOT NULL,
    CodigoCorrida  INT            NOT NULL,   -- número de corrida (1–300)
    TipoMuestreo   NVARCHAR(50)   NOT NULL,   -- normal | LatinHyperCube |
                                              -- high_to_low | low_to_high |
                                              -- inverse_normal
    Particion      NVARCHAR(10)   NOT NULL,   -- train | test
    FechaModelo    DATE           NULL,
    FechaCarga     DATETIME       NULL,

    CONSTRAINT PK_tblCorrida
        PRIMARY KEY (ID_Corrida),
    CONSTRAINT FK_tblCorrida_Modelo
        FOREIGN KEY (Modelo) REFERENCES tblModelo (Modelo),
    CONSTRAINT CK_tblCorrida_Particion
        CHECK (Particion IN ('train', 'test')),
    CONSTRAINT UQ_tblCorrida_Codigo
        UNIQUE (Modelo, CodigoCorrida)
);


-- =============================================================================
-- 3. tblPropiedadesMalla
--    Propiedades estáticas del reservorio: iguales en todas las corridas.
--    Variables: POROSITY, PERMEABILITY — cargadas una sola vez por modelo.
--
--    Filas: 2 vars × 97 capas = 194
--    Blob : DimJ × DimI × 4 bytes float32  ≈ 10 KB por fila
-- =============================================================================
CREATE TABLE tblPropiedadesMalla (
    ID_Propiedad  INT            NOT NULL IDENTITY(1,1),
    Modelo        NVARCHAR(50)   NOT NULL,
    Variable      NVARCHAR(50)   NOT NULL,    -- POROSITY | PERMEABILITY
    K             INT            NOT NULL,    -- índice de capa (1..DimK)
    CapaDatos     VARBINARY(MAX) NOT NULL,    -- grid DimJ×DimI en float32

    CONSTRAINT PK_tblPropiedadesMalla
        PRIMARY KEY (ID_Propiedad),
    CONSTRAINT FK_tblPropiedadesMalla_Modelo
        FOREIGN KEY (Modelo) REFERENCES tblModelo (Modelo),
    CONSTRAINT UQ_tblPropiedadesMalla
        UNIQUE (Modelo, Variable, K)
);


-- =============================================================================
-- 4. tblResultadoVariables_Header
--    Encabezado de resultados por variable geomecánica, corrida y timestep.
--
--    Filas: 260 corridas × (2 vars estáticas × 1 ts
--                         + 2 vars dinámicas × 362 ts) ≈ 188 760
-- =============================================================================
CREATE TABLE tblResultadoVariables_Header (
    ID_Header       INT            NOT NULL IDENTITY(1,1),
    ID_Corrida      INT            NOT NULL,
    Variable        NVARCHAR(50)   NOT NULL,   -- AFI | COHESION | SF | VD
    TimeStep        INT            NOT NULL,   -- índice temporal (0..361)
    FechaResultado  DATE           NULL,
    Unidad          NVARCHAR(20)   NULL,       -- psi | adim | ft | ...
    ValorMin        DECIMAL(10,4)  NULL,       -- derivable del blob
    ValorMax        DECIMAL(10,4)  NULL,       -- derivable del blob

    CONSTRAINT PK_tblResultadoVariables_Header
        PRIMARY KEY (ID_Header),
    CONSTRAINT FK_tblRVH_Corrida
        FOREIGN KEY (ID_Corrida) REFERENCES tblCorrida (ID_Corrida),
    CONSTRAINT UQ_tblRVH
        UNIQUE (ID_Corrida, Variable, TimeStep)
);


-- =============================================================================
-- 5. tblResultadoVariables_Detalle
--    Detalle espacial por capa — grid completo almacenado como blob float32.
--
--    Filas   : ~18.3 M  (vs ~~45 900 M~~ con row-per-cell)
--    Storage : ~183 GB en blobs de 10 KB
--    PK      : INT es suficiente (máx ~2 100 M)
-- =============================================================================
CREATE TABLE tblResultadoVariables_Detalle (
    ID_Detalle  INT            NOT NULL IDENTITY(1,1),
    ID_Header   INT            NOT NULL,
    K           INT            NOT NULL,    -- índice de capa (1..DimK)
    CapaDatos   VARBINARY(MAX) NOT NULL,    -- grid DimJ×DimI en float32

    CONSTRAINT PK_tblResultadoVariables_Detalle
        PRIMARY KEY (ID_Detalle),
    CONSTRAINT FK_tblRVD_Header
        FOREIGN KEY (ID_Header)
        REFERENCES tblResultadoVariables_Header (ID_Header),
    CONSTRAINT UQ_tblRVD
        UNIQUE (ID_Header, K)
);


-- =============================================================================
-- 6. tblTasaInyeccion
--    Series temporales de tasa de inyección por pozo y corrida.
--
--    Filas: 260 corridas × 2 pozos × 362 timesteps ≈ 188 240
-- =============================================================================
CREATE TABLE tblTasaInyeccion (
    ID_TasaInyeccion  INT            NOT NULL IDENTITY(1,1),
    ID_Corrida        INT            NOT NULL,
    NombrePozo        NVARCHAR(50)   NOT NULL,   -- TENE-1 | TENE-2
    TimeStep          INT            NOT NULL,   -- índice temporal (0..361)
    TimeDay           DECIMAL(18,9)  NOT NULL,   -- días desde inicio simulación
    Fecha             DATETIME       NULL,
    Parametro         NVARCHAR(100)  NOT NULL,   -- Gas Rate SC - Monthly (ft3/day)
    Valor             DECIMAL(18,6)  NOT NULL,
    FileModelo        NVARCHAR(50)   NULL,        -- ej. CO2_951_121

    CONSTRAINT PK_tblTasaInyeccion
        PRIMARY KEY (ID_TasaInyeccion),
    CONSTRAINT FK_tblTI_Corrida
        FOREIGN KEY (ID_Corrida) REFERENCES tblCorrida (ID_Corrida),
    CONSTRAINT UQ_tblTI
        UNIQUE (ID_Corrida, NombrePozo, TimeStep)
);


-- =============================================================================
-- ÍNDICES
-- =============================================================================

-- Detalle: patrón principal — leer todas las capas de un header
CREATE INDEX IX_tblRVD_Header_K
    ON tblResultadoVariables_Detalle (ID_Header, K);

-- Header: filtrar por corrida + variable (ej. todas las capas SF de corrida 5)
CREATE INDEX IX_tblRVH_Corrida_Variable
    ON tblResultadoVariables_Header (ID_Corrida, Variable);

-- Header: filtrar por corrida + variable + timestep (lectura de un snapshot)
CREATE INDEX IX_tblRVH_Corrida_Variable_TS
    ON tblResultadoVariables_Header (ID_Corrida, Variable, TimeStep);

-- Corrida: filtrar por modelo y tipo de muestreo (ej. todos los LatinHyperCube)
CREATE INDEX IX_tblCorrida_Modelo_TipoMuestreo
    ON tblCorrida (Modelo, TipoMuestreo);

-- Inyección: acceso por corrida y pozo (serie temporal completa)
CREATE INDEX IX_tblTI_Corrida_Pozo
    ON tblTasaInyeccion (ID_Corrida, NombrePozo);

-- Propiedades: acceso por modelo y variable (cargar grilla estática)
CREATE INDEX IX_tblPM_Modelo_Variable
    ON tblPropiedadesMalla (Modelo, Variable);
