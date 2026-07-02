-- =============================================================================
-- DDL  MySQL 8.0  --  Simulaciones Geomecánicas CO2 (CCS)
-- =============================================================================
-- Equivalencias respecto al DDL SQL Server:
--   IDENTITY(1,1)    →  AUTO_INCREMENT
--   NVARCHAR(n)      →  VARCHAR(n)  (utf8mb4)
--   VARBINARY(MAX)   →  LONGBLOB    (hasta 4 GB)
--   SCOPE_IDENTITY() →  LAST_INSERT_ID()
-- =============================================================================

CREATE DATABASE IF NOT EXISTS SimulacionesCO2
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE SimulacionesCO2;


-- =============================================================================
-- 1. tblModelo
-- =============================================================================
CREATE TABLE IF NOT EXISTS tblModelo (
    Modelo       VARCHAR(50)   NOT NULL,
    DimK         INT           NOT NULL,
    DimJ         INT           NOT NULL,
    DimI         INT           NOT NULL,
    Descripcion  VARCHAR(200)  NULL,

    CONSTRAINT PK_tblModelo PRIMARY KEY (Modelo)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- =============================================================================
-- 2. tblCorrida
-- =============================================================================
CREATE TABLE IF NOT EXISTS tblCorrida (
    ID_Corrida     INT          NOT NULL AUTO_INCREMENT,
    Modelo         VARCHAR(50)  NOT NULL,
    NombreCorrida  VARCHAR(100) NOT NULL,
    CodigoCorrida  INT          NOT NULL,
    TipoMuestreo   VARCHAR(50)  NOT NULL,
    Particion      VARCHAR(10)  NOT NULL,
    FechaModelo    DATE         NULL,
    FechaCarga     DATETIME     NULL,

    CONSTRAINT PK_tblCorrida       PRIMARY KEY (ID_Corrida),
    CONSTRAINT FK_tblCorrida_Modelo
        FOREIGN KEY (Modelo) REFERENCES tblModelo (Modelo),
    CONSTRAINT CK_tblCorrida_Particion
        CHECK (Particion IN ('train', 'test')),
    CONSTRAINT UQ_tblCorrida_Nombre
        UNIQUE (Modelo, NombreCorrida)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- =============================================================================
-- 3. tblPropiedadesMalla
--    Propiedades estáticas del reservorio (POROSITY, PERMEABILITY).
--    Filas: 2 × 97 = 194  |  Blob: ~10 KB cada una
-- =============================================================================
CREATE TABLE IF NOT EXISTS tblPropiedadesMalla (
    ID_Propiedad  INT         NOT NULL AUTO_INCREMENT,
    Modelo        VARCHAR(50) NOT NULL,
    Variable      VARCHAR(50) NOT NULL,
    K             INT         NOT NULL,
    CapaDatos     LONGBLOB    NOT NULL,

    CONSTRAINT PK_tblPropiedadesMalla  PRIMARY KEY (ID_Propiedad),
    CONSTRAINT FK_tblPM_Modelo
        FOREIGN KEY (Modelo) REFERENCES tblModelo (Modelo),
    CONSTRAINT UQ_tblPropiedadesMalla  UNIQUE (Modelo, Variable, K)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- =============================================================================
-- 4. tblResultadoVariables_Header
--    Filas: ~188 760
-- =============================================================================
CREATE TABLE IF NOT EXISTS tblResultadoVariables_Header (
    ID_Header       INT           NOT NULL AUTO_INCREMENT,
    ID_Corrida      INT           NOT NULL,
    Variable        VARCHAR(50)   NOT NULL,
    TimeStep        INT           NOT NULL,
    FechaResultado  DATE          NULL,
    Unidad          VARCHAR(20)   NULL,
    ValorMin        DECIMAL(10,4) NULL,
    ValorMax        DECIMAL(10,4) NULL,

    CONSTRAINT PK_tblRVH      PRIMARY KEY (ID_Header),
    CONSTRAINT FK_tblRVH_Corrida
        FOREIGN KEY (ID_Corrida) REFERENCES tblCorrida (ID_Corrida),
    CONSTRAINT UQ_tblRVH      UNIQUE (ID_Corrida, Variable, TimeStep)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- =============================================================================
-- 5. tblResultadoVariables_Detalle
--    Filas: ~18.3 M  |  Storage: ~183 GB en blobs de 10 KB
-- =============================================================================
CREATE TABLE IF NOT EXISTS tblResultadoVariables_Detalle (
    ID_Detalle  INT      NOT NULL AUTO_INCREMENT,
    ID_Header   INT      NOT NULL,
    K           INT      NOT NULL,
    CapaDatos   LONGBLOB NOT NULL,

    CONSTRAINT PK_tblRVD     PRIMARY KEY (ID_Detalle),
    CONSTRAINT FK_tblRVD_Header
        FOREIGN KEY (ID_Header) REFERENCES tblResultadoVariables_Header (ID_Header),
    CONSTRAINT UQ_tblRVD     UNIQUE (ID_Header, K)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- =============================================================================
-- 6. tblTasaInyeccion
--    Filas: ~188 240
-- =============================================================================
CREATE TABLE IF NOT EXISTS tblTasaInyeccion (
    ID_TasaInyeccion  INT            NOT NULL AUTO_INCREMENT,
    ID_Corrida        INT            NOT NULL,
    NombrePozo        VARCHAR(50)    NOT NULL,
    TimeStep          INT            NOT NULL,
    TimeDay           DECIMAL(18,9)  NOT NULL,
    Fecha             DATETIME       NULL,
    Parametro         VARCHAR(100)   NOT NULL,
    Valor             DECIMAL(18,6)  NOT NULL,
    FileModelo        VARCHAR(50)    NULL,

    CONSTRAINT PK_tblTI     PRIMARY KEY (ID_TasaInyeccion),
    CONSTRAINT FK_tblTI_Corrida
        FOREIGN KEY (ID_Corrida) REFERENCES tblCorrida (ID_Corrida),
    CONSTRAINT UQ_tblTI     UNIQUE (ID_Corrida, NombrePozo, TimeStep)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- =============================================================================
-- ÍNDICES
-- =============================================================================

CREATE INDEX IX_tblRVD_Header_K
    ON tblResultadoVariables_Detalle (ID_Header, K);

CREATE INDEX IX_tblRVH_Corrida_Variable
    ON tblResultadoVariables_Header (ID_Corrida, Variable);

CREATE INDEX IX_tblRVH_Corrida_Variable_TS
    ON tblResultadoVariables_Header (ID_Corrida, Variable, TimeStep);

CREATE INDEX IX_tblCorrida_Modelo_Tipo
    ON tblCorrida (Modelo, TipoMuestreo);

CREATE INDEX IX_tblTI_Corrida_Pozo
    ON tblTasaInyeccion (ID_Corrida, NombrePozo);

CREATE INDEX IX_tblPM_Modelo_Variable
    ON tblPropiedadesMalla (Modelo, Variable);
