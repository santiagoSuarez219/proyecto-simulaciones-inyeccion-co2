#!/usr/bin/env python
import argparse
from pathlib import Path

from modelo_itm.config import Config
from modelo_itm.training.loop import main


def str_to_bool(value):
    return str(value).lower() in {"1", "true", "yes", "y", "si", "sí"}


def build_parser():
    p = argparse.ArgumentParser(description="Entrenamiento del modelo FNO para predicción espacio-temporal")
    p.add_argument("--data-root", default=None, help="Raíz del directorio de datos procesados")
    p.add_argument("--train-dir", default=None, help="Directorio del conjunto de entrenamiento")
    p.add_argument("--val-dir", default=None, help="Directorio del conjunto de validación")
    p.add_argument("--output-dir", default=None, help="Directorio de salida para checkpoints y logs")
    p.add_argument("--checkpoint-dir", default=None, help="Directorio de checkpoints (default: output-dir/checkpoints)")
    p.add_argument("--epochs", type=int, default=None, help="Número de épocas de entrenamiento")
    p.add_argument("--overfit-sample-idx", type=int, default=None, help="Índice de muestra para overfitting test")
    p.add_argument("--lr", type=float, default=None, help="Learning rate inicial")
    p.add_argument("--batch-size", type=int, default=None, help="Tamaño del batch")
    p.add_argument("--num-workers", type=int, default=None, help="Número de workers para DataLoader")
    p.add_argument("--prefetch-factor", type=int, default=None, help="Prefetch factor para DataLoader")
    p.add_argument("--persistent-workers", type=str_to_bool, default=None, help="Usar persistent workers")
    p.add_argument("--progress-interval", type=int, default=None, help="Intervalo de progreso en batches")
    p.add_argument("--device", default=None, help="Dispositivo: cuda, cpu, auto, gpu")
    p.add_argument("--pause-hour", type=int, default=None, help="Hora para pausar entrenamiento (0-23)")
    p.add_argument("--auto-resume", type=str_to_bool, default=None, help="Reanudar desde checkpoint automáticamente")
    p.add_argument("--early-stopping-patience", type=int, default=None, help="Paciencia para early stopping")
    p.add_argument("--early-stopping-min-delta", type=float, default=None, help="Delta mínimo para early stopping")
    p.add_argument("--uncertainty-passes", type=int, default=None, help="Pasadas de MC Dropout para incertidumbre")
    p.add_argument("--save-epoch-pngs", type=str_to_bool, default=None, help="Guardar visualizaciones por época")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    cfg = Config()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

    main(cfg)
