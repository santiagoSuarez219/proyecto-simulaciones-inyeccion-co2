#!/usr/bin/env python
import argparse
from pathlib import Path

from fno_co2.config import Config, load_config_from_yaml
from fno_co2.training.loop import main


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
    p.add_argument(
        "--lr-scheduler", default=None,
        help="Scheduler de LR: 'cosine' o 'none' para desactivarlo (default: cosine)",
    )
    p.add_argument("--lr-min", type=float, default=None, help="LR minimo del scheduler coseno (eta_min)")
    p.add_argument("--batch-size", type=int, default=None, help="Tamaño del batch")
    p.add_argument("--num-workers", type=int, default=None, help="Número de workers para DataLoader")
    p.add_argument("--prefetch-factor", type=int, default=None, help="Prefetch factor para DataLoader")
    p.add_argument("--persistent-workers", type=str_to_bool, default=None, help="Usar persistent workers")
    p.add_argument("--progress-interval", type=int, default=None, help="Intervalo de progreso en batches")
    p.add_argument("--device", default=None, help="Dispositivo: cuda, cpu, auto, gpu")
    p.add_argument("--seed", type=int, default=None, help="Semilla aleatoria para reproducibilidad")
    p.add_argument(
        "--config", default=None,
        help="Ruta a un YAML de Config (ver configs/experiments/); los flags CLI explícitos "
             "tienen prioridad sobre los valores del archivo",
    )
    p.add_argument(
        "--experiment-name", default=None,
        help="Nombre del experimento (default en Config: 'baseline'). Si se pasa explícitamente "
             "y no se pasa --output-dir, deriva outputs/<experiment_name>/seed_<seed>/",
    )
    p.add_argument(
        "--model-variant", default=None,
        help="Variante de arquitectura a despachar vía build_model (default: 'fno_baseline')",
    )
    p.add_argument("--use-amp", type=str_to_bool, default=None, help="Mixed precision (AMP) — solo activo en CUDA")
    p.add_argument(
        "--deterministic", type=str_to_bool, default=None,
        help="Prioriza reproducibilidad sobre rendimiento en CUDA (cudnn.deterministic)",
    )
    p.add_argument("--pause-hour", type=int, default=None, help="Hora para pausar entrenamiento (0-23)")
    p.add_argument("--auto-resume", type=str_to_bool, default=None, help="Reanudar desde checkpoint automáticamente")
    p.add_argument("--early-stopping-patience", type=int, default=None, help="Paciencia para early stopping")
    p.add_argument("--early-stopping-min-delta", type=float, default=None, help="Delta mínimo para early stopping")
    p.add_argument("--uncertainty-passes", type=int, default=None, help="Pasadas de MC Dropout para incertidumbre")
    p.add_argument(
        "--uncertainty-eval-interval", type=int, default=None,
        help="Cada cuantas épocas se computa la incertidumbre MC-Dropout completa (cara); "
             "<=0 => solo en la época final. No afecta val_loss/selección de best.pt",
    )
    p.add_argument("--dropout-p", type=float, default=None, help="Probabilidad de Dropout2d en ResBlock (MC Dropout)")
    p.add_argument(
        "--use-group-norm", type=str_to_bool, default=None,
        help="EXPERIMENTAL (M3): GroupNorm en ResBlock — cambia la arquitectura, invalida checkpoints previos",
    )
    p.add_argument("--save-epoch-pngs", type=str_to_bool, default=None, help="Guardar visualizaciones por época")
    return p


def resolve_config(args: argparse.Namespace) -> Config:
    """Ensambla el `Config` efectivo de una corrida: YAML (`--config`, opcional) como base,
    luego overrides de flags CLI explícitos (siempre con prioridad sobre el YAML), y por
    último la derivación de `output_dir` de la Fase 1 del spec-001 cuando corresponde."""
    cfg = load_config_from_yaml(args.config) if args.config else Config()

    experiment_name_explicit = args.experiment_name is not None
    output_dir_explicit = args.output_dir is not None

    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        setattr(cfg, key, value)

    if cfg.lr_scheduler is not None and cfg.lr_scheduler.lower() == "none":
        cfg.lr_scheduler = None

    # Fase 1: un experiment_name explícito deriva el output_dir por seed, para que dos
    # corridas del mismo experimento con seeds distintas nunca colisionen. Si el usuario
    # también pasó --output-dir explícito, ese valor manda (no se pisa).
    if experiment_name_explicit and not output_dir_explicit:
        cfg.output_dir = f"outputs/{cfg.experiment_name}/seed_{cfg.seed}"

    return cfg


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    cfg = resolve_config(args)
    main(cfg)
