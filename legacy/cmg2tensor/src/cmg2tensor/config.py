"""
Shared constants for cmg2tensor.

Import from here instead of redefining in each module.
"""
from __future__ import annotations

from pathlib import Path

# Injection / Excel processing
DAYS_PER_MONTH: float = 365.25 / 12.0
DEFAULT_INJECTION_PARAMETER: str = "Gas Rate SC - Monthly (ft3/day)"
DEFAULT_INCLUDE_INJECTION_NAMES: tuple[str, ...] = ("TENE-1", "TENE-2")

# Default report / output paths (relative to project root)
DEFAULT_SPLIT_CSV = Path("reports") / "train_test_split_80_20.csv"
DEFAULT_BATCH_REPORT_PATH = Path("reports") / "batch_simulations_report.json"
DEFAULT_GLOBAL_NORMALIZATION_DIR = Path("reports") / "global_normalization"
DEFAULT_GLOBAL_STATS_FILE = DEFAULT_GLOBAL_NORMALIZATION_DIR / "train.json"
