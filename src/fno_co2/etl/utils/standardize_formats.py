from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parents[3]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from fno_co2.etl.utils.raw_standardization import build_standardize_arg_parser, standardize_raw_names


def main(argv: list[str] | None = None) -> int:
    parser = build_standardize_arg_parser()
    args = parser.parse_args(argv)
    return standardize_raw_names(
        raw_root=args.raw_root,
        apply_changes=args.apply,
        log_file=args.log_file,
        include_v2=args.include_v2,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
