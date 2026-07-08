from __future__ import annotations

from cmg2tensor.utils.raw_standardization import (
    build_check_formats_arg_parser,
    write_raw_structure_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_check_formats_arg_parser()
    args = parser.parse_args(argv)
    return write_raw_structure_report(raw_root=args.raw_root, out_file=args.out_file, include_v2=args.include_v2)


if __name__ == "__main__":
    raise SystemExit(main())
