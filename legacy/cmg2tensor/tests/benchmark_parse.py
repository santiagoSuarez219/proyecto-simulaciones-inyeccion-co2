"""
Performance benchmark for the CMG parser.

Measures:
  - File size (GB)
  - Time to run _count_time_blocks_streaming() (Pass 1)
  - Time to run _parse_txt_with_times() in full (Pass 1 + Pass 2)
  - Peak RAM usage during parse (via tracemalloc)
  - Time for scan_file_for_stats() (Phase 1 stats scan)

Usage (from repo root):
    python tests/benchmark_parse.py --path data/raw/001_normal/SF.txt --nz 20 --nj 100 --ni 100

Optional: compare against a second path (e.g., VD.txt) for Phase 2 simulation.
"""
from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path


def _fmt_size(n_bytes: int) -> str:
    if n_bytes >= 1e9:
        return f"{n_bytes / 1e9:.2f} GB"
    if n_bytes >= 1e6:
        return f"{n_bytes / 1e6:.2f} MB"
    return f"{n_bytes / 1e3:.2f} KB"


def benchmark(path: Path, nz: int, nj: int, ni: int) -> None:
    # Add src/ to path if running without install
    src = Path(__file__).parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from cmg2tensor.parse_txt import _count_time_blocks_streaming, _parse_txt_with_times
    from cmg2tensor.stats import scan_file_for_stats
    import numpy as np

    file_size = path.stat().st_size
    print(f"\n{'=' * 60}")
    print(f"File      : {path}")
    print(f"Size      : {_fmt_size(file_size)}")
    print(f"Grid      : NZ={nz}, NJ={nj}, NI={ni}")
    print(f"{'=' * 60}")

    # --- Pass 1 only: count TIME blocks ---
    t0 = time.perf_counter()
    n_time = _count_time_blocks_streaming(path)
    t1 = time.perf_counter()
    print(f"\n[Pass 1 - count TIME blocks]")
    print(f"  TIME blocks found : {n_time}")
    print(f"  Time              : {t1 - t0:.3f}s")

    # --- Full parse: Pass 1 + Pass 2 ---
    tracemalloc.start()
    t2 = time.perf_counter()
    tensor, time_ids = _parse_txt_with_times(path, nz, nj, ni, dtype=np.float32, strict=False)
    t3 = time.perf_counter()
    current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tensor_size = tensor.nbytes
    parse_time = t3 - t2
    throughput_mb_s = (file_size / 1e6) / parse_time if parse_time > 0 else float("inf")

    print(f"\n[Full parse (_parse_txt_with_times)]")
    print(f"  Shape             : {tensor.shape}")
    print(f"  Tensor size       : {_fmt_size(tensor_size)}")
    print(f"  Parse time        : {parse_time:.3f}s")
    print(f"  Throughput        : {throughput_mb_s:.1f} MB/s")
    print(f"  Peak RAM (traced) : {_fmt_size(peak_bytes)}")
    print(f"  RAM / file ratio  : {peak_bytes / file_size:.2f}x")

    del tensor

    # --- stats scan ---
    t4 = time.perf_counter()
    fr = scan_file_for_stats(path, "VAR", nz=nz, nj=nj, ni=ni)
    t5 = time.perf_counter()
    print(f"\n[Stats scan (scan_file_for_stats)]")
    print(f"  Source            : {fr.minmax_source}")
    print(f"  vmin / vmax       : {fr.vmin:.4f} / {fr.vmax:.4f}")
    print(f"  Scan time         : {t5 - t4:.3f}s")
    print(f"  RESULTS PROP lines: {fr.n_minmax_lines}")

    print(f"\n{'=' * 60}")
    print("Summary (targets with 8 workers, 20×100×100 grid, 15 timesteps):")
    print(f"  Phase 1 (stats scan) : ≤ 30 min total  ← scan_file_for_stats × 300")
    print(f"  Phase 2 (transform)  : ≤ 45 min total  ← full parse × 300")
    print(f"  RAM peak per worker  : ≤ 8 GB")
    print(f"\nActual for this file:")
    n_workers = 8
    phase1_est = (t5 - t4) * 300 / n_workers / 60
    phase2_est = parse_time * 300 / n_workers / 60
    print(f"  Phase 1 estimate     : {phase1_est:.1f} min (with {n_workers} workers)")
    print(f"  Phase 2 estimate     : {phase2_est:.1f} min (with {n_workers} workers)")
    print(f"  RAM per worker       : {_fmt_size(tensor_size)} (tensor only)")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cmg2tensor parser")
    parser.add_argument("--path", required=True, help="Path to CMG .txt file")
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--nj", type=int, default=100)
    parser.add_argument("--ni", type=int, default=100)
    args = parser.parse_args()
    benchmark(Path(args.path), args.nz, args.nj, args.ni)


if __name__ == "__main__":
    main()
