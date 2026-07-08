from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np

TIME_BLOCK_PATTERN = re.compile(
    r"\*\*\s*TIME\s*=\s*([-+]?(?:\d*\.\d+|\d+))(?:\s*day)?\s+"
    r"([0-9]{4}-(?:[A-Za-z]{3}|\d{2})-[0-9]{2})\s*"
    r"([\s\S]*?)(?=\*\*\s*TIME\s*=|\Z)",
    re.MULTILINE,
)
KJ_BLOCK_PATTERN = re.compile(
    r"\*\*\s*K\s*=\s*(\d+)\s*,\s*J\s*=\s*(\d+)\s*([\s\S]*?)(?=\*\*\s*K\s*=|\Z)",
    re.MULTILINE,
)
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

# Fast line-level regexes used by the streaming parser (_parse_txt_with_times).
# These are separate from the block-level patterns above, which operate on large
# multi-line strings loaded into memory and are kept for parse_cmg_text / parse_cmg_file.
_TIME_HEADER_FAST_RE = re.compile(
    r"\*\*\s*TIME\s*=\s*([-+]?(?:\d*\.\d+|\d+))(?:\s*day)?\s+"
    r"([0-9]{4}-(?:[A-Za-z]{3}|\d{2})-[0-9]{2})"
)
_KJ_HEADER_RE = re.compile(r"\*\*\s*K\s*=\s*(\d+)\s*,\s*J\s*=\s*(\d+)")


@dataclass(frozen=True)
class GridShape:
    nz: int
    nj: int
    ni: int


@dataclass(frozen=True)
class ParseMessage:
    level: str
    message: str


@dataclass(frozen=True)
class ParseResult:
    cube_4d: np.ndarray
    time_days: list[int]
    time_dates: list[str]
    messages: list[ParseMessage]


def read_txt(filepath: str | Path, encoding: str = "utf-8") -> str:
    """Read an input text file as a single string."""
    return Path(filepath).read_text(encoding=encoding)


def parse_cmg_text(
    text: str,
    grid: GridShape,
    *,
    strict: bool = False,
) -> ParseResult:
    """
    Parse CMG text and return a 4D array with shape (NT, NZ, NJ, NI).

    If no TIME blocks are found, the text is interpreted as a single snapshot.
    """
    messages: list[ParseMessage] = []
    time_matches = list(TIME_BLOCK_PATTERN.finditer(text))

    if not time_matches:
        cube, snapshot_messages = _parse_single_snapshot(
            text,
            grid,
            context="SINGLE_SNAPSHOT",
            strict=strict,
        )
        messages.extend(snapshot_messages)
        return ParseResult(
            cube_4d=np.expand_dims(cube, axis=0),
            time_days=[0],
            time_dates=["NA"],
            messages=messages,
        )

    cubes: list[np.ndarray] = []
    time_days: list[int] = []
    time_dates: list[str] = []
    for block in time_matches:
        day_str, date_str, time_content = block.groups()
        day = _parse_time_day(day_str)
        cube, snapshot_messages = _parse_single_snapshot(
            time_content,
            grid,
            context=f"TIME={day} ({date_str})",
            strict=strict,
        )
        cubes.append(cube)
        time_days.append(day)
        time_dates.append(date_str)
        messages.extend(snapshot_messages)

    return ParseResult(
        cube_4d=np.stack(cubes, axis=0),
        time_days=time_days,
        time_dates=time_dates,
        messages=messages,
    )


def parse_cmg_file(
    filepath: str | Path,
    grid: GridShape,
    *,
    encoding: str = "utf-8",
    strict: bool = False,
) -> ParseResult:
    """Load and parse a CMG text file."""
    text = read_txt(filepath, encoding=encoding)
    return parse_cmg_text(text, grid, strict=strict)


def summarize_parse_messages(messages: Iterable[ParseMessage]) -> dict[str, int]:
    """Count parser messages by level."""
    counts: dict[str, int] = {}
    for msg in messages:
        counts[msg.level] = counts.get(msg.level, 0) + 1
    return counts


def _parse_single_snapshot(
    text: str,
    grid: GridShape,
    *,
    context: str,
    strict: bool,
) -> tuple[np.ndarray, list[ParseMessage]]:
    cube = np.full((grid.nz, grid.nj, grid.ni), np.nan, dtype=np.float32)
    messages: list[ParseMessage] = []

    for match in KJ_BLOCK_PATTERN.finditer(text):
        k_str, j_str, values_block = match.groups()
        k = int(k_str)
        j = int(j_str)
        values = _extract_values(values_block)

        if values.size != grid.ni:
            detail = (
                f"{context} K={k}, J={j}: {values.size} values found, expected {grid.ni}. "
                "Values were adjusted with trim/pad."
            )
            _add_message(messages, strict, "warning", detail)
            values = _fit_expected_size(values, expected_size=grid.ni)

        if 1 <= k <= grid.nz and 1 <= j <= grid.nj:
            cube[k - 1, j - 1, :] = values
        else:
            detail = f"{context} K={k}, J={j}: out-of-range index. Block ignored."
            _add_message(messages, strict, "error", detail)

    return cube, messages


def _extract_values(block: str) -> np.ndarray:
    # Fast path: np.fromstring parses in C (~3x faster for plain whitespace-delimited numbers).
    stripped = block.strip()
    if stripped:
        try:
            result = np.fromstring(stripped, dtype=np.float32, sep=" ")
            if result.size > 0:
                return result
        except ValueError:
            pass
    # Fallback: regex handles scientific notation and mixed separators.
    tokens = NUMBER_PATTERN.findall(block)
    return np.array(tokens, dtype=np.float32) if tokens else np.empty(0, dtype=np.float32)


def _fit_expected_size(values: np.ndarray, expected_size: int) -> np.ndarray:
    if values.size > expected_size:
        return values[:expected_size]
    if values.size < expected_size:
        return np.pad(values, (0, expected_size - values.size), constant_values=np.nan)
    return values


def _add_message(
    messages: list[ParseMessage],
    strict: bool,
    level: str,
    detail: str,
) -> None:
    if strict and level == "error":
        raise ValueError(detail)
    messages.append(ParseMessage(level=level, message=detail))


def parse_txt(
    path: str | Path,
    NZ: int,
    NJ: int,
    NI: int,
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Strict parser for CMG files with K/J blocks and optional TIME headers.

    Returns:
        np.ndarray with shape (T, NZ, NJ, NI)
    """
    tensor_4d, _time_ids = _parse_txt_with_times(path, NZ, NJ, NI, dtype=dtype)
    return tensor_4d


def parse_txt_with_times(
    path: str | Path,
    NZ: int,
    NJ: int,
    NI: int,
    *,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, list[int]]:
    """
    Strict parser for CMG files with K/J blocks and optional TIME headers.

    Returns:
        (tensor_4d, time_ids) where tensor_4d has shape (T, NZ, NJ, NI).
    """
    return _parse_txt_with_times(path, NZ, NJ, NI, dtype=dtype)


def build_layer_cubes(
    sf_path: str | Path,
    vd_path: str | Path,
    NZ: int,
    NJ: int,
    NI: int,
    *,
    dtype: np.dtype = np.float32,
    torch_output: bool = False,
    return_times: bool = False,
):
    """
    Build a collection of NZ cubes, one per layer K.

    Output:
        cubes[k] has shape (2, T, NJ, NI), where variable order is [SF, VD].
    """
    sf_4d, sf_times = _parse_txt_with_times(sf_path, NZ, NJ, NI, dtype=dtype)
    vd_4d, vd_times = _parse_txt_with_times(vd_path, NZ, NJ, NI, dtype=dtype)

    if sf_times != vd_times:
        raise ValueError(
            f"TIME mismatch between files.\nSF times: {sf_times}\nVD times: {vd_times}"
        )

    # Shape: (V, T, NZ, NJ, NI) with fixed variable order [SF, VD]
    stacked = np.stack([sf_4d, vd_4d], axis=0)
    cubes = [stacked[:, :, k, :, :] for k in range(NZ)]

    if torch_output:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch is required when torch_output=True. Install `torch` first."
            ) from exc
        cubes = [torch.tensor(cube, dtype=torch.float32) for cube in cubes]

    if return_times:
        return cubes, sf_times
    return cubes


def build_single_variable_layer_cubes(
    path: str | Path,
    NZ: int,
    NJ: int,
    NI: int,
    *,
    dtype: np.dtype = np.float32,
    torch_output: bool = False,
    return_times: bool = False,
    strict: bool = True,
):
    """
    Build NZ cubes for one variable from a single CMG file.

    Output:
        cubes[k] has shape (1, T, NJ, NI), with one variable channel.
    """
    tensor_4d, time_ids = _parse_txt_with_times(path, NZ, NJ, NI, dtype=dtype, strict=strict)

    # Shape: (V=1, T, NZ, NJ, NI)
    stacked = np.expand_dims(tensor_4d, axis=0)
    cubes = [stacked[:, :, k, :, :] for k in range(NZ)]

    if torch_output:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch is required when torch_output=True. Install `torch` first."
            ) from exc
        cubes = [torch.tensor(cube, dtype=torch.float32) for cube in cubes]

    if return_times:
        return cubes, time_ids
    return cubes


def _count_time_blocks_streaming(path: Path, encoding: str = "utf-8") -> int:
    """Fast single-pass count of TIME header lines. Nearly zero memory usage."""
    count = 0
    with path.open("r", encoding=encoding, errors="ignore") as fh:
        for line in fh:
            if "**" in line and "TIME" in line and "=" in line:
                if _TIME_HEADER_FAST_RE.search(line):
                    count += 1
    return count


def _parse_txt_with_times(
    path: str | Path,
    NZ: int,
    NJ: int,
    NI: int,
    *,
    dtype: np.dtype,
    strict: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """
    Streaming incremental CMG parser. Does NOT load the full file into RAM.

    Algorithm:
    - Pass 1: count TIME header lines to know T (enables pre-allocation).
    - Pre-allocate tensor_4d = np.empty((T, NZ, NJ, NI), dtype=dtype) filled with NaN.
      This eliminates the np.stack() copy at the end (Fix O6) and uses the target
      dtype directly (Fix O4).
    - Pass 2: read line by line, maintaining a tiny state machine:
        * On "** TIME = ..." → flush pending KJ block, validate completed time block,
          increment time index.
        * On "** K=k, J=j" → flush pending KJ block, set new (k, j) state.
        * On value lines → append to value_buffer (O(NI) strings at most).
        * _flush_kj_block() → joins buffer, calls _extract_values(), writes directly
          into tensor_4d[t_idx, k-1, j-1, :].

    Memory per worker: O(NZ * NJ * NI * sizeof(dtype)) + O(NI) string buffer.
    For NZ=20, NJ=100, NI=100, T=15, float32 ≈ 120 MB (vs 3 GB+ with read_text).

    Static snapshot fallback (no TIME blocks): delegates to parse_cmg_file which
    uses read_txt — acceptable because static property files are much smaller.
    """
    path = Path(path)
    expected_blocks = NZ * NJ

    # --- Pass 1: count TIME blocks (fast, no content accumulation) ---
    n_time = _count_time_blocks_streaming(path)

    if n_time == 0:
        # Static snapshot without TIME blocks — these are small files, full load is OK.
        text = read_txt(path)
        grid = GridShape(nz=NZ, nj=NJ, ni=NI)
        result = parse_cmg_text(text, grid, strict=strict)
        arr = result.cube_4d.astype(dtype, copy=False)
        return arr, result.time_days

    # --- Pre-allocate output tensor (Fix O6: no np.stack copy; Fix O4: direct dtype) ---
    tensor_4d = np.empty((n_time, NZ, NJ, NI), dtype=dtype)
    tensor_4d.fill(np.nan)
    time_ids: list[int] = []

    # --- State machine variables ---
    t_idx: int = -1
    current_day: int = 0
    current_date: str = "NA"
    current_k: int | None = None
    current_j: int | None = None
    seen: set[tuple[int, int]] = set()
    value_buffer: list[str] = []

    def _flush_kj_block() -> None:
        nonlocal current_k, current_j, value_buffer
        if current_k is None or current_j is None:
            value_buffer = []
            return
        k, j = current_k, current_j
        current_k = None
        current_j = None
        if not (1 <= k <= NZ and 1 <= j <= NJ):
            if strict:
                raise ValueError(
                    f"{path} TIME={current_day} ({current_date}): K={k}, J={j} out of range "
                    f"(K:1..{NZ}, J:1..{NJ})"
                )
            value_buffer = []
            return
        key = (k - 1, j - 1)
        if key in seen:
            if strict:
                raise ValueError(
                    f"{path} TIME={current_day} ({current_date}): duplicated block K={k}, J={j}"
                )
        else:
            seen.add(key)
        raw = " ".join(value_buffer)
        value_buffer = []
        values = _extract_values(raw)
        if values.size != NI:
            if strict:
                raise ValueError(
                    f"{path} TIME={current_day} ({current_date}) K={k}, J={j}: "
                    f"{values.size} values found, expected {NI}"
                )
            values = _fit_expected_size(values, expected_size=NI)
        if t_idx >= 0:
            tensor_4d[t_idx, k - 1, j - 1, :] = values

    def _flush_time_block() -> None:
        if t_idx < 0:
            return
        if len(seen) != expected_blocks:
            if strict:
                raise ValueError(
                    f"{path} TIME={current_day} ({current_date}): found {len(seen)} blocks K/J, "
                    f"expected {expected_blocks} (= NZ*NJ)"
                )
        if strict and np.isnan(tensor_4d[t_idx]).any():
            raise ValueError(
                f"{path} TIME={current_day} ({current_date}): missing values detected in cube"
            )

    # --- Pass 2: streaming parse ---
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            # Detect TIME header line
            if "**" in line and "TIME" in line and "=" in line:
                tm = _TIME_HEADER_FAST_RE.search(line)
                if tm:
                    _flush_kj_block()
                    _flush_time_block()
                    current_day = _parse_time_day(tm.group(1), path)
                    current_date = tm.group(2)
                    t_idx += 1
                    seen = set()
                    time_ids.append(current_day)
                    continue

            # Detect KJ header line
            if "**" in line and "K" in line and "J" in line and "=" in line:
                kj = _KJ_HEADER_RE.search(line)
                if kj:
                    _flush_kj_block()
                    current_k = int(kj.group(1))
                    current_j = int(kj.group(2))
                    # Capture any values on the same line as the K/J header
                    after_header = line[kj.end():]
                    if after_header.strip():
                        value_buffer = [after_header.strip()]
                    continue

            # Accumulate value lines for the current KJ block
            if current_k is not None:
                stripped = line.strip()
                if stripped:
                    value_buffer.append(stripped)

    # Flush final KJ block and validate final time block
    _flush_kj_block()
    _flush_time_block()

    # Handle count mismatch between pass1 and pass2 (e.g. partial file)
    actual_t = t_idx + 1
    if actual_t != n_time:
        tensor_4d = tensor_4d[:actual_t]

    return tensor_4d, time_ids


def _parse_time_day(day_str: str, path: str | Path | None = None) -> int:
    day_float = float(day_str)
    day_int = int(round(day_float))
    if not np.isclose(day_float, day_int):
        where = f" in file {path}" if path is not None else ""
        raise ValueError(f"Non-integer TIME value '{day_str}'{where}.")
    return day_int
