# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read and write Experiment objects as CSV.

Two file formats are supported:

**Wide format** (new default)
    Temperature and magnetic field are specified as header rows, one value per
    data column. Signal columns repeat for each (T, B) condition::

        temperature (K)    298.6  298.6  298.6  305    305    305
        magnetic field (T) 4.7    4.7    4.7    4.7    4.7    4.7
        assignment         shift  width  area   shift  width  area
        tBu1               11     30     9      11     30     9

**Legacy format**
    Temperature and magnetic field are encoded as ``# key value`` comment
    lines at the top of the file, one file per (T, B) condition.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from datetime import datetime
from typing import Iterable, List

from simpnmr.__version__ import __version__
from simpnmr.core.domain.exp import Experiment, Signal
from simpnmr.io.text.parse import find_first_group

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name aliases for tolerant matching
# ---------------------------------------------------------------------------
_SHIFT_NAMES = {"shift", "shifts", "ppm", "shift (ppm)", "shifts (ppm)"}
_WIDTH_NAMES = {"width", "widths", "width (hz)", "width(hz)", "lw", "lw (hz)"}
_AREA_NAMES = {"area", "areas", "area ()", "area()", "integral", "integrals"}
_R1_NAMES = {
    "r1", "r1 (hz)", "r1 (s^-1)", "r1(hz)", "r1(s^-1)",
    "1/t1", "1/t1 (hz)", "1/t1 (s^-1)",
}
_LG_NAMES = {"l/g", "l/g ()"}
_ISOTOPE_NAMES = {"isotope", "isotope ()"}


def _norm(s: str) -> str:
    return s.strip().lower()


def _match(name: str, aliases: set) -> bool:
    return _norm(name) in aliases


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _is_wide_format(file_name: str) -> bool:
    """Return True if the first non-comment, non-empty line starts with
    'temperature'."""
    with open(file_name, encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return bool(re.match(r"temperature", stripped, re.IGNORECASE))
    return False


# ---------------------------------------------------------------------------
# Legacy format reader
# ---------------------------------------------------------------------------

def read_exp_metadata(file_name: str) -> tuple[float, float] | None:
    """Try to read legacy metadata from ``#`` comment lines.

    Returns ``(temperature, magnetic_field)`` if both are found, else ``None``.
    """
    try:
        _pat_t = r"# *temperature (\d*\.*\d*)"
        temperature = float(
            find_first_group(file_name, _pat_t, re.IGNORECASE)
        )
        magnetic_field = float(
            find_first_group(
                file_name, r"# *magnetic_field (\d*\.*\d*)", re.IGNORECASE
            )
        )
        return temperature, magnetic_field
    except (IndexError, ValueError, TypeError):
        return None


def _load_legacy_experiments(file_name: str) -> list[Experiment]:
    """Load a single Experiment from a legacy (# header comment) CSV file."""
    import pandas as pd

    meta = read_exp_metadata(file_name)
    if meta is None:
        raise ValueError(
            f"{file_name}: not a wide-format file and no legacy "
            "# temperature / # magnetic_field header found."
        )
    temperature, magnetic_field = meta

    # Read CSV, skipping comment lines
    lines = []
    with open(file_name, encoding="utf-8-sig") as f:
        for line in f:
            if not line.startswith("#"):
                lines.append(line)
    df = pd.read_csv(io.StringIO("".join(lines)))
    df = df.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")

    # Identify columns
    norm_cols = {_norm(str(c)): c for c in df.columns}

    def _find(aliases):
        for a in aliases:
            if a in norm_cols:
                return norm_cols[a]
        return None

    shift_col = _find(_SHIFT_NAMES)
    width_col = _find(_WIDTH_NAMES)
    area_col = _find(_AREA_NAMES)
    r1_col = _find(_R1_NAMES)
    lg_col = _find(_LG_NAMES)
    asgn_col = _find({"assignment", "assignments"})

    for required, name in [(shift_col, "shift"), (width_col, "width"),
                           (area_col, "area"), (asgn_col, "assignment")]:
        if required is None:
            raise KeyError(f"{file_name}: required column '{name}' not found.")

    signals = []
    for _, row in df.iterrows():
        r1 = None
        if r1_col is not None:
            v = row[r1_col]
            if v == v:  # not NaN
                try:
                    r1 = float(v)
                except (ValueError, TypeError):
                    pass
        l_to_g = float(row[lg_col]) if lg_col is not None else 1.0
        signals.append(Signal(
            float(row[shift_col]),
            float(row[width_col]),
            float(row[area_col]),
            str(row[asgn_col]),
            l_to_g=l_to_g,
            r1=r1,
        ))

    return [Experiment(temperature, magnetic_field, signals)]


# ---------------------------------------------------------------------------
# Wide format reader
# ---------------------------------------------------------------------------

def _load_wide_experiments(file_name: str) -> list[Experiment]:
    """Load one or more Experiments from a wide-format CSV file."""

    # Read non-comment, non-empty lines
    raw_lines: list[str] = []
    with open(file_name, encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                raw_lines.append(line)

    if len(raw_lines) < 3:
        raise ValueError(
            f"{file_name}: wide format requires at least 3 header rows."
        )

    # Auto-detect separator from the header row (tab or comma)
    sep = "\t" if "\t" in raw_lines[0] else ","

    def _parse_row(line: str) -> list[str]:
        return next(csv.reader([line], delimiter=sep))

    temp_row = _parse_row(raw_lines[0])    # temperature (K), v, v, ...
    field_row = _parse_row(raw_lines[1])   # magnetic field (T), v, v, ...
    header_row = _parse_row(raw_lines[2])  # assignment, col, col, ...
    data_rows = [_parse_row(ln) for ln in raw_lines[3:] if ln.strip()]

    # Strip trailing empty cells from header rows (caused by trailing delimiters)
    while len(temp_row) > 1 and temp_row[-1].strip() == "":
        temp_row.pop()
    while len(field_row) > 1 and field_row[-1].strip() == "":
        field_row.pop()

    n_data_cols = len(temp_row) - 1  # columns after the label cell

    temperatures = [float(temp_row[i + 1]) for i in range(n_data_cols)]
    fields = [float(field_row[i + 1]) for i in range(n_data_cols)]
    col_names = [header_row[i + 1] for i in range(n_data_cols)]

    # Unique (T, B) conditions in file order
    conditions: list[tuple[float, float]] = list(
        dict.fromkeys(zip(temperatures, fields))
    )

    # For each condition, record which column indices belong to it
    cond_col_map: dict[tuple, list[tuple[int, str]]] = {
        c: [] for c in conditions
    }
    for i, (t, b) in enumerate(zip(temperatures, fields)):
        cond_col_map[(t, b)].append((i, col_names[i]))

    experiments: list[Experiment] = []
    for (T, B), cols in cond_col_map.items():
        # Build normalised name -> data-column index mapping for this block
        name_to_idx: dict[str, int] = {_norm(name): idx for idx, name in cols}

        def _find_idx(aliases: set) -> int | None:
            for a in aliases:
                if a in name_to_idx:
                    return name_to_idx[a]
            return None

        shift_idx = _find_idx(_SHIFT_NAMES)
        width_idx = _find_idx(_WIDTH_NAMES)
        area_idx = _find_idx(_AREA_NAMES)
        r1_idx = _find_idx(_R1_NAMES)
        lg_idx = _find_idx(_LG_NAMES)
        isotope_idx = _find_idx(_ISOTOPE_NAMES)

        for required, name in [
            (shift_idx, "shift"), (width_idx, "width"), (area_idx, "area")
        ]:
            if required is None:
                raise KeyError(
                    f"{file_name}: required column '{name}' not found "
                    f"for condition T={T} K, B={B} T."
                )

        signals: list[Signal] = []
        for row in data_rows:
            if not row:
                continue
            assignment = row[0].strip()
            if not assignment:
                raise ValueError(
                    f"{file_name}: row has an empty assignment. "
                    "Every signal must have a unique assignment label."
                )

            def _get(idx: int | None) -> float | None:
                if idx is None:
                    return None
                # +1 because col 0 is the assignment label
                try:
                    val = row[idx + 1].strip()
                    return float(val) if val else None
                except (IndexError, ValueError):
                    return None

            def _get_str(idx: int | None) -> str | None:
                if idx is None:
                    return None
                try:
                    val = row[idx + 1].strip()
                    return val if val else None
                except IndexError:
                    return None

            shift = _get(shift_idx)
            width = _get(width_idx)
            area = _get(area_idx)
            if shift is None or width is None or area is None:
                continue

            signals.append(Signal(
                shift,
                width,
                area,
                assignment,
                l_to_g=_get(lg_idx) or 1.0,
                r1=_get(r1_idx),
                isotope=_get_str(isotope_idx),
            ))

        experiments.append(Experiment(T, B, signals))

    return experiments


# ---------------------------------------------------------------------------
# Public read API
# ---------------------------------------------------------------------------

def load_experiments_from_csv(
    file_names: str | Iterable[str],
) -> List[Experiment]:
    """Load one or more Experiment objects from CSV file(s).

    Accepts both wide-format files (T and B as header rows) and legacy files
    (T and B as ``# key value`` comment lines).

    Args:
        file_names: Path or iterable of paths to CSV experiment files.

    Returns:
        List of Experiment objects sorted by temperature then magnetic field.
    """
    if isinstance(file_names, str):
        file_names = [file_names]

    experiments: list[Experiment] = []
    for file_name in file_names:
        if _is_wide_format(file_name):
            experiments.extend(_load_wide_experiments(file_name))
        else:
            experiments.extend(_load_legacy_experiments(file_name))

    experiments.sort(key=lambda e: (e.temperature, e.magnetic_field))
    return experiments


# ---------------------------------------------------------------------------
# Wide format writer
# ---------------------------------------------------------------------------

def write_experiments_to_csv(
    experiments: list[Experiment],
    file_name: str,
    *,
    delimiter: str = ",",
    verbose: bool = True,
) -> None:
    """Write one or more Experiment objects to a single wide-format CSV file.

    Args:
        experiments: Experiments to write (may span different T and B).
        file_name: Output file path.
        delimiter: Column separator. Defaults to tab.
        verbose: Log saved path when True.
    """
    has_r1 = any(
        sig.r1 is not None for exp in experiments for sig in exp.signals
    )
    has_lg = any(
        sig.l_to_g != 1.0 for exp in experiments for sig in exp.signals
    )

    has_isotope = any(
        sig.isotope is not None
        for exp in experiments
        for sig in exp.signals
    )

    sig_cols: list[str] = ["shift (ppm)", "width (Hz)", "area"]
    if has_r1:
        sig_cols.append("r1 (Hz)")
    if has_lg:
        sig_cols.append("L/G")
    if has_isotope:
        sig_cols.append("isotope")

    # Union of assignments in file order
    all_assignments: list[str] = list(
        dict.fromkeys(
            sig.assignment
            for exp in experiments
            for sig in exp.signals
        )
    )

    # Quick lookup: (T, B) -> assignment -> Signal
    sig_map: dict[tuple, dict[str, Signal]] = {
        (exp.temperature, exp.magnetic_field): {
            sig.assignment: sig for sig in exp.signals
        }
        for exp in experiments
    }

    d = delimiter

    def _fmt(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    rows: list[str] = []

    # Header rows
    temp_vals = d.join(
        _fmt(exp.temperature) for exp in experiments for _ in sig_cols
    )
    field_vals = d.join(
        _fmt(exp.magnetic_field) for exp in experiments for _ in sig_cols
    )
    col_header = d.join(col for _ in experiments for col in sig_cols)

    rows.append(f"temperature (K){d}{temp_vals}\n")
    rows.append(f"magnetic field (T){d}{field_vals}\n")
    rows.append(f"assignment{d}{col_header}\n")

    def _sig_cells(sig: Signal | None) -> list[str]:
        cells: list[str] = []
        for col in sig_cols:
            if sig is None:
                cells.append("")
            elif col == "shift (ppm)":
                cells.append(_fmt(sig.shift))
            elif col == "width (Hz)":
                cells.append(_fmt(sig.width))
            elif col == "area":
                cells.append(_fmt(sig.area))
            elif col == "r1 (Hz)":
                cells.append(_fmt(sig.r1))
            elif col == "L/G":
                cells.append(_fmt(sig.l_to_g))
            elif col == "isotope":
                cells.append(sig.isotope or "")
        return cells

    for asgn in all_assignments:
        cells = [asgn]
        for exp in experiments:
            sig = sig_map[(exp.temperature, exp.magnetic_field)].get(asgn)
            cells.extend(_sig_cells(sig))
        rows.append(d.join(cells) + "\n")

    timestamp = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    with open(file_name, "w", encoding="utf-8-sig", newline="") as f:
        f.write(f"# Generated by SimpNMR v{__version__} at {timestamp}\n")
        f.writelines(rows)

    if verbose:
        logger.info("Experiments written to %s", file_name)


def write_experiment_to_csv(
    experiment: Experiment,
    file_name: str,
    *,
    delimiter: str = ",",
    comment: str = "",
    verbose: bool = True,
) -> None:
    """Write a single Experiment to a wide-format CSV file.

    Args:
        experiment: Experiment instance to serialize.
        file_name: Output CSV file path.
        delimiter: Column separator.
        comment: Unused (kept for API compatibility).
        verbose: Log saved path when True.
    """
    write_experiments_to_csv(
        [experiment], file_name, delimiter=delimiter, verbose=verbose
    )
