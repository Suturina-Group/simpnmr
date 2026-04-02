# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Identify and label methyl and tert-butyl groups in XYZ structures.

Reads a molecular structure XYZ file (standard or Chemcraft format),
identifies all unique methyl (CH3) and tert-butyl (C(CH3)3) groups,
and produces:

  1. A labeled XYZ file  (<input>_labeled.xyz)
       - element symbols are never renamed
       - group labels are appended in quotes after the coordinates
  2. A CSV label file    (<input>_labels.csv)
       - columns: atom_label, chem_label  (C and H atoms only)

Supports standard XYZ (first line = atom count, second = comment) and
Chemcraft XYZ (no header; lines: index  atomic_num  x  y  z  ["label"]).

Bond detection is distance-based:
    C-H  <  1.35 Å
    C-C  <  1.85 Å
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict

from simpnmr.core.const.isotopes import DEFAULT_ISOTOPES
from simpnmr.tools.coords import xyz_fmt as xyzf

logger = logging.getLogger(__name__)

# ── Bond-length cutoffs (Å) ────────────────────────────────────────────────
_BOND_CUTOFFS = {
    frozenset(["C", "H"]): 1.35,
    frozenset(["C", "C"]): 1.85,
}


# ═══════════════════════════════════════════════════════════════════════════
#  I / O
# ═══════════════════════════════════════════════════════════════════════════

def _is_chemcraft(lines: list[str]) -> bool:
    """Return True if the file looks like Chemcraft format.

    Chemcraft XYZ has no header; each data line starts with an integer
    index followed by an integer atomic number and three floats.
    """
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            int(parts[0])    # line index
            int(parts[1])    # atomic number
            float(parts[2])  # x
            float(parts[3])  # y
            float(parts[4])  # z
            return True
        except (ValueError, IndexError):
            return False
    return False


def _read_file(
    path: str,
) -> tuple[str, str, list[list], list[list]]:
    """Auto-detect format and parse atoms.

    Returns:
        fmt:      "standard" or "chemcraft"
        comment:  comment string (empty for chemcraft)
        atoms:    list of [symbol, x, y, z]
        raw_rows: list of [sym_or_anum, xs, ys, zs, existing_label]
                  preserving original string fields for precision-faithful
                  rewriting.
    """
    with open(path) as f:
        lines = f.readlines()

    non_empty = [ln for ln in lines if ln.strip()]

    if _is_chemcraft(non_empty):
        fmt = "chemcraft"
        comment = ""
        atoms: list[list] = []
        raw_rows: list[list] = []
        for line in non_empty:
            parts = line.split()
            anum_s = parts[1]
            xs, ys, zs = parts[2], parts[3], parts[4]
            sym = xyzf.num_to_lab([int(anum_s)], numbered=False)[0]
            existing = parts[5].strip() if len(parts) >= 6 else ""
            atoms.append([sym, float(xs), float(ys), float(zs)])
            raw_rows.append([anum_s, xs, ys, zs, existing])
    else:
        fmt = "standard"
        n = int(non_empty[0].strip())
        comment = non_empty[1].rstrip("\n")
        atoms = []
        raw_rows = []
        for line in non_empty[2: 2 + n]:
            parts = line.split()
            sym = parts[0]
            xs, ys, zs = parts[1], parts[2], parts[3]
            existing = " ".join(parts[4:]) if len(parts) >= 5 else ""
            atoms.append([sym, float(xs), float(ys), float(zs)])
            raw_rows.append([sym, xs, ys, zs, existing])

    return fmt, comment, atoms, raw_rows


def _write_labeled_xyz(
    path: str,
    fmt: str,
    comment: str,
    atoms: list[list],
    raw_rows: list[list],
    group_labels: list[str],
) -> None:
    """Write labeled XYZ with group tags appended in quotes."""
    lines = []

    if fmt == "standard":
        lines.append(f"{len(atoms)}\n")
        lines.append(f"{comment} [labeled]\n")
        for i, atom in enumerate(atoms):
            sym = atom[0]
            xs, ys, zs = raw_rows[i][1], raw_rows[i][2], raw_rows[i][3]
            tag = group_labels[i]
            if tag:
                lines.append(
                    f"{sym:<4s} {xs:>14s} {ys:>14s} {zs:>14s}"
                    f'   "{tag}"\n'
                )
            else:
                lines.append(
                    f"{sym:<4s} {xs:>14s} {ys:>14s} {zs:>14s}\n"
                )
    else:
        for i, atom in enumerate(atoms):
            anum_s, xs, ys, zs = (
                raw_rows[i][0], raw_rows[i][1],
                raw_rows[i][2], raw_rows[i][3],
            )
            tag = group_labels[i]
            if tag:
                lines.append(
                    f"{anum_s:>5s}"
                    f"  {xs:>14s}  {ys:>14s}  {zs:>14s}"
                    f'      "{tag}"\n'
                )
            else:
                lines.append(
                    f"{anum_s:>5s}"
                    f"  {xs:>14s}  {ys:>14s}  {zs:>14s}\n"
                )

    with open(path, "w") as f:
        f.writelines(lines)
    logger.info("Labeled XYZ  ->  %s", path)


def _write_csv(
    path: str,
    atoms: list[list],
    group_labels: list[str],
    raw_rows: list[list],
) -> None:
    """Write atom_label,chem_label,isotope CSV for C and H atoms."""
    c_counter = 0
    h_counter = 0
    rows = []
    for i, atom in enumerate(atoms):
        sym = atom[0]
        if sym not in ("C", "H"):
            continue
        label = group_labels[i]
        if not label:
            existing = raw_rows[i][-1].strip().strip('"')
            label = existing
        if sym == "C":
            c_counter += 1
            atom_label = f"C{c_counter}"
        else:
            h_counter += 1
            atom_label = f"H{h_counter}"
        if not label:
            label = atom_label
        isotope = DEFAULT_ISOTOPES.get(sym, "")
        rows.append(f"{atom_label},{label},{isotope}")

    with open(path, "w") as f:
        f.write("atom_label,chem_label,isotope\n")
        f.write("\n".join(rows) + "\n")
    logger.info("CSV labels   ->  %s", path)


# ═══════════════════════════════════════════════════════════════════════════
#  Chemistry
# ═══════════════════════════════════════════════════════════════════════════

def _build_adjacency(atoms: list[list]) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = defaultdict(list)
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = atoms[i][0], atoms[j][0]
            cutoff = _BOND_CUTOFFS.get(frozenset([si, sj]))
            if cutoff is None:
                continue
            dx = atoms[i][1] - atoms[j][1]
            dy = atoms[i][2] - atoms[j][2]
            dz = atoms[i][3] - atoms[j][3]
            if dx * dx + dy * dy + dz * dz <= cutoff * cutoff:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _find_methyl_carbons(
    atoms: list[list], adj: dict[int, list[int]]
) -> set[int]:
    """C bonded to exactly 3 H and at most 1 non-H neighbour."""
    methyl_C: set[int] = set()
    for i, atom in enumerate(atoms):
        if atom[0] != "C":
            continue
        h_count = sum(1 for nb in adj[i] if atoms[nb][0] == "H")
        non_h = sum(1 for nb in adj[i] if atoms[nb][0] != "H")
        if h_count == 3 and non_h <= 1:
            methyl_C.add(i)
    return methyl_C


def _find_tbu_carbons(
    atoms: list[list],
    adj: dict[int, list[int]],
    methyl_C: set[int],
) -> set[int]:
    """C bonded to exactly 3 methyl-C, 0 H, and at most 1 other non-H."""
    tbu_C: set[int] = set()
    for i, atom in enumerate(atoms):
        if atom[0] != "C":
            continue
        h_count = sum(1 for nb in adj[i] if atoms[nb][0] == "H")
        me_count = sum(1 for nb in adj[i] if nb in methyl_C)
        other = sum(
            1 for nb in adj[i]
            if atoms[nb][0] != "H" and nb not in methyl_C
        )
        if h_count == 0 and me_count == 3 and other <= 1:
            tbu_C.add(i)
    return tbu_C


def _collect_group_atoms(
    center_idx: int,
    atoms: list[list],
    adj: dict[int, list[int]],
    methyl_C: set[int],
    kind: str,
) -> set[int]:
    members = {center_idx}
    if kind == "methyl":
        for nb in adj[center_idx]:
            if atoms[nb][0] == "H":
                members.add(nb)
    elif kind == "tbu":
        for nb in adj[center_idx]:
            if nb in methyl_C:
                members.add(nb)
                for h in adj[nb]:
                    if atoms[h][0] == "H":
                        members.add(h)
    return members


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def label_groups(input_path: str) -> None:
    """Identify methyl/tBu groups and write labeled XYZ and CSV.

    Args:
        input_path: Path to the input XYZ file.
    """
    base = os.path.splitext(input_path)[0]
    xyz_out = base + "_labeled.xyz"
    csv_out = base + "_labels.csv"

    fmt, comment, atoms, raw_rows = _read_file(input_path)
    logger.info("Format detected: %s  (%d atoms)", fmt, len(atoms))

    adj = _build_adjacency(atoms)
    methyl_C = _find_methyl_carbons(atoms, adj)
    tbu_C = _find_tbu_carbons(atoms, adj, methyl_C)

    tbu_methyl_C: set[int] = set()
    for tc in tbu_C:
        for nb in adj[tc]:
            if nb in methyl_C:
                tbu_methyl_C.add(nb)
    standalone_methyl_C = methyl_C - tbu_methyl_C

    group_labels = [""] * len(atoms)
    groups: list[dict] = []

    for gid, tc in enumerate(sorted(tbu_C), start=1):
        tag = f"tBu{gid}"
        members = _collect_group_atoms(tc, atoms, adj, methyl_C, "tbu")
        for idx in members:
            group_labels[idx] = tag
        groups.append({
            "tag": tag,
            "kind": "tert-butyl",
            "center": tc,
            "members": sorted(members),
        })

    for gid, mc in enumerate(sorted(standalone_methyl_C), start=1):
        tag = f"Me{gid}"
        members = _collect_group_atoms(mc, atoms, adj, methyl_C, "methyl")
        for idx in members:
            group_labels[idx] = tag
        groups.append({
            "tag": tag,
            "kind": "methyl",
            "center": mc,
            "members": sorted(members),
        })

    _write_labeled_xyz(xyz_out, fmt, comment, atoms, raw_rows, group_labels)
    _write_csv(csv_out, atoms, group_labels, raw_rows)

    n_tbu = sum(1 for g in groups if g["kind"] == "tert-butyl")
    n_me = sum(1 for g in groups if g["kind"] == "methyl")
    logger.info(
        "Found %d tert-butyl group(s) and %d standalone methyl group(s).",
        n_tbu, n_me,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Identify methyl (CH3) and tert-butyl (C(CH3)3) groups in an\n"
            "XYZ structure file and write:\n"
            "  <input>_labeled.xyz  – XYZ with group labels in quotes\n"
            "  <input>_labels.csv   – atom_label,chem_label CSV"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="XYZ file to process (standard or Chemcraft format)",
    )
    uargs = parser.parse_args()
    label_groups(uargs.input_file)


if __name__ == "__main__":
    main()
