"""Align all 8 quartet F1..F8 ORCA .out files to a common frame and rewrite
each one with aligned geometry + correspondingly-rotated HFC tensors.

Standard frame definition (per request):
  - Co at the origin (0, 0, 0)
  - N1 along +x axis (projection into the ring plane is along +x)
  - The CoGa7 metal ring lies in the xy plane (z = ring-normal direction)

After running this, the 8 aligned .out files can be safely passed to
  simpnmr average_conformers <aligned_files> --centre Co1 --weights ...

because the (3,3) A tensors are now all expressed in the same Cartesian
basis, so the per-atom average is geometrically meaningful (whereas the
unaligned files have ~0.5 Å rigid-body offsets and small rotations
between F-orientations).

USAGE:
  cd /path/to/opt_rotated
  python3 align_quartet_to_standard_frame.py

This writes Ga7Co_B3LYP_def2TZVP_straddleF{k}_opt_quartet_aligned.out for
each k = 1..8, leaving the originals untouched.

The script parses and rewrites the geometry block, the per-nucleus
"Total HFC matrix" blocks, the A(FC)/A(SD)/A(Tot) principal-value lines,
AND the Orientation: eigenvector block — so the reconstructed FC/SD tensors
in simpnmr are correctly expressed in the aligned frame.
"""
from __future__ import annotations

import os
import re
import sys
import numpy as np
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent

IDX_GA = list(range(0, 7))
IDX_CO = 7
IDX_RING = [IDX_CO] + IDX_GA
IDX_N1 = 282

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
CART_HDR_RE = re.compile(r"^\s*CARTESIAN COORDINATES \(ANGSTROEM\)\s*$")
NUC_HDR_RE = re.compile(
    r"^\s*Nucleus\s+(\d+)([A-Z][a-z]?)\s*:\s*A\s*:\s*Isotope"
)


def read_last_geometry(lines: list[str]) -> tuple[list[str], np.ndarray, int]:
    starts = [i for i, ln in enumerate(lines) if CART_HDR_RE.match(ln)]
    if not starts:
        raise RuntimeError("No CARTESIAN COORDINATES block found")
    start = starts[-1]
    elements: list[str] = []
    coords: list[list[float]] = []
    j = start + 2
    while j < len(lines):
        s = lines[j].strip()
        if not s:
            break
        parts = s.split()
        if len(parts) == 4:
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                elements.append(parts[0])
                coords.append([x, y, z])
                j += 1
                continue
            except ValueError:
                break
        else:
            break
    return elements, np.asarray(coords, dtype=float), start


def iter_hfc_blocks(lines: list[str]) -> Iterator[tuple[int, int, int, str]]:
    for i, ln in enumerate(lines):
        m = NUC_HDR_RE.match(ln)
        if not m:
            continue
        idx = int(m.group(1))
        el = m.group(2)
        j = i + 1
        while j < len(lines):
            if "A(Tot)" in lines[j] and "A(iso)" in lines[j]:
                break
            j += 1
        else:
            continue
        yield (i, j, idx, el)


def parse_total_hfc_matrix(lines: list[str], hdr_start: int) -> np.ndarray:
    j = hdr_start
    while j < len(lines):
        if "Total HFC matrix" in lines[j]:
            break
        j += 1
    rows = []
    k = j + 2
    while k < len(lines) and len(rows) < 3:
        s = lines[k].strip()
        if not s:
            k += 1
            continue
        parts = s.split()
        if len(parts) >= 3:
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                k += 1
                if len(rows) == 3:
                    break
                continue
            except ValueError:
                pass
        k += 1
    return np.asarray(rows, dtype=float)


def parse_orientation_block(
    lines: list[str], atot_end: int, search_limit: int = 10
) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Find the Orientation: block after atot_end and return (r_mat, orient_hdr_idx).

    r_mat has the same convention as the ORCA reader: r_mat_new = R @ r_mat_orig
    gives the correctly rotated orientation matrix.  Returns (None, None) if not
    found within search_limit lines.
    """
    for i in range(atot_end + 1, min(atot_end + 1 + search_limit, len(lines))):
        if "Orientation:" in lines[i]:
            r_rows = []
            for j in range(i + 1, i + 4):
                parts = lines[j].split()
                r_rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
            return np.array(r_rows, dtype=float), i
    return None, None


# ---------------------------------------------------------------------------
# Frame alignment
# ---------------------------------------------------------------------------
def build_alignment_R(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    co = coords[IDX_CO].copy()
    ring_pts = coords[IDX_RING] - co
    n1_rel = coords[IDX_N1] - co

    _, _, vt = np.linalg.svd(ring_pts, full_matrices=False)
    z_hat = vt[-1]
    if np.dot(z_hat, n1_rel) < 0:
        z_hat = -z_hat
    z_hat /= np.linalg.norm(z_hat)

    n1_proj = n1_rel - np.dot(n1_rel, z_hat) * z_hat
    nrm = np.linalg.norm(n1_proj)
    if nrm < 1e-6:
        ga1_rel = coords[IDX_GA[0]] - co
        n1_proj = ga1_rel - np.dot(ga1_rel, z_hat) * z_hat
        nrm = np.linalg.norm(n1_proj)
    x_hat = n1_proj / nrm
    y_hat = np.cross(z_hat, x_hat)
    y_hat /= np.linalg.norm(y_hat)
    z_hat = np.cross(x_hat, y_hat)
    z_hat /= np.linalg.norm(z_hat)

    R = np.vstack([x_hat, y_hat, z_hat])
    return R, co


def apply_alignment(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ (coords - t).T).T


def rotate_tensor(A: np.ndarray, R: np.ndarray) -> np.ndarray:
    return R @ A @ R.T


# ---------------------------------------------------------------------------
# Principal values
# ---------------------------------------------------------------------------
def principal_values_from_total(A_total: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    A_iso = float(np.trace(A_total) / 3.0)
    A_dip = A_total - np.eye(3) * A_iso
    dip_vals = np.linalg.eigvalsh(A_dip)
    tot_vals = np.linalg.eigvalsh(A_total)
    return A_iso, dip_vals, tot_vals


# ---------------------------------------------------------------------------
# Emission helpers
# ---------------------------------------------------------------------------
def emit_geometry_block(elements: list[str], coords: np.ndarray) -> list[str]:
    out = [
        "CARTESIAN COORDINATES (ANGSTROEM)\n",
        "---------------------------------\n",
    ]
    for el, (x, y, z) in zip(elements, coords):
        out.append(f"  {el:<2s}   {x:10.6f}   {y:10.6f}   {z:10.6f}\n")
    return out


def emit_total_hfc_matrix_block(A_total: np.ndarray) -> list[str]:
    rows = []
    for i in range(3):
        rows.append(
            f"               {A_total[i, 0]:9.4f}            "
            f"{A_total[i, 1]:9.4f}            {A_total[i, 2]:9.4f}\n"
        )
    return rows


def emit_principal_lines(A_iso: float, dip_vals: np.ndarray, tot_vals: np.ndarray) -> list[str]:
    fc = A_iso
    return [
        f" A(FC)        {fc:11.4f}        {fc:11.4f}        {fc:11.4f}\n",
        f" A(SD)        {dip_vals[0]:11.4f}        {dip_vals[1]:11.4f}        {dip_vals[2]:11.4f}\n",
        "             ----------           ----------           ----------\n",
        f" A(Tot)       {tot_vals[0]:11.4f}        {tot_vals[1]:11.4f}        {tot_vals[2]:11.4f}    A(iso)={A_iso:11.4f}\n",
    ]


def emit_orientation_block(r_mat_new: np.ndarray) -> list[str]:
    """Emit updated Orientation: rows (NOT including the 'Orientation:' header line)."""
    out = []
    for axis, row in zip("XYZ", r_mat_new):
        out.append(
            f"               {axis}        "
            f"{row[0]:10.7f}   {row[1]:10.7f}   {row[2]:10.7f}\n"
        )
    return out


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------
def rewrite_out(in_path: Path, out_path: Path) -> dict:
    lines = in_path.read_text().splitlines(keepends=True)

    elements, coords, geom_start = read_last_geometry(lines)
    R, t = build_alignment_R(coords)
    new_coords = apply_alignment(coords, R, t)

    blocks = list(iter_hfc_blocks(lines))

    new_lines = list(lines)

    # Replace last geometry block
    j = geom_start + 2
    while j < len(new_lines):
        s = new_lines[j].strip()
        if not s or len(s.split()) != 4:
            break
        try:
            float(s.split()[1])
        except ValueError:
            break
        j += 1
    new_geom = emit_geometry_block(elements, new_coords)
    new_lines[geom_start:j] = new_geom

    lines2 = "".join(new_lines).splitlines(keepends=True)

    blocks2 = list(iter_hfc_blocks(lines2))
    if len(blocks2) != len(blocks):
        raise RuntimeError(
            f"Block count mismatch after geometry rewrite: {len(blocks2)} vs {len(blocks)}"
        )

    out_lines: list[str] = []
    cursor = 0
    for hdr_start, atot_end, _nuc_idx, _el in blocks2:
        A_orig = parse_total_hfc_matrix(lines2, hdr_start)
        A_new = rotate_tensor(A_orig, R)
        iso_v, dip_v, tot_v = principal_values_from_total(A_new)

        k = hdr_start
        while k < atot_end and "Total HFC matrix" not in lines2[k]:
            k += 1
        mat_rows_start = k + 2
        mat_rows_end = mat_rows_start + 3

        out_lines.extend(lines2[cursor:mat_rows_start])
        out_lines.extend(emit_total_hfc_matrix_block(A_new))

        afc_idx = mat_rows_end
        while afc_idx < atot_end and "A(FC)" not in lines2[afc_idx]:
            afc_idx += 1
        out_lines.extend(lines2[mat_rows_end:afc_idx])
        out_lines.extend(emit_principal_lines(iso_v, dip_v, tot_v))

        # --- Update Orientation: block (NEW) ---
        r_mat_orig, orient_hdr = parse_orientation_block(lines2, atot_end)
        if r_mat_orig is not None:
            # r_mat_new = R @ r_mat_orig  (rotate each eigenvector column)
            r_mat_new = R @ r_mat_orig
            # Copy any lines between A(Tot) and "Orientation:" verbatim
            out_lines.extend(lines2[atot_end + 1 : orient_hdr + 1])
            # Emit new orientation rows, skip old ones
            out_lines.extend(emit_orientation_block(r_mat_new))
            cursor = orient_hdr + 4  # past "Orientation:" + 3 data rows
        else:
            cursor = atot_end + 1

    out_lines.extend(lines2[cursor:])
    out_path.write_text("".join(out_lines))

    R_axis_angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return {
        "n_nuclei": len(blocks2),
        "rotation_angle_deg": float(R_axis_angle),
        "translation_norm": float(np.linalg.norm(t)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    files = sorted(HERE.glob("Ga7Co_B3LYP_def2TZVP_straddleF*_opt_quartet.out"))
    if not files:
        print("No quartet .out files found in", HERE, file=sys.stderr)
        return 1

    print(f"Aligning {len(files)} quartet output(s) to standard frame")
    print(f"  Co at origin, N1 along +x, CoGa7 ring in xy plane\n")
    print(f"{'file':<60s} {'#nuclei':>8s} {'|t| (Å)':>10s} {'rot°':>10s}")
    print("-" * 92)

    for f in files:
        if "_aligned" in f.name:
            continue
        out = f.with_name(f.stem + "_aligned.out")
        try:
            info = rewrite_out(f, out)
            print(
                f"{f.name:<60s} {info['n_nuclei']:>8d} "
                f"{info['translation_norm']:>10.4f} "
                f"{info['rotation_angle_deg']:>10.2f}"
            )
        except Exception as e:
            print(f"{f.name:<60s}  ERROR: {e}")

    print("\nAfter alignment, run:")
    print(
        "  simpnmr average_conformers *_opt_quartet_aligned.out \\\n"
        "    --centre Co1 --weights <your weights> \\\n"
        "    --output conformer_avg_quartet_aligned.csv"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
