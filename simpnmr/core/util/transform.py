# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Transform coordinates for PCS geometry mapping.

Provides pure helpers to align susceptibility-source and HFC geometries and to
rotate coordinates into the susceptibility principal-axis (chi) frame.
"""

import logging
import os
import re

import numpy as np
import numpy.linalg as la

from simpnmr.io.xyz import xyz_write
from simpnmr.tools.coords import xyz_fmt

logger = logging.getLogger(__name__)

# Kabsch alignment of the susceptibility-source geometry onto the hyperfine
# geometry pairs atoms by row order. A large post-alignment RMSD therefore
# signals that the two structures do not correspond (different atom ordering,
# or a genuinely different geometry), which would produce a wrongly oriented
# susceptibility. Above this threshold (Å) we emit a prominent warning rather
# than failing silently.
_ALIGN_RMSD_WARN_ANGSTROM = 1.0


def rotate_coords(coords: np.ndarray, rot_mat: np.ndarray) -> np.ndarray:
    """Rotate Cartesian coordinates by a rotation matrix.

    Args:
        coords: Cartesian coordinates with shape ``(n_atoms, 3)``.
        rot_mat: Rotation matrix with shape ``(3, 3)``.

    Returns:
        np.ndarray: Rotated coordinates with shape ``(n_atoms, 3)``.
    """
    coords = np.asarray(coords, dtype=float)
    rot_mat = np.asarray(rot_mat, dtype=float)
    return coords @ rot_mat.T


def rotate_tensor(tensor: np.ndarray, rot_mat: np.ndarray) -> np.ndarray:
    """Rotate a rank-2 Cartesian tensor by a rotation matrix.

    Args:
        tensor: Tensor with shape ``(3, 3)``.
        rot_mat: Rotation matrix with shape ``(3, 3)``.

    Returns:
        np.ndarray: Rotated tensor with shape ``(3, 3)``.
    """
    tensor = np.asarray(tensor, dtype=float)
    rot_mat = np.asarray(rot_mat, dtype=float)
    return rot_mat @ tensor @ rot_mat.T


def get_rotation_and_transformation(
    *,
    chi_tensor: np.ndarray,
    temperature: float,
    chi_source_coords: np.ndarray,
    dft_coords: np.ndarray,
):
    """Compute the rotation and chi-frame transformation for PCS mapping.

    Args:
        chi_tensor: Susceptibility tensor at the target temperature.
        temperature: Temperature corresponding to ``chi_tensor``.
        chi_source_coords: Coordinates from the susceptibility/chi source
            geometry.
        dft_coords: Coordinates from the HFC/DFT geometry.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - rot_mat (np.ndarray): Rotation matrix with shape (3, 3) that maps
              DFT-frame coordinates/components into the susceptibility-source
              frame.
            - trans_mat (np.ndarray): Transformation matrix with shape (3, 3)
              that maps DFT-frame coordinates/components into the susceptibility
              principal-axis (chi) frame.

    Raises:
        ValueError: If the susceptibility-source and DFT coordinate sets do not
            have the same number of atoms.
    """
    chi_tensor = np.asarray(chi_tensor, dtype=float)
    chi_source_coords = np.asarray(chi_source_coords, dtype=float)
    dft_coords = np.asarray(dft_coords, dtype=float)

    if len(chi_source_coords) != len(dft_coords):
        raise ValueError(
            "NEVPT2 and DFT coordinate sets have different lengths; cannot determine "
            "a meaningful rotational alignment."
        )

    # Compute rotation aligning DFT → NEVPT2 (susceptibility frame).
    # `find_rotation(coords_1, coords_2)` returns R
    # such that coords_2 rotated onto coords_1.
    if np.allclose(chi_source_coords, dft_coords, rtol=1e-6, atol=1e-8):
        rot_mat = np.eye(3)
        rmsd = 0.0
    else:
        rot_mat, rmsd = xyz_fmt.find_rotation(chi_source_coords, dft_coords)

    # Temperature-normalised tensor (scaling does not change eigenvectors)
    chi = chi_tensor / temperature

    # Use a single chi-frame convention everywhere:
    # 1) remove isotropic (trace) component
    # 2) diagonalise
    # 3) sort axes by |eigenvalue| (same convention as rotate_coords_to_chi_frame)
    chi_traceless = chi - np.eye(3) * (np.trace(chi) / 3.0)
    evals, evecs = la.eigh(chi_traceless)
    idx = np.argsort(np.abs(evals))
    evecs = evecs[:, idx]

    # Transformation matrix mapping DFT → chi frame
    trans_mat = evecs.T @ rot_mat

    if rmsd > _ALIGN_RMSD_WARN_ANGSTROM:
        logger.warning(
            "Susceptibility and hyperfine geometries do not correspond well: "
            "alignment RMSD = %.2f A (> %.2f A). The susceptibility tensor's "
            "orientation, and hence the pseudocontact shifts and PCS field, may "
            "be unreliable. They should be the same molecule with matching atom "
            "ordering; check the two structure files. If the tensor is already "
            "in the hyperfine structure's frame, supply it as a CSV instead of "
            "an ab-initio geometry.",
            rmsd,
            _ALIGN_RMSD_WARN_ANGSTROM,
        )
    elif rmsd > 0.0:
        logger.warning(
            "Distinct Susceptibility and DFT geometries detected; "
            "applied rotational alignment (RMSD = %.2f A).",
            rmsd,
        )

    return rot_mat, trans_mat


def rotate_coords_to_chi_frame(
    file_path,
    *,
    chi_tensor: np.ndarray,
    chi_source_labels,
    chi_source_coords,
):
    """Rotate susceptibility-source coordinates into the chi principal-axis frame.

    Args:
        file_path (str): Directory in which the output chi-frame XYZ file should
            be saved.
        chi_tensor (np.ndarray): Susceptibility tensor at the target
            temperature.
        chi_source_labels: Atomic labels from the susceptibility/chi source
            geometry.
        chi_source_coords: Atomic coordinates from the susceptibility/chi source
            geometry.

    Returns:
        list[tuple[str, np.ndarray]]: A list of ``(label, coordinate)`` pairs
        representing the rotated structure, suitable for downstream processing.
    """
    chi_tensor = np.asarray(chi_tensor, dtype=float)
    chi_source_coords = np.asarray(chi_source_coords, dtype=float)

    # Subtract isotropic component (trace)
    chi_tensor_traceless = chi_tensor - np.eye(3) * (np.trace(chi_tensor) / 3.0)

    # Diagonalize matrix
    eigvals_traceless, eigvecs_traceless = la.eigh(chi_tensor_traceless)

    idx = np.argsort(np.abs(eigvals_traceless))

    # Single chi-frame convention (must match get_rotation_and_transformation):
    # sort eigenvectors by |eigenvalue|, no additional arbitrary global-axis alignment.
    eigenvecs_sort_traceless = eigvecs_traceless[:, idx]

    # Center susceptibility-source coordinates
    chi_source_coords_center = chi_source_coords.mean(axis=0, keepdims=True)
    chi_source_coords_centerless = chi_source_coords - chi_source_coords_center

    # Convert susceptibility-source coordinates to chi frame
    chi_source_coords_chi_frame = (
        rotate_coords(chi_source_coords_centerless, eigenvecs_sort_traceless.T)
        + chi_source_coords_center
    )

    # Clean labels (remove numeric indices, if any)
    clean_labels = [re.sub(r"\d+", "", str(label)) for label in chi_source_labels]

    # Prepare output directory and filename
    os.makedirs(file_path, exist_ok=True)
    xyz_filename = os.path.join(file_path, "chi_frame_structure.xyz")

    # Build a descriptive comment line
    _comment = "NEVPT2 coordinates rotated into the susceptibility (chi) frame."

    # Save XYZ
    xyz_write.save_xyz(
        file_name=xyz_filename,
        labels=clean_labels,
        coords=chi_source_coords_chi_frame,
        verbose=False,
        comment=_comment,
    )

    logger.info("Chi-frame coordinates saved to %s", xyz_filename)

    # Return list of (label, coord) tuples for possible downstream use
    coords_chi_frame_out = list(zip(clean_labels, chi_source_coords_chi_frame))

    return coords_chi_frame_out
