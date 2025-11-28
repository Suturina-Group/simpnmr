"""
Module for rotation, alignment, and chi-frame transformations in PCS-related
coordinate mapping between NEVPT2 and DFT geometries. The module loads input
settings from a YAML configuration, reads susceptibility and hyperfine data,
and exposes `get_rotation_and_transformation` as its main public API.
"""
import re
import os
import glob
import datetime
import numpy as np
import numpy.linalg as la
import xyz_py as xyzp

from . import readers as rdrs
from . import utils as ut
from .inputs import PredictConfig
from .__version__ import __version__

def access_input_data():
    default_ymls = glob.glob(os.path.join(os.getcwd(), "*.yml"))

    if not default_ymls:
        raise FileNotFoundError("No .yml file found in current directory")

    INPUT_YML = os.environ.get("SIMPNMR_INPUT", default_ymls[0])

    cfg = PredictConfig.from_file(INPUT_YML)
    susc_path = cfg.susceptibility_file
    hfc_path = cfg.hyperfine_file

    # Temperatures come from YAML; we treat it as a single-element list for now
    temperature = cfg.susceptibility_temperatures

    # NEVPT2 coordinates
    nevpt2_labels, nevpt2_coords = rdrs.read_orca5_output_xyz(susc_path)

    # DFT coordinates
    qca = rdrs.QCA.guess_from_file(hfc_path)
    nevpt2_labels = qca.labels
    dft_coords = qca.coords

    # Susceptibility tensor
    chi_dict = rdrs.read_orca_susceptibility(susc_path, section="nevpt2")
    chiT = chi_dict[temperature[0]]

    return chiT, temperature, nevpt2_labels,nevpt2_coords, dft_coords


def get_rotation_and_transformation():
    """
    Compute and return both the rotation matrix (R) aligning NEVPT2 and DFT
    coordinate sets, and the final transformation matrix (trans_mat) using
    the susceptibility tensor.

    Returns
    -------
    tuple of numpy.ndarray
        R : (3, 3)
            Rotation matrix that aligns NEVPT2 geometry to DFT geometry.
        trans_mat : (3, 3)
            Transformation matrix used for PCS-related coordinate mapping.
    """

    chiT, temperature, _, nevpt2_coords, dft_coords = access_input_data()

    if np.allclose(nevpt2_coords, dft_coords, rtol=1e-6, atol=1e-8):
        return np.eye(3), np.eye(3)
    else:
        pass

    if len(nevpt2_coords) != len(dft_coords):
        raise ValueError

    # Compute rotation aligning NEVPT2 → DFT
    rot_mat, rmsd = xyzp.find_rotation(nevpt2_coords, dft_coords)

    # Temperature-normalised tensor
    chi = chiT / temperature[0]

    # Eigen-decomposition
    evals, evecs = la.eigh(chi)

    # Transformation matrix
    trans_mat = evecs.T @ rot_mat

    ut.cprint(
        f' Distinct Susceptibility and DFT geometries detected; applied rotational alignment (RMSD = {rmsd:.6f}). \n',
          'cyan'
          )
    
    # Need to add an additional functional to check if the HFC coords are in chi frame because it leads to the wrong prediction

    return rot_mat, trans_mat

def rotate_coords_to_chi_frame(file_path):

    chiT, _, nevpt2_labels, nevpt2_coords, _ = access_input_data()

    # Subtract isotropic component (trace)
    chiT_traceless = chiT - np.eye(3) * (np.trace(chiT) / 3.0)

    # Diagonalize matrix
    eigvals_traceless, eigvecs_traceless = la.eigh(chiT_traceless)

    idx = np.argsort(np.abs(eigvals_traceless))

    # Rotate eigenvectors so principal axis aligns with global Z
    eigvecs_sorted = eigvecs_traceless[:, idx]
    u = eigvecs_sorted[:, 2]
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(u, z_axis)
    if np.linalg.norm(cross) < 1e-8:
        R = np.eye(3)

    else:
        a = cross / np.linalg.norm(cross)
        theta = np.arccos(np.dot(u, z_axis))
        A = np.array([
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0]
        ])

        R = np.eye(3) + np.sin(theta) * A + (1.0 - np.cos(theta)) * (A @ A)

    eigenvecs_sort_traceless = R @ eigvecs_sorted

    # Center NEVPT2 coordinates
    nevpt2_coords_center = nevpt2_coords.mean(axis=0, keepdims=True)
    nevpt2_coords_centerless = nevpt2_coords - nevpt2_coords_center

    # Convert NEVPT2 coordinates to chi frame
    nevpt2_coords_chi_frame = (
        nevpt2_coords_centerless @ eigenvecs_sort_traceless
        + nevpt2_coords_center
    )

    # Clean labels (remove numeric indices, if any)
    clean_labels = [re.sub(r"\d+", "", str(label)) for label in nevpt2_labels]

    # Prepare output directory and filename
    os.makedirs(file_path, exist_ok=True)
    xyz_filename = os.path.join(file_path, "chi_frame_structure.xyz")

    # Build a descriptive comment line
    _comment = (
        f"NEVPT2 coordinates rotated into the susceptibility (chi) frame. "
        f"This file was generated with SimpNMR v{__version__} "
        f"at {datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y')}."
    )

    # Save XYZ using the helper from xyz_py
    xyzp.save_xyz(
        xyz_filename,
        labels=clean_labels,
        coords=nevpt2_coords_chi_frame,
        verbose=False,
        comment=_comment,
    )

    ut.cprint(f"\n Chi-frame coordinates saved to {xyz_filename}\n", "cyan")

    # Return list of (label, coord) tuples for possible downstream use
    coords_chi_frame_out = list(zip(clean_labels, nevpt2_coords_chi_frame))
    return coords_chi_frame_out