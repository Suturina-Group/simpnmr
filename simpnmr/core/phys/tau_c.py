# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Rotational correlation time calculator for NMR relaxation.

Computes the isotropic rotational correlation time τ_R from molecular
coordinates using the Perrin ellipsoid model (default) or the
HYDRONMR-style bead-shell model.

References:
    - Perrin, J Phys Radium 5, 497 (1934)
    - García de la Torre et al., Biophys J 78, 719 (2000)
    - Rotne & Prager, J Chem Phys 50, 4831 (1969)
    - Hu & Zwanzig, J Chem Phys 60, 4354 (1974)
"""

from __future__ import annotations

import logging
import os
import re

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_kB = 1.380649e-23  # J/K
_pi = np.pi

# ---------------------------------------------------------------------------
# Atomic data
# ---------------------------------------------------------------------------
ATOMIC_MASS: dict[str, float] = {
    'H':   1.008,  'He':   4.003, 'Li':   6.941, 'Be':   9.012,
    'B':  10.81,   'C':  12.011,  'N':  14.007,  'O':  15.999,
    'F':  18.998,  'Ne':  20.180, 'Na':  22.990,  'Mg':  24.305,
    'Al':  26.982, 'Si':  28.086, 'P':  30.974,  'S':  32.065,
    'Cl':  35.453, 'Ar':  39.948, 'K':  39.098,  'Ca':  40.078,
    'Sc':  44.956, 'Ti':  47.867, 'V':  50.942,  'Cr':  51.996,
    'Mn':  54.938, 'Fe':  55.845, 'Co':  58.933,  'Ni':  58.693,
    'Cu':  63.546, 'Zn':  65.38,  'Ga':  69.723, 'Ge':  72.630,
    'As':  74.922, 'Se':  78.971, 'Br':  79.904,  'Kr':  83.798,
    'Rb':  85.468, 'Sr':  87.62,  'Y':  88.906,  'Zr':  91.224,
    'Nb':  92.906, 'Mo':  95.95,  'Tc':  98.0,   'Ru': 101.07,
    'Rh': 102.91,  'Pd': 106.42, 'Ag': 107.87,  'Cd': 112.41,
    'In': 114.82,  'Sn': 118.71, 'Sb': 121.76,  'Te': 127.60,
    'I':  126.90,  'Xe': 131.29, 'Cs': 132.91,  'Ba': 137.33,
    'La': 138.91,  'Ce': 140.12, 'Pr': 140.91,  'Nd': 144.24,
    'Pm': 145.0,   'Sm': 150.36, 'Eu': 151.96,  'Gd': 157.25,
    'Tb': 158.93,  'Dy': 162.50, 'Ho': 164.93,  'Er': 167.26,
    'Tm': 168.93,  'Yb': 173.05, 'Lu': 174.97,  'Hf': 178.49,
    'Ta': 180.95,  'W':  183.84, 'Re': 186.21,  'Os': 190.23,
    'Ir': 192.22,  'Pt': 195.08, 'Au': 196.97,  'Hg': 200.59,
    'Tl': 204.38,  'Pb': 207.2,  'Bi': 208.98,  'Th': 232.04,
    'U':  238.03,
}

VDW_RADIUS: dict[str, float] = {
    'H':  1.20, 'He': 1.40, 'Li': 1.82, 'Be': 1.53,
    'B':  1.92, 'C':  1.70, 'N':  1.55, 'O':  1.52,
    'F':  1.47, 'Ne': 1.54, 'Na': 2.27, 'Mg': 1.73,
    'Al': 1.84, 'Si': 2.10, 'P':  1.80, 'S':  1.80,
    'Cl': 1.75, 'Ar': 1.88, 'K':  2.75, 'Ca': 2.31,
    'Sc': 2.15, 'Ti': 2.11, 'V':  2.07, 'Cr': 2.06,
    'Mn': 2.05, 'Fe': 2.05, 'Co': 2.00, 'Ni': 1.63,
    'Cu': 1.40, 'Zn': 1.39, 'Ga': 1.87, 'Ge': 2.11,
    'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
    'Rb': 3.03, 'Sr': 2.49, 'Y':  2.19, 'Zr': 2.16,
    'Nb': 2.08, 'Mo': 2.09, 'Tc': 2.09, 'Ru': 2.05,
    'Rh': 2.00, 'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58,
    'In': 1.93, 'Sn': 2.17, 'Sb': 2.06, 'Te': 2.06,
    'I':  1.98, 'Xe': 2.16, 'Cs': 3.43, 'Ba': 2.68,
    'La': 2.43, 'Ce': 2.42, 'Pr': 2.40, 'Nd': 2.39,
    'Pm': 2.38, 'Sm': 2.36, 'Eu': 2.35, 'Gd': 2.34,
    'Tb': 2.33, 'Dy': 2.31, 'Ho': 2.30, 'Er': 2.29,
    'Tm': 2.27, 'Yb': 2.26, 'Lu': 2.24, 'Hf': 2.23,
    'Ta': 2.22, 'W':  2.18, 'Re': 2.16, 'Os': 2.16,
    'Ir': 2.20, 'Pt': 1.75, 'Au': 1.66, 'Hg': 1.55,
    'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07, 'Th': 2.40,
    'U':  1.86,
}

# ---------------------------------------------------------------------------
# Solvent database: viscosity in Pa·s at 298 K with temperature correction
# via Arrhenius-like scaling.  Ea values (J/mol) are approximate.
# ---------------------------------------------------------------------------
SOLVENTS: dict[str, dict] = {
    'D2O':         {'eta': 1.100e-3, 'name': 'D₂O',         'Ea': 17800},
    'CDCl3':       {'eta': 0.540e-3, 'name': 'CDCl₃',       'Ea': 10200},
    'CD2Cl2':      {'eta': 0.410e-3, 'name': 'CD₂Cl₂',      'Ea':  9800},
    'DMSO':        {'eta': 1.990e-3, 'name': 'DMSO-d₆',     'Ea': 18200},
    'DMSO-d6':     {'eta': 1.990e-3, 'name': 'DMSO-d₆',     'Ea': 18200},
    'MeOD':        {'eta': 0.545e-3, 'name': 'CD₃OD',       'Ea': 12500},
    'CD3OD':       {'eta': 0.545e-3, 'name': 'CD₃OD',       'Ea': 12500},
    'acetone':     {'eta': 0.310e-3, 'name': 'acetone-d₆',  'Ea':  8400},
    'acetone-d6':  {'eta': 0.310e-3, 'name': 'acetone-d₆',  'Ea':  8400},
    'C6D6':        {'eta': 0.600e-3, 'name': 'C₆D₆',        'Ea': 11200},
    'benzene-d6':  {'eta': 0.600e-3, 'name': 'C₆D₆',        'Ea': 11200},
    'toluene-d8':  {'eta': 0.560e-3, 'name': 'toluene-d₈',  'Ea': 11000},
    'THF-d8':      {'eta': 0.460e-3, 'name': 'THF-d₈',      'Ea': 10000},
    'CD3CN':       {'eta': 0.340e-3, 'name': 'CD₃CN',       'Ea':  9200},
    'MeCN-d3':     {'eta': 0.340e-3, 'name': 'CD₃CN',       'Ea':  9200},
    'DMF-d7':      {'eta': 0.790e-3, 'name': 'DMF-d₇',      'Ea': 14000},
    'pyridine-d5': {'eta': 0.880e-3, 'name': 'pyridine-d₅', 'Ea': 13000},
    'water':       {'eta': 0.890e-3, 'name': 'H₂O',         'Ea': 17800},
    'methanol':    {'eta': 0.544e-3, 'name': 'MeOH',        'Ea': 12500},
    'ethanol':     {'eta': 1.074e-3, 'name': 'EtOH',        'Ea': 15200},
    'chloroform':  {'eta': 0.537e-3, 'name': 'CHCl₃',       'Ea': 10200},
    'DCM':         {'eta': 0.413e-3, 'name': 'CH₂Cl₂',      'Ea':  9800},
}

_R_GAS = 8.314  # J/(mol·K)
_T_REF = 298.0  # K — reference temperature for solvent database


def get_viscosity(solvent: str, temperature: float) -> float:
    """Return solvent viscosity (Pa·s) at the given temperature (K).

    Applies Arrhenius temperature scaling relative to the 298 K reference
    value.  Raises ``ValueError`` for unknown solvents.
    """
    if solvent not in SOLVENTS:
        available = ', '.join(sorted(SOLVENTS.keys()))
        raise ValueError(
            f"Unknown solvent '{solvent}'. Available: {available}. "
            "Or pass eta directly."
        )
    s = SOLVENTS[solvent]
    eta_ref = s['eta']
    Ea = s.get('Ea', 12000.0)
    # Arrhenius: η(T) = η_ref * exp(Ea/R * (1/T - 1/T_ref))
    return eta_ref * np.exp(Ea / _R_GAS * (1.0 / temperature - 1.0 / _T_REF))


# ---------------------------------------------------------------------------
# Coordinate parsers
# ---------------------------------------------------------------------------

def _get_mass(elem: str) -> float:
    return ATOMIC_MASS.get(elem, 12.0)


def _get_vdw(elem: str) -> float:
    return VDW_RADIUS.get(elem, 1.70)


def parse_xyz(filename: str) -> list[tuple[str, np.ndarray]]:
    """Parse XYZ file → list of (element, position) tuples."""
    atoms: list[tuple[str, np.ndarray]] = []
    with open(filename) as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    for line in lines[2:2 + natoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        elem = re.sub(r'\d+', '', parts[0])
        atoms.append((elem, np.array([float(parts[1]),
                                      float(parts[2]),
                                      float(parts[3])])))
    return atoms


def parse_pdb(filename: str) -> list[tuple[str, np.ndarray]]:
    """Parse PDB file → list of (element, position) tuples."""
    atoms: list[tuple[str, np.ndarray]] = []
    with open(filename) as f:
        for line in f:
            if line[:6] in ('ATOM  ', 'HETATM'):
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                elem = line[76:78].strip()
                if not elem:
                    elem = re.sub(r'\d+', '', line[12:16].strip())
                atoms.append((elem, np.array([x, y, z])))
    return atoms


def load_molecule(filename: str) -> list[tuple[str, np.ndarray]]:
    """Auto-detect format (.xyz or .pdb) and parse molecular coordinates."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.xyz':
        atoms = parse_xyz(filename)
    elif ext == '.pdb':
        atoms = parse_pdb(filename)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .xyz or .pdb.")
    unknown = {e for e, _ in atoms if e not in ATOMIC_MASS}
    if unknown:
        logger.warning("Unknown elements %s — using mass=12, r_vdW=1.70 Å", unknown)
    return atoms


# ---------------------------------------------------------------------------
# Molecular geometry helpers
# ---------------------------------------------------------------------------

def _center_of_mass(atoms: list[tuple[str, np.ndarray]]) -> np.ndarray:
    total_mass = sum(_get_mass(e) for e, _ in atoms)
    com = sum(_get_mass(e) * p for e, p in atoms)
    return com / total_mass


def _spatial_extent_ellipsoid(
    atoms: list[tuple[str, np.ndarray]],
    shell: float = 0.0,
) -> tuple[float, float, float]:
    """Return (a, b, c) semi-axes in Å (a ≥ b ≥ c) including vdW radii."""
    coords = np.array([p for _, p in atoms])
    radii = np.array([_get_vdw(e) + shell for e, _ in atoms])
    center = coords.mean(axis=0)
    centered = coords - center
    _, eigvec = np.linalg.eigh(np.cov(centered.T))
    proj = centered @ eigvec
    semi_axes = [
        (np.max(proj[:, d] + radii) - np.min(proj[:, d] - radii)) / 2.0
        for d in range(3)
    ]
    a, b, c = sorted(semi_axes, reverse=True)
    return a, b, c


def _vdw_volume(atoms: list[tuple[str, np.ndarray]]) -> float:
    return sum((4 / 3) * _pi * _get_vdw(e) ** 3 for e, _ in atoms)


# ---------------------------------------------------------------------------
# Method 1: Perrin ellipsoid
# ---------------------------------------------------------------------------

def _ellipsoid_alpha_integrals(
    a: float, b: float, c: float, N: int = 10000
) -> tuple[float, float, float]:
    log_s = np.linspace(np.log(1e-6), np.log(1e8), N)
    s = np.exp(log_s)
    denom = np.sqrt((a ** 2 + s) * (b ** 2 + s) * (c ** 2 + s))
    alpha_a = np.trapz(s / ((a ** 2 + s) * denom), log_s)
    alpha_b = np.trapz(s / ((b ** 2 + s) * denom), log_s)
    alpha_c = np.trapz(s / ((c ** 2 + s) * denom), log_s)
    return alpha_a, alpha_b, alpha_c


def _perrin_diffusion(
    a: float, b: float, c: float, eta: float, T: float
) -> tuple[float, float, float, float, float]:
    """Return (D_a, D_b, D_c, D_iso, tau_iso) for a triaxial ellipsoid."""
    alpha_a, alpha_b, alpha_c = _ellipsoid_alpha_integrals(a, b, c)
    chi_a = a * b * c * alpha_a
    chi_b = a * b * c * alpha_b
    chi_c = a * b * c * alpha_c
    abc_m3 = a * b * c * 1e-30
    a2, b2, c2 = a ** 2, b ** 2, c ** 2
    xi_a = 16 * _pi * eta * abc_m3 * (b2 + c2) / (3 * (b2 * chi_c + c2 * chi_b))
    xi_b = 16 * _pi * eta * abc_m3 * (a2 + c2) / (3 * (a2 * chi_c + c2 * chi_a))
    xi_c = 16 * _pi * eta * abc_m3 * (a2 + b2) / (3 * (a2 * chi_b + b2 * chi_a))
    D_a = _kB * T / xi_a
    D_b = _kB * T / xi_b
    D_c = _kB * T / xi_c
    D_iso = (D_a + D_b + D_c) / 3
    tau_iso = 1 / (6 * D_iso)
    return D_a, D_b, D_c, D_iso, tau_iso


def run_ellipsoid(
    atoms: list[tuple[str, np.ndarray]],
    eta: float,
    T: float,
    shell: float = 0.0,
) -> dict:
    """Run the Perrin ellipsoid model.

    Args:
        atoms: List of (element, position) tuples (positions in Å).
        eta: Solvent viscosity (Pa·s).
        T: Temperature (K).
        shell: Solvent shell thickness to add to vdW radii (Å).

    Returns:
        Dict with keys ``tau_iso`` (s), ``D_iso`` (rad²/s), ``D`` (array),
        ``axes`` (a, b, c in Å), ``anisotropy`` (D_a/D_c), ``method``.
    """
    a, b, c = _spatial_extent_ellipsoid(atoms, shell=shell)
    D_a, D_b, D_c, D_iso, tau_iso = _perrin_diffusion(a, b, c, eta, T)
    logger.debug(
        "Perrin ellipsoid: a=%.2f b=%.2f c=%.2f Å | "
        "D_iso=%.3e rad²/s | τ_R=%.1f ps",
        a, b, c, D_iso, tau_iso * 1e12,
    )
    return {
        'method': 'ellipsoid',
        'tau_iso': tau_iso,
        'D_iso': D_iso,
        'D': np.array([D_a, D_b, D_c]),
        'axes': (a, b, c),
        'anisotropy': D_a / D_c if D_c > 0 else float('nan'),
    }


# ---------------------------------------------------------------------------
# Method 2: Bead-shell model
# ---------------------------------------------------------------------------

def _fibonacci_sphere(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=float)
    phi = np.arccos(1 - 2 * (idx + 0.5) / n)
    theta = _pi * (1 + 5 ** 0.5) * idx
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


def _generate_surface_beads(
    atoms: list[tuple[str, np.ndarray]],
    sigma: float,
    shell: float = 0.0,
) -> np.ndarray:
    centers = np.array([p for _, p in atoms])
    radii = np.array([_get_vdw(e) + shell for e, _ in atoms])
    all_beads = []
    for i, (R, center) in enumerate(zip(radii, centers)):
        n_pts = max(int(4 * R ** 2 / sigma ** 2), 12)
        pts = center + R * _fibonacci_sphere(n_pts)
        keep = np.ones(len(pts), dtype=bool)
        for j, (Rj, cj) in enumerate(zip(radii, centers)):
            if j == i:
                continue
            keep &= np.linalg.norm(pts - cj, axis=1) >= Rj - 0.1
        all_beads.append(pts[keep])
    beads = np.vstack(all_beads)
    if len(beads) > 1:
        keep = np.ones(len(beads), dtype=bool)
        for i in range(len(beads)):
            if not keep[i]:
                continue
            dists = np.linalg.norm(beads[i + 1:] - beads[i], axis=1)
            keep[np.where(dists < 2 * sigma)[0] + i + 1] = False
        beads = beads[keep]
    return beads


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _build_friction_matrix(
    beads_m: np.ndarray, sigma_m: float, eta: float
) -> np.ndarray:
    N = len(beads_m)
    center = beads_m.mean(axis=0)
    r = beads_m - center
    mu = np.zeros((3 * N, 3 * N))
    I3 = np.eye(3)
    self_mob = 1.0 / (6 * _pi * eta * sigma_m)
    for i in range(N):
        mu[3 * i:3 * i + 3, 3 * i:3 * i + 3] = self_mob * I3
        for j in range(i + 1, N):
            rij = beads_m[j] - beads_m[i]
            dist = np.linalg.norm(rij)
            rhat = rij / dist
            rr = np.outer(rhat, rhat)
            if dist > 2 * sigma_m:
                c = 1.0 / (8 * _pi * eta * dist)
                s2r2 = 2 * sigma_m ** 2 / (3 * dist ** 2)
                mu_ij = c * ((1 + s2r2) * I3 + (1 - 3 * s2r2) * rr)
            else:
                q = dist / (2 * sigma_m)
                mu_ij = self_mob * (
                    (1 - 9 * q / 8 + 3 * q ** 3 / 8) * I3
                    + (3 * q / 8 - 3 * q ** 3 / 8) * rr
                )
            mu[3 * i:3 * i + 3, 3 * j:3 * j + 3] = mu_ij
            mu[3 * j:3 * j + 3, 3 * i:3 * i + 3] = mu_ij
    zeta = np.linalg.inv(mu)
    Xi_rr = np.zeros((3, 3))
    for i in range(N):
        ri_x = _skew(r[i])
        for j in range(N):
            Xi_rr += ri_x @ zeta[3 * i:3 * i + 3, 3 * j:3 * j + 3] @ _skew(r[j]).T
    return Xi_rr


def run_beadshell(
    atoms: list[tuple[str, np.ndarray]],
    eta: float,
    T: float,
    sigma: float = 0.6,
    shell: float = 0.0,
) -> dict:
    """Run the bead-shell hydrodynamic model.

    Args:
        atoms: List of (element, position) tuples (positions in Å).
        eta: Solvent viscosity (Pa·s).
        T: Temperature (K).
        sigma: Minibead radius (Å).
        shell: Solvent shell thickness (Å).

    Returns:
        Dict with keys ``tau_iso``, ``D_iso``, ``D``, ``n_beads``,
        ``anisotropy``, ``method``.  Returns ``None`` if too few beads.
    """
    beads = _generate_surface_beads(atoms, sigma, shell=shell)
    N = len(beads)
    if N < 10:
        logger.warning("Bead-shell: only %d surface beads — try smaller σ.", N)
        return None
    Xi_rr = _build_friction_matrix(beads * 1e-10, sigma * 1e-10, eta)
    D_rot = _kB * T * np.linalg.inv(Xi_rr)
    D_eig = np.sort(np.linalg.eigvalsh(D_rot))[::-1]
    D_iso = D_eig.mean()
    tau_iso = 1 / (6 * D_iso)
    logger.debug(
        "Bead-shell (σ=%.2f Å, N=%d): D_iso=%.3e rad²/s | τ_R=%.1f ps",
        sigma, N, D_iso, tau_iso * 1e12,
    )
    return {
        'method': 'beadshell',
        'tau_iso': tau_iso,
        'D_iso': D_iso,
        'D': D_eig,
        'n_beads': N,
        'anisotropy': D_eig[0] / D_eig[2] if D_eig[2] > 0 else float('nan'),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_tau_r(
    xyz_file: str,
    temperature: float,
    *,
    solvent: str | None = None,
    eta: float | None = None,
    method: str = 'ellipsoid',
    shell: float = 0.0,
    sigma: float = 0.6,
) -> dict:
    """Compute the rotational correlation time τ_R from molecular coordinates.

    Args:
        xyz_file: Path to .xyz or .pdb coordinate file.
        temperature: Temperature in K.
        solvent: Solvent name from the built-in database.  Ignored when
            ``eta`` is provided.
        eta: Solvent viscosity (Pa·s).  Overrides ``solvent``.
        method: ``"ellipsoid"`` (default) or ``"beadshell"``.
        shell: Solvent shell thickness to add to the molecular surface (Å).
        sigma: Minibead radius for the bead-shell method (Å).

    Returns:
        Result dict from :func:`run_ellipsoid` or :func:`run_beadshell`,
        with an additional ``'eta'`` key recording the viscosity used.

    Raises:
        ValueError: If neither ``solvent`` nor ``eta`` is provided, or if
            the solvent name is unknown.
    """
    if eta is None:
        if solvent is None:
            raise ValueError(
                "Either 'solvent' or 'eta' must be provided to compute_tau_r."
            )
        eta = get_viscosity(solvent, temperature)

    atoms = load_molecule(xyz_file)

    if method == 'ellipsoid':
        result = run_ellipsoid(atoms, eta, temperature, shell=shell)
    elif method == 'beadshell':
        result = run_beadshell(atoms, eta, temperature, sigma=sigma, shell=shell)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'ellipsoid' or 'beadshell'.")

    if result is not None:
        result['eta'] = eta
        result['temperature'] = temperature
    return result
