# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""Utility helpers for SimpNMR.

This module provides constants, parsing helpers, formatting utilities, and
relaxation-rate helper functions used across the package.
"""

import math
import re
import sys
from os import PathLike

import numpy as np
import scipy.constants as consts
from numpy.typing import NDArray

from .scripts.coords_tools import label_format as lf

# Physical constants
MU0 = consts.physical_constants["vacuum mag. permeability"][0]  # [N A^-2]
MUB = consts.physical_constants["Bohr magneton"][0]
HBAR = consts.hbar  # [J s radian-1]
H = consts.h  # [J s radian-1]
KB = consts.physical_constants["Boltzmann constant"][0]  # Boltzmann constant k [J·K⁻¹]
GE = abs(consts.physical_constants["electron g factor"][0])  # g value of free electron
EGAMMA = consts.physical_constants["electron gyromag. ratio in MHz/T"][0]


# Values from easyspin, most abundant isotope taken
# unless otherwise stated
NUCLEAR_GAMMAS = {  # MHz / T
    "H": 42.57747844,
    "He": 0,
    "Li": 16.54827639,
    "Be": -5.983354553,
    "B": 13.6629846,
    "C": 10.70839886,  # 13C
    "N": 3.077705864,
    "O": 0,
    "F": 40.07758282,
    "Ne": 0,
    "Na": 11.26884545,
    "Mg": 0,
    "Al": 11.10309064,
    "Si": 0,
    "P": 17.25145299,
    "S": 0,
    "Cl": 4.176542315,
    "Ar": 0,
    "K": 1.98934438,
    "Ca": 0,
    "Sc": 10.35902797,
    "Ti": 0,
    "V": 11.21329199,
    "Cr": 0,
    "Mn": 10.52908802,
    "Co": 10.07706825,
    "Ni": 0,
    "Cu": 11.2997322,
    "Zn": 0,
    "Ga": 13.0207613,
    "Ge": 0,
    "As": 7.31502159,
    "Se": 0,
    "Br": 10.70415612,
    "Kr": 0,
    "Rb": 4.125286474,
    "Sr": 0,
    "Y": -2.094923395,
    "Zr": 0,
    "Nb": 10.45209983,
    "Mo": 0,
    "Tc": 9.628859764,
    "Ru": 0,
    "Rh": -1.347674483,
    "Pd": 0,
    "Ag": -1.731395826,
    "Cd": 0,
    "In": 9.38569904,
    "Sn": 0,
    "Sb": 10.25543693,
    "Te": 0,
    "I": 8.577780384,
    "Xe": 0,
    "Cs": 5.623350147,
    "Ba": 0,
    "La": 6.06115074,
    "Ce": 0,
    "Pr": 13.03615894,
    "Nd": 0,
    "Pm": 5.617851208,
    "Sm": 0,
    "Eu": 4.675698685,
    "Gd": 0,
    "Tb": 10.2371427,
    "Dy": 0,
    "Ho": 12.7144855,
    "Er": 0,
    "Tm": -3.521638071,
    "Yb": 0,
    "Lu": 4.86168996,
    "Hf": 0,
    "Ta": 5.162706167,
    "W": 0,
    "Re": 9.817137817,
    "Os": 0,
    "Ir": 0.831624921,
    "Pt": 0,
    "Au": 0.740641648,
    "Hg": 0,
    "Tl": 24.97488703,
    "Pb": 0,
    "Bi": 6.962476653,
}

DEFAULT_ISOTOPES = {
    "H": "1H",
    "C": "13C",
    "P": "31P",
    "N": "15N",
    "Si": "29Si",
    "B": "10B",
    "Li": "6Li",
}

OTHER_ISOTOPES = ["2H"]

SUPPORTED_ISOTOPES = list(DEFAULT_ISOTOPES.values()) + OTHER_ISOTOPES


def a_tensor_mhz_to_angstrom(a_tensors: dict[str, NDArray]) -> dict[str, NDArray]:
    """Converts hyperfine A tensors from MHz to ``ppm Å^-3``.

    Uses the gyromagnetic ratio of each nucleus (looked up from `NUCLEAR_GAMMAS`).

    Args:
        a_tensors: Mapping from atom label (with global index, e.g. ``"H34"``) to a
            ``(3, 3)`` hyperfine tensor in MHz.

    Returns:
        Mapping from atom label to hyperfine tensor in ``ppm Å^-3``. Labels whose
        element has no gamma defined (gamma=0) are omitted.
    """

    a_tensors_ang = {
        key: _mhz_to_angstrom(val, NUCLEAR_GAMMAS[lf.remove_numbers(key)])
        for key, val in a_tensors.items()
        if lf.remove_numbers(key) in NUCLEAR_GAMMAS.keys()
        and NUCLEAR_GAMMAS[lf.remove_numbers(key)]  # noqa
    }

    return a_tensors_ang


def _mhz_to_angstrom(val_mhz: NDArray | float, nuclear_gamma: float) -> NDArray | float:  # noqa
    """Converts a hyperfine coupling value from MHz to ``ppm Å^-3``.

    Args:
        val_mhz: Hyperfine tensor as a ``(3, 3)`` array or an isotropic value in MHz.
        nuclear_gamma: Nuclear gyromagnetic ratio for the nucleus (MHz/T).

    Returns:
        The converted value in ``ppm Å^-3`` with the same shape as `val_mhz`.
    """

    val_mhz = np.asarray(val_mhz)

    # Conversion factor for MHz to ppm Angstrom^-3
    val = 1e-18 / (H * EGAMMA * nuclear_gamma * 1e12 * MU0)

    val_ang = val_mhz * val

    return val_ang


def flatten(biglist: list) -> list:
    """Flattens a nested list by one level.

    Args:
        biglist: A list of lists.

    Returns:
        A single list containing the concatenated elements.
    """
    return [item for sublist in biglist for item in sublist]


def find_mean_values(values: list[float], thresh: float = 0.1) -> list[int]:
    """Finds indices where a 1D sequence changes by at least a threshold.

    Args:
        values: Values to analyze.
        thresh: Threshold for identifying a step change.

    Returns:
        Indices at which the step size ``abs(diff(values))`` is greater than or equal
        to `thresh`.
    """

    # Find values for which step size is >= thresh
    mask = np.abs(np.diff(values)) >= thresh
    # and mark indices at which to split
    split_indices = np.where(mask)[0] + 1

    return split_indices.tolist()


def comp2ind(comp_str: str) -> list[int]:
    """Converts a tensor component label into matrix indices.

    Args:
        comp_str: Component string, e.g. ``"xy"``.

    Returns:
        A tuple ``(row, col)`` for the corresponding element of a ``(3, 3)`` tensor.
    """

    _c2i = {
        "xx": [0, 0],
        "xy": [0, 1],
        "xz": [0, 2],
        "yx": [1, 0],
        "yy": [1, 1],
        "yz": [1, 2],
        "zx": [2, 0],
        "zy": [2, 1],
        "zz": [2, 2],
    }

    return _c2i[comp_str][0], _c2i[comp_str][1]


def cstr(string: str, color: str):
    """Applies ANSI color codes to a string.

    Args:
        string: String to colorize.
        color: Color name. Supported values are ``red``, ``green``, ``yellow``,
            ``blue``, ``magenta``, ``cyan``, ``white``, ``black_yellowbg``, and
            ``black_bluebg``.

    Returns:
        The input string wrapped in ANSI escape codes.
    """

    ccodes = {
        "red": "\u001b[31m",
        "green": "\u001b[32m",
        "yellow": "\u001b[33m",
        "blue": "\u001b[34m",
        "magenta": "\u001b[35m",
        "cyan": "\u001b[36m",
        "white": "\u001b[37m",
        "black_yellowbg": "\u001b[30;43m\u001b[K",
        "black_bluebg": "\u001b[30;44m\u001b[K",
    }
    end = "\033[0m\u001b[K"

    # Count newlines at neither beginning nor end
    num_c_nl = string.rstrip("\n").lstrip("\n").count("\n")

    # Remove right new lines to count left new lines
    num_l_nl = string.rstrip("\n").count("\n") - num_c_nl
    l_nl = "".join(["\n"] * num_l_nl)

    # Remove left new lines to count right new lines
    num_r_nl = string.lstrip("\n").count("\n") - num_c_nl
    r_nl = "".join(["\n"] * num_r_nl)

    # Remove left and right newlines, will add in again later
    _string = string.rstrip("\n").lstrip("\n")

    out = "{}{}{}{}{}".format(l_nl, ccodes[color], _string, end, r_nl)

    return out


def can_float(s: str) -> bool:
    """Returns whether a string can be parsed as a float."""
    out = True
    try:
        s = float(s.strip())
    except ValueError:
        out = False

    return out


def cprint(string: str, color: str):
    """Prints a colorized string to stdout.

    Args:
        string: String to print.
        color: Color name supported by `cstr`.

    Returns:
        None.
    """
    return print(cstr(string, color))


def red_exit(string: str) -> None:
    """Prints an error message in red and exits with code ``-1``.

    Args:
        string: Message to print.

    Returns:
        None.
    """
    cprint(string, "red")
    sys.exit(-1)
    return


def read_exp_metadata(file_name: str) -> tuple[float, float, str]:
    """Reads metadata from an experiment CSV file.

    Metadata is stored as single comment lines beginning with ``#`` and formatted as
    ``name value``. Supported keys are ``temperature``, ``magnetic_field``, and
    ``isotope``.

    Args:
        file_name: Path to the experiment file.

    Returns:
        A tuple ``(temperature, magnetic_field, isotope)`` where temperature is in K,
        magnetic field is in T, and isotope is formatted like ``"1H"`` or ``"13C"``.

    Raises:
        IndexError: If a required metadata line is missing.
        ValueError: If a numeric metadata value cannot be parsed.
    """

    temperature, magnetic_field, isotope = None, None, None

    temperature = float(
        find_first_group(file_name, r"# *temperature (\d*\.*\d*)", re.IGNORECASE)
    )

    magnetic_field = float(
        find_first_group(file_name, r"# *magnetic_field (\d*\.*\d*)", re.IGNORECASE)
    )

    isotope = str(
        find_first_group(file_name, r"# *isotope (\d{0,3}[A-Za-z]{0,2})", re.IGNORECASE)
    )

    return temperature, magnetic_field, isotope


def find_index_of_nearest(array, value):
    """Returns the index of the nearest value in a sorted array."""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):  # noqa
        return idx - 1
    else:
        return idx


def find_first_group(
    file_name: str | PathLike[str],
    pattern: str,
    flags: int = 0,
) -> str:
    """Return the first captured group for the first regex match in a text file

    The function scans the file line by line and applies `pattern` using `re.search`
    It returns group 1 from the first match, so `pattern` must contain at least one
    capturing group in parentheses

    Args:
        file_name: Path to the text file to scan
        pattern: Regular expression pattern with at least one capturing group
        flags: Regex flags passed to `re.compile`, e.g. `re.IGNORECASE`

    Returns:
        The first captured group from the first matching line

    Raises:
        ValueError: If no matching line is found
        IndexError: If the pattern matches but has no capturing group 1
    """
    rx = re.compile(pattern, flags)

    with open(file_name, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = rx.search(line)
            if m:
                return m.group(1)

    raise ValueError(f"No relevant data found in {file_name} for pattern: {pattern}")


def isotope_format(isotope_string: str) -> str:
    r"""Formats an isotope label as Matplotlib mathtext.

    Args:
        isotope_string: Isotope label, e.g. ``"1H"`` or ``"13C"``.

    Returns:
        A mathtext string, e.g. ``$^\mathregular{13} \mathregular{C}$``.
    """

    # Split at number letter boundary
    for it, char in enumerate(isotope_string):
        if char not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            split_at = it
            break
    nums = isotope_string[:split_at]
    lets = isotope_string[split_at:]

    return r"$^\mathregular{{{}}} \mathregular{{{}}}$".format(nums, lets)


def calc_g_eff(spin: float, orbit: float, total_momentum_J: float | None):
    """Computes an effective electron g-factor.

    For spin-only systems (transition metals, organic radicals) where no total J is
    defined, this returns the free-electron g value `GE`.

    For systems with well-defined ``L``, ``S``, and ``J`` (e.g. lanthanides), this
    returns the Landé ``g_J`` factor.

    Args:
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum quantum number ``J``. If ``None`` or
            ``0``, the function falls back to `GE`.

    Returns:
        Effective g-factor (either `GE` or ``g_J``).
    """

    # Spin-only case: no total J provided or explicitly zero
    if total_momentum_J is None or total_momentum_J == 0.0:
        return GE

    # Landé g_J expression using S, L and J
    J = float(total_momentum_J)

    return 1.5 + (spin * (spin + 1) - orbit * (orbit + 1)) / (2.0 * J * (J + 1))


def choose_S_eff(spin: float, total_momentum_J: float | None):
    """Returns the effective angular momentum quantum number used in prefactors.

    Args:
        spin: Spin quantum number ``S``.
        total_momentum_J: Total angular momentum ``J``. If ``None``, `spin` is used.

    Returns:
        ``S`` for spin-only systems, or ``J`` when `total_momentum_J` is provided.
    """

    return spin if total_momentum_J is None else total_momentum_J


def get_spin_only_susceptibility(
    spin: float, orbit: float, total_momentum_J: float | None, temperature: float
) -> float:
    """Computes the spin-only isotropic molar susceptibility in Å³.

    Uses the Curie law with an effective g-factor and an effective angular momentum
    quantum number (``S`` for spin-only systems, ``J`` when defined).

    Args:
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum ``J`` or ``None`` for spin-only.
        temperature: Temperature in Kelvin.

    Returns:
        Spin-only isotropic molar susceptibility in Å³.
    """

    # Landé g-factor uses S, L, J
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)

    # Effective moment quantum number for Curie law:
    # S for transition metals, J for lanthanides
    S_eff = choose_S_eff(spin, total_momentum_J)

    # Chi (SI, m^3 mol^-1)
    chi_only_iso_SI = (
        MU0 * MUB**2 * g_eff**2 * S_eff * (S_eff + 1) / (3 * KB * temperature)
    )

    # Convert m^3 to Å^3: 1 Å^3 = 1e-30 m^3
    chi_only_iso = chi_only_iso_SI * 1e30

    return chi_only_iso


def get_true_iso_susceptibility(
    spin: float,
    orbit: float,
    g_tensor: NDArray,
    chi_tensors: dict[float, NDArray],
    total_momentum_J: float | None,
) -> float:
    """Computes a g-tensor-corrected isotropic susceptibility in Å³.

    Uses susceptibility principal components (from `chi_tensors`) and the supplied
    g-tensor to compute an effective isotropic value.

    Args:
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        g_tensor: g-tensor as a ``(3, 3)`` array.
        chi_tensors: chi tensors in A^3.
        total_momentum_J: Total angular momentum ``J`` or ``None``.

    Returns:
        Corrected isotropic susceptibility in A^3.

    """

    # Use Landé g_J (or GE) to get an effective g-factor
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)

    # Trace-based expression with g correction (cm^3 mol^-1)
    chi_true_iso = g_eff / 3.0 * np.trace(chi_tensors * np.linalg.inv(g_tensor.T))

    return chi_true_iso


def sbm_r1_dipolar(
    nuclei_labels,
    nuclei_coords,
    electron_coords,
    gamma_I_dict,
    omega_I_dict,
    omega_S,
    tau_c1,
    tau_c2,
    spin,
    orbit,
    total_momentum_J,
):
    """Computes SBM dipolar R1 relaxation rates for each nucleus.

    Implements the Solomon-Bloembergen-Morgan expression for the longitudinal
    relaxation rate due to electron-nucleus dipolar interactions.

    Args:
        nuclei_labels: Labels of nuclei for which rates are computed.
        nuclei_coords: Mapping from label to Cartesian coordinates (Å).
        electron_coords: Cartesian coordinates (Å) of the effective electron center.
        gamma_I_dict: Nuclear gyromagnetic ratios (rad s^-1 T^-1) by label.
        omega_I_dict: Nuclear Larmor angular frequencies (rad s^-1) by label.
        omega_S: Electron Larmor angular frequency (rad s^-1).
        tau_c1: Correlation time for the dipolar interaction (s).
        tau_c2: Correlation time entering cross terms (s).
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum quantum number ``J`` or ``None``.

    Returns:
        Mapping from nucleus label to dipolar R1 relaxation rate (s^-1).
    """

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    # Effective g-factor and angular momentum entering the prefactor
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R1 rates
    for label in nuclei_labels:
        r = np.linalg.norm(nuclei_coords[label] - electron_coords) * 1e-10
        gamma_I = gamma_I_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (
            (1 / 10)
            * (1 / r**6)
            * (MU0 / (4 * np.pi)) ** 2
            * (gamma_I * g_eff * MUB) ** 2
            * S_eff
            * (S_eff + 1)
        )
        spectral_density = (
            3 * J(omega_I, tau_c1)
            + 6 * J(omega_I + omega_S, tau_c2)
            + J(omega_I - omega_S, tau_c2)
        )
        rate = prefactor * spectral_density
        rates[label] = rate

    return rates


def sbm_r2_dipolar(
    nuclei_labels,
    nuclei_coords,
    electron_coords,
    gamma_I_dict,
    omega_I_dict,
    omega_S,
    tau_c1,
    tau_c2,
    spin,
    orbit,
    total_momentum_J,
):
    """Computes SBM dipolar R2 relaxation rates for each nucleus.

    Implements the Solomon-Bloembergen-Morgan expression for the transverse
    relaxation rate due to electron-nucleus dipolar interactions.

    Args:
        nuclei_labels: Labels of nuclei for which rates are computed.
        nuclei_coords: Mapping from label to Cartesian coordinates (Å).
        electron_coords: Cartesian coordinates (Å) of the effective electron center.
        gamma_I_dict: Nuclear gyromagnetic ratios (rad s^-1 T^-1) by label.
        omega_I_dict: Nuclear Larmor angular frequencies (rad s^-1) by label.
        omega_S: Electron Larmor angular frequency (rad s^-1).
        tau_c1: Correlation time for the dipolar interaction (s).
        tau_c2: Correlation time entering cross terms (s).
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum quantum number ``J`` or ``None``.

    Returns:
        Mapping from nucleus label to dipolar R2 relaxation rate (s^-1).
    """

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    # Effective g-factor and angular momentum entering the prefactor
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R2 rates
    for label in nuclei_labels:
        r = np.linalg.norm(nuclei_coords[label] - electron_coords) * 1e-10
        gamma_I = gamma_I_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (
            (1 / 15)
            * (1 / r**6)
            * (MU0 / (4 * np.pi)) ** 2
            * (gamma_I * g_eff * MUB) ** 2
            * S_eff
            * (S_eff + 1)
        )
        spectral_density = (
            4 * J(0, tau_c1)
            + 3 * J(omega_I, tau_c1)
            + 6 * J(omega_S, tau_c2)
            + 6 * J(omega_I + omega_S, tau_c2)
            + J(omega_I - omega_S, tau_c2)
        )
        rate = prefactor * spectral_density
        rates[label] = rate

    return rates


def sbm_r1_contact(
    nuclei_labels, Aiso_dict, omega_I_dict, omega_S, tau_e2, spin, total_momentum_J
):
    """Computes SBM contact R1 relaxation rates for each nucleus.

    Implements the Solomon-Bloembergen-Morgan expression for the longitudinal
    relaxation rate due to isotropic Fermi-contact hyperfine coupling.

    Args:
        nuclei_labels: Labels of nuclei for which rates are computed.
        Aiso_dict: Isotropic hyperfine couplings ``A_iso`` (angular frequency units)
            by label.
        omega_I_dict: Nuclear Larmor angular frequencies (rad s^-1) by label.
        omega_S: Electron Larmor angular frequency (rad s^-1).
        tau_e2: Electronic correlation time entering the spectral density (s).
        spin: Spin quantum number ``S``.
        total_momentum_J: Total angular momentum ``J`` or ``None``.

    Returns:
        Mapping from nucleus label to contact R1 relaxation rate (s^-1).
    """

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    # Effective angular momentum quantum number for the contact term
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R1 contact rates
    for label in nuclei_labels:
        Aiso = Aiso_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (2 / 3) * Aiso**2 * S_eff * (S_eff + 1)
        spectral_density = J(omega_I - omega_S, tau_e2)
        rate = prefactor * spectral_density
        rates[label] = rate
    return rates


def sbm_r2_contact(
    nuclei_labels,
    Aiso_dict,
    omega_I_dict,
    omega_S,
    tau_e1,
    tau_e2,
    spin,
    total_momentum_J,
):
    """Computes SBM contact R2 relaxation rates for each nucleus.

    Implements the Solomon-Bloembergen-Morgan expression for the transverse
    relaxation rate due to isotropic Fermi-contact hyperfine coupling.

    Args:
        nuclei_labels: Labels of nuclei for which rates are computed.
        Aiso_dict: Isotropic hyperfine couplings ``A_iso`` (angular frequency units)
            by label.
        omega_I_dict: Nuclear Larmor angular frequencies (rad s^-1) by label.
        omega_S: Electron Larmor angular frequency (rad s^-1).
        tau_e1: Electronic correlation time entering the zero-frequency term (s).
        tau_e2: Electronic correlation time entering the finite-frequency term (s).
        spin: Spin quantum number ``S``.
        total_momentum_J: Total angular momentum ``J`` or ``None``.

    Returns:
        Mapping from nucleus label to contact R2 relaxation rate (s^-1).
    """

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    # Effective angular momentum quantum number for the contact term
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R2 contact rates
    for label in nuclei_labels:
        Aiso = Aiso_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (1 / 3) * Aiso**2 * S_eff * (S_eff + 1)
        spectral_density = J(0, tau_e1) + J(omega_I - omega_S, tau_e2)
        rate = prefactor * spectral_density
        rates[label] = rate
    return rates


def gueron_r1_curie(
    nuclei_labels,
    nuclei_coords,
    electron_coords,
    omega_I_dict,
    T,
    tau_R,
    spin,
    orbit,
    total_momentum_J,
):
    """Computes Guéron Curie R1 relaxation rates for each nucleus.

    Implements the Guéron expression for longitudinal relaxation due to the Curie
    (static) dipolar interaction in the point-dipole approximation.

    Args:
        nuclei_labels: Labels of nuclei for which rates are computed.
        nuclei_coords: Mapping from label to Cartesian coordinates (Å).
        electron_coords: Cartesian coordinates (Å) of the effective electron center.
        omega_I_dict: Nuclear Larmor angular frequencies (rad s^-1) by label.
        T: Temperature in Kelvin.
        tau_R: Rotational correlation time (s).
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum quantum number ``J`` or ``None``.

    Returns:
        Mapping from nucleus label to Curie R1 relaxation rate (s^-1).
    """

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    # Effective g-factor and angular momentum entering the Curie term
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R1 Curie rates
    for label in nuclei_labels:
        r = np.linalg.norm(nuclei_coords[label] - electron_coords) * 1e-10
        omega_I = omega_I_dict[label]
        prefactor = (
            (2 / 5)
            * (1 / r**6)
            * (MU0 / (4 * np.pi)) ** 2
            * (omega_I / (3 * consts.k * T)) ** 2
            * (g_eff * MUB) ** 4
            * (S_eff * (S_eff + 1)) ** 2
        )
        spectral_density = 3 * J(omega_I, tau_R)
        rate = prefactor * spectral_density
        rates[label] = rate

    return rates


def gueron_r2_curie(
    nuclei_labels,
    nuclei_coords,
    electron_coords,
    omega_I_dict,
    T,
    tau_R,
    spin,
    orbit,
    total_momentum_J,
):
    """Computes Guéron Curie R2 relaxation rates for each nucleus.

    Implements the Guéron expression for transverse relaxation due to the Curie
    (static) dipolar interaction in the point-dipole approximation.

    Args:
        nuclei_labels: Labels of nuclei for which rates are computed.
        nuclei_coords: Mapping from label to Cartesian coordinates (Å).
        electron_coords: Cartesian coordinates (Å) of the effective electron center.
        omega_I_dict: Nuclear Larmor angular frequencies (rad s^-1) by label.
        T: Temperature in Kelvin.
        tau_R: Rotational correlation time (s).
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum quantum number ``J`` or ``None``.

    Returns:
        Mapping from nucleus label to Curie R2 relaxation rate (s^-1).
    """

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    # Effective g-factor and angular momentum entering the Curie term
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R2 Curie rates
    for label in nuclei_labels:
        r = np.linalg.norm(nuclei_coords[label] - electron_coords) * 1e-10
        omega_I = omega_I_dict[label]
        prefactor = (
            (1 / 5)
            * (1 / r**6)
            * (MU0 / (4 * np.pi)) ** 2
            * (omega_I / (3 * consts.k * T)) ** 2
            * (g_eff * MUB) ** 4
            * (S_eff * (S_eff + 1)) ** 2
        )
        spectral_density = 4 * J(0, tau_R) + 3 * J(omega_I, tau_R)
        rate = prefactor * spectral_density
        rates[label] = rate

    return rates
