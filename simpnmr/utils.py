
"""
This module contains utility objects and methods
"""

import numpy as np
from numpy.typing import NDArray
import scipy.constants as consts
import sys
from extto.core import find_lines
import math
import re
from scipy import constants
from collections import defaultdict
from . import string_tools as st
from . import readers as rdrs
from . import inputs as inps

# Physical constants
MU0 = consts.physical_constants["vacuum mag. permeability"][0]  # [N A^-2]
MUB = consts.physical_constants["Bohr magneton"][0]
HBAR = consts.hbar  # [J s radian-1]
H = consts.h  # [J s radian-1]
KB = 1.380649e-23  # Boltzmann constant k [J·K⁻¹]
GE = 2.002319  # g value of free electron
EGAMMA = consts.physical_constants["electron gyromag. ratio in MHz/T"][0]


# Values from easyspin, most abundant isotope taken
# unless otherwise stated
NUCLEAR_GAMMAS = {  # MHz / T
    'H': 42.57747844,
    'He': 0,
    'Li': 16.54827639,
    'Be': -5.983354553,
    'B': 13.6629846,
    'C': 10.70839886,  # 13C
    'N': 3.077705864,
    'O': 0,
    'F': 40.07758282,
    'Ne': 0,
    'Na': 11.26884545,
    'Mg': 0,
    'Al': 11.10309064,
    'Si': 0,
    'P': 17.25145299,
    'S': 0,
    'Cl': 4.176542315,
    'Ar': 0,
    'K': 1.98934438,
    'Ca': 0,
    'Sc': 10.35902797,
    'Ti': 0,
    'V': 11.21329199,
    'Cr': 0,
    'Mn': 10.52908802,
    'Co': 10.07706825,
    'Ni': 0,
    'Ni': 0,
    'Cu': 11.2997322,
    'Zn': 0,
    'Ga': 13.0207613,
    'Ge': 0,
    'As': 7.31502159,
    'Se': 0,
    'Br': 10.70415612,
    'Kr': 0,
    'Rb': 4.125286474,
    'Sr': 0,
    'Y': -2.094923395,
    'Zr': 0,
    'Nb': 10.45209983,
    'Mo': 0,
    'Tc': 9.628859764,
    'Ru': 0,
    'Rh': -1.347674483,
    'Pd': 0,
    'Ag': -1.731395826,
    'Cd': 0,
    'In': 9.38569904,
    'Sn': 0,
    'Sb': 10.25543693,
    'Te': 0,
    'I': 8.577780384,
    'Xe': 0,
    'Cs': 5.623350147,
    'Ba': 0,
    'La': 6.06115074,
    'Ce': 0,
    'Pr': 13.03615894,
    'Nd': 0,
    'Pm': 5.617851208,
    'Sm': 0,
    'Eu': 4.675698685,
    'Gd': 0,
    'Tb': 10.2371427,
    'Dy': 0,
    'Ho': 12.7144855,
    'Er': 0,
    'Tm': -3.521638071,
    'Yb': 0,
    'Lu': 4.86168996,
    'Hf': 0,
    'Ta': 5.162706167,
    'W': 0,
    'Re': 9.817137817,
    'Os': 0,
    'Ir': 0.831624921,
    'Pt': 0,
    'Au': 0.740641648,
    'Hg': 0,
    'Tl': 24.97488703,
    'Pb': 0,
    'Bi': 6.962476653
}

DEFAULT_ISOTOPES = {
    'H': '1H',
    'C': '13C',
    'P': '31P',
    'N': '15N',
    'Si': '29Si',
    'B': '10B',
    'Li': '6Li',
}

OTHER_ISOTOPES = [
    '2H'
]

SUPPORTED_ISOTOPES = list(DEFAULT_ISOTOPES.values()) + OTHER_ISOTOPES


def a_tensor_mhz_to_angstrom(a_tensors: dict[str: NDArray]) -> dict[
        str: NDArray]:
    """
    Converts A tensor from MHz to ppm angstrom^-3 using gyromagnetic ratio of
    given nucleus

    Parameters
    ----------
    a_tensors: dict[str: np.array]
        Key is atomic label with global (1...N_total) indexing number
        (e.g key=H34), and value is raw A tensor as 3x3 np.array in units of
        MHz

    Returns
    -------
    dict[str: np.array]
        Key is atomic label with global (1...N_total) indexing number
        (e.g key=H34), and value is raw A tensor as 3x3 np.array in units of
        Angstrom^-3
    """

    a_tensors_ang = {
        key: _mhz_to_angstrom(val, NUCLEAR_GAMMAS[st.remove_numbers(key)])
        for key, val in a_tensors.items()
        if st.remove_numbers(key) in NUCLEAR_GAMMAS.keys() and NUCLEAR_GAMMAS[st.remove_numbers(key)]  # noqa
    }

    return a_tensors_ang


def _mhz_to_angstrom(val_mhz: NDArray | float, nuclear_gamma: float) -> NDArray | float:  # noqa
    """
    Converts A tensor in MHz to ppm Angstrom^-3 using specified nuclear
    gyromagnetic ratio

    Parameters
    ----------
    val_mhz: array_like | float
        3x3 array containing A tensor, or isotropic A value in MHz
    nuclear_gamm: float
        Nuclear gyromagnetic ratio for current nucleus in MHz/T

    Returns
    -------
    ndarray of floats | float
        3x3 array containing A tensor, or isotropic A value in ppm Angstrom^-3
    """

    val_mhz = np.asarray(val_mhz)

    # Conversion factor for MHz to ppm Angstrom^-3
    val = 1E-18 / (H * EGAMMA * nuclear_gamma * 1E12 * MU0)

    val_ang = val_mhz * val

    return val_ang


def flatten(biglist: list) -> list:
    '''
    Recursively flattens list

    Parameters
    ----------
    biglist: list[list]

    Returns
    -------
    list
        Flattened list
    '''
    return [item for sublist in biglist for item in sublist]


def find_mean_values(values: list[float], thresh: float = 0.1) -> list[int]:
    '''
    Finds mean value from a list of values by locating values for which
    step size is >= `thresh`

    Returns list of same length with all values replaced by mean(s)

    Parameters
    ----------
    values: list[float]
        Values to look at
    thresh: float, default 0.1
        Threshold used to discriminate between values

    Returns
    -------
    list[int]
        indices of original list at which value changes by more than
        0.1
    '''

    # Find values for which step size is >= thresh
    mask = np.abs(np.diff(values)) >= thresh
    # and mark indices at which to split
    split_indices = np.where(mask)[0] + 1

    return split_indices.tolist()


def comp2ind(comp_str: str) -> list[int]:
    '''
    Convert component string to element indices of 3x3 tensor

    Parameters
    ----------
    comp_str: str
        Component string e.g. xy

    Returns
    -------
    list[int]
        row and column index of component
    '''

    _c2i = {
        'xx': [0, 0],
        'xy': [0, 1],
        'xz': [0, 2],
        'yx': [1, 0],
        'yy': [1, 1],
        'yz': [1, 2],
        'zx': [2, 0],
        'zy': [2, 1],
        'zz': [2, 2],
    }

    return _c2i[comp_str][0], _c2i[comp_str][1]


def cstr(string: str, color: str):
    '''
    Produces colorised string

    Parameters
    ----------
    string: str
        String to print
    color: str {'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'black_yellowbg', 'black_bluebg'}
        String name of color

    Returns
    -------
    str
        Input string with colours
    '''  # noqa

    ccodes = {
        'red': '\u001b[31m',
        'green': '\u001b[32m',
        'yellow': '\u001b[33m',
        'blue': '\u001b[34m',
        'magenta': '\u001b[35m',
        'cyan': '\u001b[36m',
        'white': '\u001b[37m',
        'black_yellowbg': '\u001b[30;43m\u001b[K',
        'black_bluebg': '\u001b[30;44m\u001b[K',
    }
    end = '\033[0m\u001b[K'

    # Count newlines at neither beginning nor end
    num_c_nl = string.rstrip('\n').lstrip('\n').count('\n')

    # Remove right new lines to count left new lines
    num_l_nl = string.rstrip('\n').count('\n') - num_c_nl
    l_nl = ''.join(['\n'] * num_l_nl)

    # Remove left new lines to count right new lines
    num_r_nl = string.lstrip('\n').count('\n') - num_c_nl
    r_nl = ''.join(['\n'] * num_r_nl)

    # Remove left and right newlines, will add in again later
    _string = string.rstrip('\n').lstrip('\n')

    out = '{}{}{}{}{}'.format(l_nl, ccodes[color], _string, end, r_nl)

    return out


def can_float(s: str) -> bool:
    '''
    For a given string, checks if conversion to float is possible

    Parameters
    ----------
    s: str
        string to check

    Returns
    -------
    bool
        True if value can be converted to float
    '''
    out = True
    try:
        s = float(s.strip())
    except ValueError:
        out = False

    return out


def cprint(string: str, color: str):
    '''
    Prints colored output to screen

    Parameters
    ----------
    string: str
        String to print
    color: str {'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'black_yellowbg', 'black_bluebg'}
        String name of color

    Returns
    -------
    None
    '''  # noqa

    return print(cstr(string, color))


def red_exit(string: str) -> None:
    '''
    Prints a red string and then exits with return code of -1

    Parameters
    ----------
    string: str
        String to print
    '''
    cprint(string, 'red')
    sys.exit(-1)
    return


def read_exp_metadata(file_name: str) -> tuple[float, float, str]:
    '''
    Reads metadata from experiment files. Metadata is stored as single lines\n
    beginning with comment character # and formatted as\n
    NAME=VALUE
    where NAME is one of temperature, magnetic_field, or isotope

    Parameters
    ----------
    file_name: str
        File to read

    Returns
    -------
    float
        Temperature in Kelvin
    float
        Magnetic field in Tesla
    str
        Isotope symbol formatted as nucleon number followed by atomic symbol\n
        e.g 1H or 13C
    '''

    temperature, magnetic_field, isotope = None, None, None

    temperature = float(find_lines(
        file_name,
        r'# *temperature (\d*\.*\d*)',
        re.IGNORECASE
    )[0])

    magnetic_field = float(find_lines(
        file_name,
        r'# *magnetic_field (\d*\.*\d*)',
        re.IGNORECASE
    )[0])

    isotope = find_lines(
        file_name,
        r'# *isotope (\d{0,3}[A-Za-z]{0,2})',
        re.IGNORECASE
    )[0]

    return temperature, magnetic_field, isotope


def find_index_of_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):  # noqa
        return idx - 1
    else:
        return idx


def isotope_format(isotope_string: str) -> str:
    r'''
    Converts isotope string into Mathtext, compatible with matplotlib

    Parameters
    ----------
    isotope_string: str
        e.g. 1H, 13C

    Returns
    -------
    str
        Mathtext formatted string with enclosing $$\n
        e.g. $^\mathregular{13}\mathregular{C}$
    '''  # noqa

    # Split at number letter boundary
    for it, char in enumerate(isotope_string):
        if char not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            split_at = it
            break
    nums = isotope_string[:split_at]
    lets = isotope_string[split_at:]

    return r'$^\mathregular{{{}}} \mathregular{{{}}}$'.format(nums, lets)


def tensor_invariants(tensor: NDArray) -> tuple[float, float, float]:
    """
    Returns two scalar invariants of a second-rank tensor:

    I2 = T_xx - T_yy)^2 + (T_yy - T_zz)^2 + (T_zz - T_xx)^2
    I3 = (2*T_xy)^2 + (2*T_xz)^2 + (2*T_yz)^2
    """
    tensor = np.asarray(tensor, dtype=float)

    I1 = (
        (tensor[0, 1] - tensor[1, 0])**2
        + (tensor[0, 2] - tensor[2, 0])**2
        + (tensor[1, 2] - tensor[2, 1])**2
    )

    I2 = (
        (tensor[0, 0] - tensor[1, 1])**2
        + (tensor[1, 1] - tensor[2, 2])**2
        + (tensor[2, 2] - tensor[0, 0])**2
    )

    I3 = (
        (tensor[0, 1] + tensor[1, 0])**2
        + (tensor[0, 2] + tensor[2, 0])**2
        + (tensor[1, 2] + tensor[2, 0])**2
    )

    return I1, I2, I3


def calc_g_eff(spin: float, orbit: float, total_momentum_J: float | None):
    """Compute an effective electron g-factor.

    For spin-only systems (transition metals, organic radicals) where no
    total J is defined, this returns the free-electron g value GE.

    For lanthanides (or any system with well-defined L, S, J), this returns
    the Landé g_J factor computed from the supplied spin (S), orbital
    angular momentum (L = orbit) and total angular momentum J.

    Parameters
    ----------
    spin : float
        Spin quantum number S.
    orbit : float
        Orbital angular momentum quantum number L.
    total_momentum_J : float | None
        Total angular momentum quantum number J. If None or zero, the
        function falls back to GE.

    Returns
    -------
    float
        Effective g-factor (GE or g_J, depending on total_momentum_J).
    """

    # Spin-only case: no total J provided or explicitly zero
    if total_momentum_J is None or total_momentum_J == 0.0:
        return GE

    # Landé g_J expression using S, L and J
    J = float(total_momentum_J)
    # return 1.5 + (spin * (spin + 1) - orbit * (orbit + 1)) / (2.0 * J * (J + 1)) # -> see 6:47 EQ.
    return 1.5 + (spin * (spin + 1) - orbit * (orbit + 1)) / (2.0 * J * (J + 1))


def choose_S_eff(spin: float, total_momentum_J: float | None):
    """Return the effective angular momentum quantum number S_eff.

    In the relaxation and Curie expressions used here we want a single
    "effective" quantum number that controls the size of the magnetic
    moment:

    * For spin-only centres (transition metals, organic radicals), this is
      just the spin quantum number S.
    * For lanthanides (or other systems with well-defined J), we use the
      total angular momentum J instead.

    Parameters
    ----------
    spin : float
        Spin quantum number S.
    total_momentum_J : float | None
        Total angular momentum J. If None, spin is returned.

    Returns
    -------
    float
        S for spin-only systems, or J for lanthanides.
    """

    return spin if total_momentum_J is None else total_momentum_J


def get_spin_only_susceptibility(
    spin: float,
    orbit: float,
    total_momentum_J: float | None,
    temperature: float
) -> float:
    """Compute spin-only isotropic molar susceptibility in Å^3.

    This uses the Curie law with an effective g-factor and effective
    angular momentum quantum number S_eff (S for transition metals,
    J for lanthanides) at the supplied temperature. The susceptibility
    is first computed in SI units (m^3 mol^-1) and then converted to
    Å^3, which is the internal unit used in the rest of the code.

    Parameters
    ----------
    spin : float
        Spin quantum number S of the paramagnetic centre.
    orbit : float
        Orbital angular momentum quantum number L.
    total_momentum_J : float | None
        Total angular momentum quantum number J. If None, a pure
        spin-only description is assumed.
    temperature : float
        Temperature in Kelvin at which the spin-only susceptibility is
        evaluated.

    Returns
    -------
    float
        Spin-only isotropic molar susceptibility in Å^3.
    """

    # Landé g-factor uses S, L, J
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)

    # Effective moment quantum number for Curie law:
    # S for transition metals, J for lanthanides
    S_eff = choose_S_eff(spin, total_momentum_J)

    T = float(temperature)

    # Chi (SI, m^3 mol^-1)
    chi_only_iso_SI = (
        MU0 * MUB**2 * g_eff**2 * S_eff * (S_eff + 1)
        / (3 * KB * T)
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
    temperature: float,
) -> float:
    """Return the "true" isotropic susceptibility χ_true,iso in Å^3.

    This applies a correction for the anisotropic g-tensor using ORCA
    output. The susceptibility tensor is read from the ORCA file and
    combined with the g-tensor to give an effective isotropic value:

        χ_true,iso ≈ (g_eff / 3) * Σ_i χ_i / g_i,

    where χ_i and g_i are principal components of the susceptibility and
    g-tensors, respectively. The final value is returned in Å^3 per mole.

    Parameters
    ----------
    spin : float
        Spin quantum number S of the paramagnetic centre.
    orbit : float
        Orbital angular momentum quantum number L.
    total_momentum_J : float | None
        Total angular momentum quantum number J. If None, a spin-only
        description for g_eff is used.
    temperature : float
        Temperature in Kelvin at which the susceptibility tensor is
        evaluated.

    Returns
    -------
    float
        "True" isotropic susceptibility χ_true,iso in Å^3 per mole.
    """

    T = float(temperature)

    # Lookup susceptibility tensor at temperature T, divide by T if file contains chi*T
    chi_tensors = chi_tensors[T] / T

    # Use Landé g_J (or GE) to get an effective g-factor
    g_eff = calc_g_eff(spin, orbit, total_momentum_J)

    # Trace-based expression with g correction (cm^3 mol^-1)
    chi_true_iso = g_eff / 3.0 * np.trace(
        chi_tensors * np.linalg.inv(g_tensor.T)
    )

    # Convert from cm^3 mol^-1 to Å^3 per mole
    chi_true_iso = chi_true_iso * (
        1 / (1e-24 * constants.Avogadro / (4 * np.pi))
    )

    return chi_true_iso


def spectral_density_J(omega, tau):
    return tau / (1.0 + (omega * tau) ** 2)


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
    total_momentum_J
):
    """Compute SBM R1 dipolar relaxation rates for each nucleus.

    Implements the Solomon-Bloembergen-Morgan (SBM) expression for the
    nuclear longitudinal relaxation rate R1 arising from electron-nucleus
    dipolar interactions. The expression is evaluated for each nucleus
    in the system using its distance to the paramagnetic centre and the
    appropriate spectral density terms.

    Parameters
    ----------
    nuclei_labels : list[str]
        Labels of the nuclei for which rates are computed.
    nuclei_coords : dict[str, np.ndarray]
        Cartesian coordinates (in Å) of each nucleus.
    electron_coords : np.ndarray
        Cartesian coordinates (in Å) of the effective electron spin
        centre.
    gamma_I_dict : dict[str, float]
        Nuclear gyromagnetic ratios (rad s^-1 T^-1) for each label.
    omega_I_dict : dict[str, float]
        Nuclear Larmor angular frequencies (rad s^-1) for each label.
    omega_S : float
        Electron Larmor angular frequency (rad s^-1).
    tau_c1 : float
        Correlation time for the electron-nuclear dipolar interaction
        (usually rotational correlation time) in seconds.
    tau_c2 : float
        Correlation time entering the cross terms (often tau_c1 or
        an effective electronic correlation time) in seconds.
    spin : float
        Spin quantum number S of the paramagnetic centre.
    orbit : float
        Orbital angular momentum quantum number L.
    total_momentum_J : float | None
        Total angular momentum quantum number J, if defined.

    Returns
    -------
    dict[str, float]
        Mapping from nucleus label to R1 dipolar relaxation rate (s^-1).
    """

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
            * (MU0 / (4 * np.pi))**2
            * (gamma_I * g_eff * MUB)**2
            * S_eff * (S_eff + 1)
        )
        spectral_density = (
            3 * spectral_density_J(omega_I, tau_c1)
            + 6 * spectral_density_J(omega_I + omega_S, tau_c2)
            + spectral_density_J(omega_I - omega_S, tau_c2)
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
        total_momentum_J
):
    """Compute SBM R2 dipolar relaxation rates for each nucleus.

    Implements the Solomon-Bloembergen-Morgan (SBM) expression for the
    nuclear transverse relaxation rate R2 arising from electron-nucleus
    dipolar interactions. The expression includes both zero- and
    non-zero-frequency spectral density terms.

    Parameters
    ----------
    nuclei_labels : list[str]
        Labels of the nuclei for which rates are computed.
    nuclei_coords : dict[str, np.ndarray]
        Cartesian coordinates (in Å) of each nucleus.
    electron_coords : np.ndarray
        Cartesian coordinates (in Å) of the effective electron spin
        centre.
    gamma_I_dict : dict[str, float]
        Nuclear gyromagnetic ratios (rad s^-1 T^-1) for each label.
    omega_I_dict : dict[str, float]
        Nuclear Larmor angular frequencies (rad s^-1) for each label.
    omega_S : float
        Electron Larmor angular frequency (rad s^-1).
    tau_c1 : float
        Correlation time involving T1e.
    tau_c2 : float
        Correlation time involving T2e.
    spin : float
        Spin quantum number S of the paramagnetic centre.
    orbit : float
        Orbital angular momentum quantum number L.
    total_momentum_J : float | None
        Total angular momentum quantum number J, if defined.

    Returns
    -------
    dict[str, float]
        Mapping from nucleus label to R2 dipolar relaxation rate (s^-1).
    """

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
            * (MU0 / (4 * np.pi))**2
            * (gamma_I * g_eff * MUB)**2
            * S_eff * (S_eff + 1)
        )
        spectral_density = (
            4 * spectral_density_J(0, tau_c1)
            + 3 * spectral_density_J(omega_I, tau_c1)
            + 6 * spectral_density_J(omega_S, tau_c2)
            + 6 * spectral_density_J(omega_I + omega_S, tau_c2)
            + spectral_density_J(omega_I - omega_S, tau_c2)
        )
        rate = prefactor * spectral_density
        rates[label] = rate

    return rates


def sbm_r1_contact(
    nuclei_labels,
    Aiso_dict,
    omega_I_dict,
    omega_S,
    tau_e2,
    spin,
    total_momentum_J
):
    """Compute SBM R1 contact relaxation rates for each nucleus.

    Implements the Solomon-Bloembergen-Morgan (SBM) expression for the
    nuclear longitudinal relaxation rate R1 arising from isotropic
    Fermi-contact hyperfine coupling to the electron spin.

    Parameters
    ----------
    nuclei_labels : list[str]
        Labels of the nuclei for which rates are computed.
    Aiso_dict : dict[str, float]
        Isotropic hyperfine coupling constants A_iso (in angular
        frequency units) for each nucleus.
    omega_I_dict : dict[str, float]
        Nuclear Larmor angular frequencies (rad s^-1) for each label.
    omega_S : float
        Electron Larmor angular frequency (rad s^-1).
    tau_e2 : float
        Electronic correlation time entering the spectral density in
        seconds.
    spin : float
        Spin quantum number S of the paramagnetic centre.
    total_momentum_J : float | None
        Total angular momentum quantum number J, if defined.

    Returns
    -------
    dict[str, float]
        Mapping from nucleus label to R1 contact relaxation rate (s^-1).
    """

    # Effective angular momentum quantum number for the contact term
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R1 contact rates
    for label in nuclei_labels:
        Aiso = Aiso_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (
            (2 / 3)
            * Aiso**2
            * S_eff * (S_eff + 1)
        )
        spectral_density = (
            spectral_density_J(omega_I - omega_S, tau_e2)
        )
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
        total_momentum_J
):
    """Compute SBM R2 contact relaxation rates for each nucleus.

    Implements the Solomon-Bloembergen-Morgan (SBM) expression for the
    nuclear transverse relaxation rate R2 arising from isotropic
    Fermi-contact hyperfine coupling to the electron spin.

    Parameters
    ----------
    nuclei_labels : list[str]
        Labels of the nuclei for which rates are computed.
    Aiso_dict : dict[str, float]
        Isotropic hyperfine coupling constants A_iso (in angular
        frequency units) for each nucleus.
    omega_I_dict : dict[str, float]
        Nuclear Larmor angular frequencies (rad s^-1) for each label.
    omega_S : float
        Electron Larmor angular frequency (rad s^-1).
    tau_e1 : float
        Electronic correlation time entering the zero-frequency term,
        in seconds.
    tau_e2 : float
        Electronic correlation time entering the finite-frequency term,
        in seconds.
    spin : float
        Spin quantum number S of the paramagnetic centre.
    total_momentum_J : float | None
        Total angular momentum quantum number J, if defined.

    Returns
    -------
    dict[str, float]
        Mapping from nucleus label to R2 contact relaxation rate (s^-1).
    """

    # Effective angular momentum quantum number for the contact term
    S_eff = choose_S_eff(spin, total_momentum_J)

    rates = {}

    # Loop over nuclei and assemble individual R2 contact rates
    for label in nuclei_labels:
        Aiso = Aiso_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (
            (1 / 3)
            * Aiso**2
            * S_eff * (S_eff + 1)
        )
        spectral_density = (
            spectral_density_J(0, tau_e1)
            + spectral_density_J(omega_I - omega_S, tau_e2)
        )
        rate = prefactor * spectral_density
        rates[label] = rate
    return rates


def diagonalise_tensor_and_order_eigenvalues(tensor: NDArray) -> tuple[NDArray, NDArray]:
    """Diagonalise and order a second-rank tensor.

    Parameters
    ----------
    tensor : np.ndarray
        3x3 second-rank tensor.

    Returns
    -------
    eigvecs : np.ndarray
        3x3 matrix whose columns are the ordered eigenvectors of the
        input tensor.
    eigvals : np.ndarray
        1D array of length 3 containing the ordered eigenvalues of the
        input tensor.
    """
    tensor = np.asarray(tensor, dtype=float)

    # Symmetrise tensor
    tensor_sym = 0.5 * (tensor + tensor.T)

    # Diagonalise tensor
    eigvals, eigvecs = np.linalg.eigh(tensor_sym)

    # Isotropic part
    eig_iso = np.trace(tensor_sym) / 3

    # Sort eigenvalues according to:
    # |eig_yy - eig_iso| <= |eig_xx - eig_iso| <= |eig_zz - eig_iso|
    eig_order = np.argsort(eigvals - eig_iso)
    eigvecs = eigvecs[:, eig_order]
    eigvals = eigvals[eig_order]

    return eigvecs, eigvals


def ion_nucleus_polar_in_g_frame(
        r_vec: NDArray,
        g_tensor: NDArray,
) -> tuple[float, float]:
    """Compute polar coordinates of ion-nucleus vector in g-tensor PAS.

    Parameters
    ----------
    r_vec : np.ndarray
        Ion-nucleus vecor in the molecular frame.
    g_tensor : np.ndarray
        3x3 g-tensor in the molecular frame.

    Returns
    -------
    theta : float
        Polar angle θ (in radians) of nucleus in g-tensor frame.
    phi : float
        Azimuthal angle φ (in radians) of nucleus in g-tensor frame.
    """
    r_vec = np.asarray(r_vec, dtype=float)

    # Symmetrise g-tensor, diagonalise to PAS, and order eigenvalues.
    eigvecs, eigvals = diagonalise_tensor_and_order_eigenvalues(g_tensor)

    # Rotate r_vec into g-frame
    r_g_frame = eigvecs.T @ r_vec

    # Convert to polar coordinates
    x, y, z = r_g_frame
    r = np.linalg.norm(r_g_frame)
    if r == 0.0:
        raise ValueError(
            "Ion-nucleus vector has zero length; polar angles undefined.")
    cos_theta = np.clip(z / r, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    phi = np.arctan2(y, x)

    return theta, phi


def anisotropic_coefficients(
        g_aniso: float,
        eta: float,
        theta: float,
        phi: float,
) -> tuple[complex, complex, complex]:
    """
    Compute anisotropic coefficients a, b, c for Zeeman-limit relaxation.

    Parameters
    ----------
    g_aniso : float
        Anisotropy of g-tensor, gzz - giso.
    eta : float
        Asymmetry parameter of g-tensor, (gyy - gxx) / g_aniso.
    theta : float
        Polar angle θ (in radians) of nucleus in g-tensor frame.
    phi : float
        Azimuthal angle φ (in radians) of nucleus in g-tensor frame.
    Returns
    -------
    a, b, c : complex
        Anisotropic coefficients
    """
    a = - (g_aniso / 2) * (1 - 3 * np.cos(theta)**2 + eta * np.cos(2 * phi) * np.sin(theta)**2)  # noqa

    b = - (g_aniso / 2) * ((3 / 2) * np.sin(2 * theta)
                          + (eta / 2) * np.sin(2 * theta) * np.cos(2 * phi)
                          - 1.0j * eta * np.sin(2 * phi) * np.sin(theta))  # noqa

    c = - (g_aniso / 2) * (1 - 3 * np.sin(theta)**2
                              + eta * np.cos(2 * phi) * (1 + np.cos(theta)**2)
                              - 2.0j * eta * np.cos(theta) * np.sin(2 * phi))  # noqa

    return a, b, c


def anisotropic_zeeman_limit_r1_dipolar(
    nuclei_labels,
    nuclei_coords,
    electron_coords,
    g_tensor,
    gamma_I_dict,
    omega_I_dict,
    omega_S,
    tau_C11,
    tau_C12,
    tau_C21,
    tau_C22,
    spin
):
    """
    Compute Zeeman-limit anisotropic dipolar contribution to R1 using the g-tensor,
    as described in Vasavada and Rao (Journal of Magnetic Resonance, 1989), DOI: 10.1016/0022-2364(89)90059-0.

    The scaling of the electron Larmor frequency, omega_S_avg = omega_S * g_iso / GE, is included as in Bertini, 
    Luchinat, and Vasavada (Journal of Magnetic Resonance, 1990), DOI: 10.1016/0022-2364(90)90231.

    Parameters
    ----------
    g_tensor : (3,3) array_like
        Electronic g-tensor in the molecular frame.

    Notes
    -----
    The g-tensor is diagonalised and its eigenvalues are ordered; g_iso, g_aniso and eta are derived from those
    eigenvalues. The ion-nucleus vector is expressed in the g-tensor PAS, with the orientation parameterised
    by polar angles.
    """
    # Diagonalise g-tensor and order eigenvalues
    g_eigvals, g_eigvecs = diagonalise_tensor_and_order_eigenvalues(g_tensor)

    # Use g tensor to define the asymmetry parameter (eta)
    g_yy, g_xx, g_zz = g_eigvals
    g_iso = (g_xx + g_yy + g_zz) / 3.0
    g_aniso = g_zz - g_iso
    eta = (g_yy - g_xx) / g_aniso if g_aniso != 0 else 0.0
    omega_S_avg = omega_S * g_iso / GE

    # Define first- and second-rank rotational correlation times

    rates = {}

    for label in nuclei_labels:
        # Ion-nucleus vecor in moelcular frame
        r_vec = nuclei_coords[label] - electron_coords
        # Polar angles in g-PAS
        theta, phi = ion_nucleus_polar_in_g_frame(r_vec, g_tensor)
        # Angular coefficients
        a, b, c = anisotropic_coefficients(g_aniso, eta, theta, phi)

        r = np.linalg.norm(r_vec) * 1e-10
        gamma_I_dict = gamma_I_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (
            (1.0 / 3.0) * (MU0 / (4 * np.pi))**2
            * (MUB**2 * gamma_I_dict**2 / r**6)
            * spin * (spin + 1)
        )
        spectral_density = (1.0 / 5.0 * (
                    6.0 * (g_iso + (a / 2.0))**2
                    + np.abs(b)**2 / 2.0
                    + np.abs(c)**2 / 2.0
                )
                * (
                    2.0 * spectral_density_J(omega_S_avg + omega_I, tau_C22)
                   + spectral_density_J(omega_I, tau_C21)
                    + (1.0 / 3.0) * spectral_density_J(omega_S_avg - omega_I, tau_C22))
                      + (3.0 / 2.0) * np.abs(b)**2 * (
                          spectral_density_J(
                              omega_I, tau_C11) + spectral_density_J(omega_S_avg - omega_I, tau_C12)
                          )
                          )  # noqa

        rate = prefactor * spectral_density
        rates[label] = rate

    return rates


def anisotropic_zeeman_limit_r2_dipolar(
        nuclei_labels,
        nuclei_coords,
        electron_coords,
        g_tensor,
        gamma_I_dict,
        omega_I_dict,
        omega_S,
        tau_C11,
        tau_C12,
        tau_C21,
        tau_C22,
        spin
):
    """
    Compute Zeeman-limit anisotropic dipolar contribution to R2 using the g-tensor,
    as described in Vasavada and Rao (Journal of Magnetic Resonance, 1989), DOI: 10.1016/0022-2364(89)90059-0.

    The scaling of the electron Larmor frequency, omega_S_avg = omega_S * g_iso / GE, is included as in Bertini,
    Luchinat, and Vasavada (Journal of Magnetic Resonance, 1990), DOI: 10.1016/0022-2364(90)90231.

    Parameters
    ----------
    g_tensor : (3,3) array_like
        Electronic g-tensor in the molecular frame.

    Notes
    -----
    The g-tensor is diagonalised and its eigenvalues are ordered using
    diagonalise_g_tensor; g_iso, g_aniso and eta are derived from those
    eigenvalues, consistent with ion_nucleus_polar_in_g_frame.
    """
    # Diagonalise g-tensor and order eigenvalues
    g_eigvals, g_eigvecs = diagonalise_tensor_and_order_eigenvalues(g_tensor)

    # Use g tensor to define the asymmetry parameter (eta)
    g_yy, g_xx, g_zz = g_eigvals
    g_iso = (g_xx + g_yy + g_zz) / 3.0
    g_aniso = g_zz - g_iso
    eta = (g_yy - g_xx) / g_aniso if g_aniso != 0 else 0.0
    omega_S_avg = omega_S * g_iso / GE

    rates = {}

    for label in nuclei_labels:
        # Ion-nucleus vecor in moelcular frame
        r_vec = nuclei_coords[label] - electron_coords
        # Polar angles in g-PAS
        theta, phi = ion_nucleus_polar_in_g_frame(r_vec, g_tensor)
        # Angular coefficients
        a, b, c = anisotropic_coefficients(g_aniso, eta, theta, phi)

        r = np.linalg.norm(r_vec) * 1e-10
        gamma_I_dict = gamma_I_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (
            (1.0 / 3.0) * (MU0 / (4 * np.pi))**2
            * (MUB**2 * gamma_I_dict**2 / r**6)
            * spin * (spin + 1)
        )
        spectral_density = (1.0 / 5.0 * (
                    6.0 * (g_iso + (a / 2.0))**2
                    + np.abs(b)**2 / 2.0
                    + np.abs(c)**2 / 2.0
                )
                * (
                    spectral_density_J(omega_S_avg + omega_I, tau_C22)
                   + (1.0 / 2.0) * spectral_density_J(omega_I, tau_C21)
                    + spectral_density_J(omega_S_avg, tau_C22)
                     + (2.0 / 3.0) * spectral_density_J(0.0, tau_C21)
                      + (1.0 / 6.0) * spectral_density_J(omega_S_avg - omega_I, tau_C22))
                      + (3.0 / 2.0) * np.abs(b)**2 * (
                          spectral_density_J(omega_I, tau_C11)
                          + spectral_density_J(omega_S_avg, tau_C12)
                          + (1.0 / 2.0) *
                             spectral_density_J(omega_S_avg - omega_I, tau_C12)
                          )
                          )  # noqa

        rate = prefactor * spectral_density
        rates[label] = rate

    return rates


def anisotropic_zeeman_limit_r1_contact(
    nuclei_labels,
    nuclei_coords,
    electron_coords,
    g_tensor,
    Aiso_dict,
    omega_I_dict,
    omega_S,
    tau_e2,
    spin
):
    """
    Compute Zeeman-limit anisotropic contact contribution to R1,
    as described in Vasavada and Rao (Journal of Magnetic Resonance, 1989), 
    DOI: 10.1016/0022-2364(89)90059-0.

    The scaling of the electron Larmor frequency, omega_S_avg = omega_S * g_iso / GE, is included as in Bertini,
    Luchinat, and Vasavada (Journal of Magnetic Resonance, 1990), DOI: 10.1016/0022-2364(90)90231.

    Parameters
    ----------
    g_tensor : (3,3) array_like
        Electronic g-tensor in the molecular frame.

    Notes
    -----
    The g-tensor is diagonalised and its eigenvalues are ordered; g_iso is derived from those eigenvalues.
    """

    # Diagonalise g-tensor and order eigenvalues
    g_eigvals, g_eigvecs = diagonalise_tensor_and_order_eigenvalues(g_tensor)

    # Use g tensor to define the asymmetry parameter (eta)
    g_yy, g_xx, g_zz = g_eigvals
    g_iso = (g_xx + g_yy + g_zz) / 3.0
    g_aniso = g_zz - g_iso
    eta = (g_yy - g_xx) / g_aniso if g_aniso != 0 else 0.0
    omega_S_avg = omega_S * g_iso / GE

    rates = {}

    for label in nuclei_labels:
        # Ion-nucleus vecor in moelcular frame
        r_vec = nuclei_coords[label] - electron_coords
        # Polar angles in g-PAS
        theta, phi = ion_nucleus_polar_in_g_frame(r_vec, g_tensor)
        # Angular coefficients
        a, _, _ = anisotropic_coefficients(g_aniso, eta, theta, phi)

        r = np.linalg.norm(r_vec) * 1e-10
        Aiso = Aiso_dict[label]
        gamma_I_dict = gamma_I_dict[label]
        omega_I = omega_I_dict[label]

        prefactor = (Aiso - (gamma_I_dict * MUB * a / r**3)
                     )**2 * (2.0 / 3.0) * spin * (spin + 1)
        spectral_density = spectral_density_J(omega_S_avg - omega_I, tau_e2)

        rate = prefactor * spectral_density
        rates[label] = rate

    return rates


def anisotropic_zeeman_limit_r2_contact(
    nuclei_labels,
    nuclei_coords,
    electron_coords,
    g_tensor,
    Aiso_dict,
    omega_I_dict,
    omega_S,
    tau_e1,
    tau_e2,
    spin
):
    """
    Compute Zeeman-limit anisotropic contact contribution to R2,
    as described in Vasavada and Rao (Journal of Magnetic Resonance, 1989), 
    DOI: 10.1016/0022-2364(89)90059-0.

    The scaling of the electron Larmor frequency, omega_S_avg = omega_S * g_iso / GE, is included as in Bertini,
    Luchinat, and Vasavada (Journal of Magnetic Resonance, 1990), DOI: 10.1016/0022-2364(90)90231.

    Parameters
    ----------
    g_tensor : (3,3) array_like
        Electronic g-tensor in the molecular frame.

    Notes
    -----
    The g-tensor is diagonalised and its eigenvalues are ordered; g_iso is derived from those eigenvalues.
    """

    # Diagonalise g-tensor and order eigenvalues
    g_eigvals, g_eigvecs = diagonalise_tensor_and_order_eigenvalues(g_tensor)

    # Use g tensor to define the asymmetry parameter (eta)
    g_yy, g_xx, g_zz = g_eigvals
    g_iso = (g_xx + g_yy + g_zz) / 3.0
    g_aniso = g_zz - g_iso
    eta = (g_yy - g_xx) / g_aniso if g_aniso != 0 else 0.0
    omega_S_avg = omega_S * g_iso / GE

    rates = {}

    for label in nuclei_labels:
        # Ion-nucleus vecor in moelcular frame
        r_vec = nuclei_coords[label] - electron_coords
        # Polar angles in g-PAS
        theta, phi = ion_nucleus_polar_in_g_frame(r_vec, g_tensor)
        # Angular coefficients
        a, _, _ = anisotropic_coefficients(g_aniso, eta, theta, phi)

        r = np.linalg.norm(r_vec) * 1e-10
        Aiso = Aiso_dict[label]
        gamma_I_dict = gamma_I_dict[label]
        omega_I = omega_I_dict[label]

        prefactor = (Aiso - (gamma_I_dict * MUB * a / r**3)
                     )**2 * (1.0 / 3.0) * spin * (spin + 1)
        spectral_density = spectral_density_J(
            0, tau_e1) + spectral_density_J(omega_S_avg - omega_I, tau_e2)

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
    total_momentum_J
):
    """Compute Guéron R1 Curie relaxation rates for each nucleus.

    Implements the Guéron expression for nuclear longitudinal
    relaxation (R1) due to the Curie (static) dipolar interaction with
    an anisotropic electron magnetic susceptibility tensor. The formula
    is evaluated in the point-dipole approximation.

    Parameters
    ----------
    nuclei_labels : list[str]
        Labels of the nuclei for which rates are computed.
    nuclei_coords : dict[str, np.ndarray]
        Cartesian coordinates (in Å) of each nucleus.
    electron_coords : np.ndarray
        Cartesian coordinates (in Å) of the effective electron spin
        centre.
    omega_I_dict : dict[str, float]
        Nuclear Larmor angular frequencies (rad s^-1) for each label.
    T : float
        Temperature in Kelvin.
    tau_R : float
        Rotational correlation time in seconds.
    spin : float
        Spin quantum number S of the paramagnetic centre.
    orbit : float
        Orbital angular momentum quantum number L.
    total_momentum_J : float | None
        Total angular momentum quantum number J, if defined.

    Returns
    -------
    dict[str, float]
        Mapping from nucleus label to R1 Curie relaxation rate (s^-1).
    """

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
            * (MU0 / (4 * np.pi))**2
            * (omega_I / (3 * consts.k * T))**2
            * (g_eff * MUB)**4
            * (S_eff * (S_eff + 1))**2
        )
        spectral_density = (3 * spectral_density_J(omega_I, tau_R))
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
        total_momentum_J
):
    """Compute Guéron R2 Curie relaxation rates for each nucleus.

    Implements the Guéron expression for nuclear transverse relaxation
    (R2) due to the Curie (static) dipolar interaction with an
    anisotropic electron magnetic susceptibility tensor, in the
    point-dipole approximation.

    Parameters
    ----------
    nuclei_labels : list[str]
        Labels of the nuclei for which rates are computed.
    nuclei_coords : dict[str, np.ndarray]
        Cartesian coordinates (in Å) of each nucleus.
    electron_coords : np.ndarray
        Cartesian coordinates (in Å) of the effective electron spin
        centre.
    omega_I_dict : dict[str, float]
        Nuclear Larmor angular frequencies (rad s^-1) for each label.
    T : float
        Temperature in Kelvin.
    tau_R : float
        Rotational correlation time in seconds.
    spin : float
        Spin quantum number S of the paramagnetic centre.
    orbit : float
        Orbital angular momentum quantum number L.
    total_momentum_J : float | None
        Total angular momentum quantum number J, if defined.

    Returns
    -------
    dict[str, float]
        Mapping from nucleus label to R2 Curie relaxation rate (s^-1).
    """

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
            * (MU0 / (4 * np.pi))**2
            * (omega_I / (3 * consts.k * T))**2
            * (g_eff * MUB)**4
            * (S_eff * (S_eff + 1))**2
        )
        spectral_density = (4 * spectral_density_J(0, tau_R) +
                            3 * spectral_density_J(omega_I, tau_R))
        rate = prefactor * spectral_density
        rates[label] = rate

    return rates


def r1_anisotropic_curie(
        nuclei_labels,
        nuclei_coords,
        electron_coords,
        omega_I_dict,
        susc_tensor,
        shielding_tensor_dict,
        tau_R
):
    """
    As described in DOI: 10.1039/c8cp01332b, equation (17).
    """
    rates = {}

    for label in nuclei_labels:

        # Get electron-nuclea distance vector and magnitude
        r_vec = nuclei_coords[label] - electron_coords
        r = np.linalg.norm(r_vec)
        # Define shielding tensor for each nucleus
        shielding_tensor = shielding_tensor_dict[label]

        # Construct dipolar and shielding tensors
        dipolar_tensor = (1.0 / (4.0 * np.pi)) * (3.0 * (np.outer(r_vec, r_vec) / r**5) - (1.0 / r**3) * np.eye(3))  # noqa
        shielding_tensor_full = shielding_tensor - \
            (dipolar_tensor @ susc_tensor)

        # Calculate first- and second-rank invariants
        I1, I2, I3 = tensor_invariants(shielding_tensor_full)
        Lambda_first_rank = np.sqrt(I1)
        Lambda_second_rank = np.sqrt(I2 + I3)

        omega_I = omega_I_dict[label]

        rate_rank1 = (0.5) * Lambda_first_rank**2 * omega_I**2 * spectral_density_J(3*omega_I, tau_R)  # noqa
        rate_rank2 = (2.0 / 15.0) * Lambda_second_rank**2 * omega_I**2 * spectral_density_J(omega_I, tau_R)  # noqa
        rate = rate_rank1 + rate_rank2

        rates[label] = rate

    return rates


def r2_anisotropic_curie(
        nuclei_labels,
        nuclei_coords,
        electron_coords,
        omega_I_dict,
        susc_tensor,
        shielding_tensor_dict,
        tau_R
):
    """
    As described in DOI: 10.1039/c8cp01332b, equation (17).
    """
    rates = {}

    for label in nuclei_labels:

        # Get electron-nuclear distance vector and magnitude
        r_vec = nuclei_coords[label] - electron_coords
        r = np.linalg.norm(r_vec)
        # Define shielding tensor for each nucleus
        shielding_tensor = shielding_tensor_dict[label]

        # Construct dipolar and shielding tensors
        dipolar_tensor = (1.0 / (4.0 * np.pi)) * (3.0 * (np.outer(r_vec, r_vec) / r**5) - (1.0 / r**3) * np.eye(3))  # noqa
        shielding_tensor_full = shielding_tensor - \
            (dipolar_tensor @ susc_tensor)

        # Calculate first- and second-rank invariants
        I1, I2, I3 = tensor_invariants(shielding_tensor_full)
        Lambda_first_rank = np.sqrt(I1)
        Lambda_second_rank = np.sqrt(I2 + I3)

        omega_I = omega_I_dict[label]

        rate_rank1 = (0.25) * Lambda_first_rank**2 * omega_I**2 * spectral_density_J(3*omega_I, tau_R)  # noqa
        rate_rank2 = (1.0 / 45.0) * Lambda_second_rank**2 * omega_I**2 * (4.0 * spectral_density_J(0, tau_R) + spectral_density_J(omega_I, tau_R))  # noqa
        rate = rate_rank1 + rate_rank2

        rates[label] = rate

    return rates


def r1_zfs_anisotropic_dipolar(
        nuclei_labels,
        nuclei_coords,
        electron_coords,
        gamma_I_dict,
        spectral_density_tensor_omega,
):
    """
    As described in DOI: 10.1039/c8cp01332b, equation (31).
    """
    rates = {}

    for label in nuclei_labels:

        r_vec = nuclei_coords[label] - electron_coords
        r = np.linalg.norm(r_vec) * 1e-10  # in meters
        r_unit = r_vec / np.linalg.norm(r_vec)

        G_omega = spectral_density_tensor_omega
        gamma_I = gamma_I_dict[label]
        prefactor = (2.0/3.0) * (MU0 / (4.0 * np.pi))**2 * \
            (1.0 / r**6) * gamma_I**2
        spectral_density = (3.0 * (np.outer(r_unit, r_unit)) - np.eye(3))**2 @ G_omega  # noqa

        rate = prefactor * spectral_density

        rates[label] = rate

    return rates


def r2_zfs_anisotropic_dipolar(
        nuclei_labels,
        nuclei_coords,
        electron_coords,
        gamma_I_dict,
        spectral_density_tensor_zero,
        spectral_density_tensor_omega  # specify units
):
    """
    As described in DOI: 10.1039/c8cp01332b, equation (31).
    """
    rates = {}

    for label in nuclei_labels:

        r_vec = nuclei_coords[label] - electron_coords
        r = np.linalg.norm(r_vec) * 1e-10  # in meters
        r_unit = r_vec / np.linalg.norm(r_vec)

        G_zero = spectral_density_tensor_zero
        G_omega = spectral_density_tensor_omega
        gamma_I = gamma_I_dict[label]

        prefactor = (1.0/3.0) * (MU0 / (4.0 * np.pi))**2 * \
            (1.0 / r**6) * gamma_I**2
        spectral_density = (3.0 * (np.outer(r_unit, r_unit)) -
                            np.eye(3))**2 @ (G_zero + G_omega)

        rate = prefactor * spectral_density

        rates[label] = rate

    return rates
