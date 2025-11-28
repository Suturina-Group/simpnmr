
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
    where NAME is one of temperature, larmor, or isotope

    Parameters
    ----------
    file_name: str
        File to read

    Returns
    -------
    float
        Temperature in Kelvin
    float
        Larmor frequency for free nucleus in this spectrometer in MHz
    str
        Isotope symbol formatted as nucleon number followed by atomic symbol\n
        e.g 1H or 13C
    '''

    temperature, larmor, isotope = None, None, None

    temperature = float(find_lines(
        file_name,
        r'# *temperature (\d*\.*\d*)',
        re.IGNORECASE
    )[0])

    larmor = float(find_lines(
        file_name,
        r'# *larmor (\d*\.*\d*)',
        re.IGNORECASE
    )[0])

    isotope = find_lines(
        file_name,
        r'# *isotope (\d{0,3}[A-Za-z]{0,2})',
        re.IGNORECASE
    )[0]

    return temperature, larmor, isotope


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


def get_spin_only_susceptibility(uargs, temperature):
    config = inps.PredictConfig.from_file(uargs.input_file)

    if config.spin_S is not None:
        spin = config.spin_S
    else:
        spin = rdrs.QCSpin.guess_from_file(config.hyperfine_file).S

    T = temperature

    # Calculate spin-only magnetic susceptibility
    chi_only_iso = (MU0 * MUB**2 * GE**2 * spin * (spin + 1)) / (3 * KB * T) * (10 ** 32)  # Si [10^-32 m^3]

    # Convert from Si to A^3
    chi_only_iso = chi_only_iso * 10**-2

    return chi_only_iso

def get_true_iso_susceptibility(uargs, temperature):
    """
    Returns the "true" isotropic susceptibility χ_true,iso, corrected for the
    anisotropic g-tensor.

    χ_true,iso = (g_e / 3) * (χ_x / g_x + χ_y / g_y + χ_z / g_z)
    """
    config = inps.PredictConfig.from_file(uargs.input_file)

    T = float(temperature)

    section = config.susceptibility_format.split("orca_")[1]

    g_tensor = rdrs.read_orca_g_tensor(
        config.susceptibility_file,
        section=section
    )

    chi_tensors = rdrs.read_orca_susceptibility(
        config.susceptibility_file,
        section=section
    )

    if T in chi_tensors:
        chi_tensor = chi_tensors[T] / T

    chi_true_iso = GE / 3.0 * np.trace(chi_tensor * np.linalg.inv(g_tensor.T)) # cm3 mol-1

    # Convert cm^3 mol^-1 to A^3
    chi_true_iso = chi_true_iso * (1 / (1e-24 * constants.Avogadro / (4 * np.pi))) # A3
    
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
    spin
):
    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    rates = {}
    for label in nuclei_labels:
        r = np.linalg.norm(nuclei_coords[label] - electron_coords) * 1e-10
        gamma_I = gamma_I_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (
            (1 / 10)
            * (1 / r**6)
            * (MU0 / (4 * np.pi))**2
            * (gamma_I * GE * MUB)**2
            * spin * (spin + 1)
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
        spin
):
    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    rates = {}
    for label in nuclei_labels:
        r = np.linalg.norm(nuclei_coords[label] - electron_coords) * 1e-10
        gamma_I = gamma_I_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (
            (1 / 15)
            * (1 / r**6)
            * (MU0 / (4 * np.pi))**2
            * (gamma_I * GE * MUB)**2
            * spin * (spin + 1)
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
    nuclei_labels,
    Aiso_dict,
    omega_I_dict,
    omega_S,
    tau_e2,
    spin
):
    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    rates = {}

    for label in nuclei_labels:
        Aiso = Aiso_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (
            (2 / 3)
            * Aiso**2
            * spin * (spin + 1)
        )
        spectral_density = (
            J(omega_I - omega_S, tau_e2)
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
        spin
):
    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    rates = {}

    for label in nuclei_labels:
        Aiso = Aiso_dict[label]
        omega_I = omega_I_dict[label]
        prefactor = (
            (1 / 3)
            * Aiso**2
            * spin * (spin + 1)
        )
        spectral_density = (
            J(0, tau_e1)
            + J(omega_I - omega_S, tau_e2)
        )
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
    spin
):

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    rates = {}

    for label in nuclei_labels:
        r = np.linalg.norm(nuclei_coords[label] - electron_coords) * 1e-10
        omega_I = omega_I_dict[label]
        prefactor = (
            (2 / 5)
            * (1 / r**6)
            * (MU0 / (4 * np.pi))**2
            * (omega_I / (3 * consts.k * T))**2
            * (GE * MUB)**4
            * (spin * (spin + 1))**2
        )
        spectral_density = (3 * J(omega_I, tau_R))
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
        spin
):

    def J(omega, tau):
        return tau / (1 + (omega * tau) ** 2)

    rates = {}

    for label in nuclei_labels:
        r = np.linalg.norm(nuclei_coords[label] - electron_coords) * 1e-10
        omega_I = omega_I_dict[label]
        prefactor = (
            (1 / 5)
            * (1 / r**6)
            * (MU0 / (4 * np.pi))**2
            * (omega_I / (3 * consts.k * T))**2
            * (GE * MUB)**4
            * (spin * (spin + 1))**2
        )
        spectral_density = (4 * J(0, tau_R) + 3 * J(omega_I, tau_R))
        rate = prefactor * spectral_density
        rates[label] = rate

    return rates
