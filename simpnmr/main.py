'''
This module contains objects and methods for working with paramagnetic nmr
spectra and susceptibility tensors
'''

import numpy as np
from numpy.typing import ArrayLike, NDArray
import numpy.linalg as la
import copy
import xyz_py as xyzp
import xyz_py.atomic as atomic
import scipy.constants as constants
import pandas as pd
from itertools import permutations, product, chain
import datetime
import warnings
import re


from . import utils as ut
from . import readers as rdrs
from . import string_tools as st
from .__version__ import __version__


class Signal():
    '''
    Contains information on a single signal in an NMR experiment

    Parameters
    ----------
    shift: float
        Chemical Shift in ppm
    width: float
        Width of signal in ppm
    area: float
        Area (integral) of signal
    assignment: str, default 'UNK'
        Chemical label of signal
    l_to_g: float, default 1.
        Ratio of Lorentzian to Gaussian lineshape for this signal

    Attributes
    ----------
    shift: float
        Chemical Shift in ppm
    width: float
        Width of signal in ppm
    area: float
        Area (integral) of signal
    assignment: str
        Chemical label of signal
    l_to_g: float
        Ratio of Lorentzian to Gaussian lineshape for this signal
    '''

    def __init__(self, shift, width, area, assignment='UNK', l_to_g=1):

        self.shift = shift
        self.width = width
        self.area = area
        self.assignment = assignment
        self.l_to_g = l_to_g

        return


class Experiment():
    '''
    Contains information on a single pNMR experiment at a given temperature

    Parameters
    ----------
    temperature: float
        Temperature of experiment
    larmor: float
        Larmor frequency of nucleus in spectrometer
    nucleus: str
        Nucleus of experiment
    signals: list[Signal]
        Signals of experiment
    spectrum: ArrayLike, optional
        Experimental spectrum as (N, 2) array. First column is ppm, second\n
        is intensity.

    Attributes
    ----------
    temperature: float
        Temperature of experiment
    signals: list[Signal]
        Signals of experiment
    larmor: float
        Larmor frequency of nucleus in spectrometer
    isotope: str
        Isotope of experiment
    spectrum: None | ArrayLike
        Experimental spectrum as (N, 2) array. First column is ppm, second\n
        is intensity.
    '''

    def __init__(self, temperature: float, larmor: float, isotope: str,
                 signals: list[Signal], spectrum: ArrayLike = None) -> None:
        self._signals = signals
        self.temperature = temperature
        self.larmor = larmor
        self.isotope = isotope.title()

        if spectrum is not None:
            self.spectrum = spectrum
        else:
            self._spectrum = None
        return

    def load_spectrum_from_file(self, file_name: str):
        '''
        Add spectrum data attribute from file containing x and y values\n
        of spectrum

        Parameters
        ----------
        file_name: str
            .csv file containing spectrum values. 1st column shift,
            2nd intensity with no header line
        '''

        self.spectrum = np.loadtxt(file_name, delimiter=',', comments='#')

        self.spectrum = self.spectrum[np.flipud(self.spectrum[:, 0].argsort())]

        return

    def keys(self):
        '''
        Keys method which allows signals to be selected from an experiment
        using their assignment
        '''
        return [signal.assignment for signal in self.signals]

    def __getitem__(self, item):
        # This is probably slow
        lookup = {
            signal.assignment: signal
            for signal in self.signals
        }
        return lookup[item]

    def __iter__(self):
        return iter(self.signals)

    @property
    def spectrum(self):
        return self._spectrum

    @spectrum.setter
    def spectrum(self, value: ArrayLike):
        self._spectrum = np.asarray(value)

    @property
    def signals(self):
        return self._signals

    @signals.setter
    def signals(self, value):
        self._signals = value
        return

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        try:
            value = float(value)
        except TypeError:
            raise TypeError('temperature must be floatable')
        self._temperature = value
        return

    @property
    def larmor(self) -> float:
        return self._larmor

    @larmor.setter
    def larmor(self, value: float):
        try:
            value = float(value)
        except TypeError:
            raise TypeError('larmor must be floatable')
        self._larmor = value
        return

    @property
    def isotope(self) -> str:
        return self._isotope

    @isotope.setter
    def isotope(self, value: str):
        if not isinstance(value, str):
            raise ValueError('isotope must be str')
        self._isotope = value
        return

    def __str__(self):
        out = f'Temperature {self.temperature:f} K\n'
        out += 'assignment, shift, width, area\n'
        width = max([len(signal.assignment) for signal in self.signals])
        for signal in self.signals:
            out += '{}, {: 10.4f}, {:7.4f}, {:5.2f}\n'.format(
                signal.assignment.ljust(width),
                signal.shift,
                signal.width,
                signal.area
            )
        return out

    @classmethod
    def from_file(cls, file_names: str | list[str]) -> list['Experiment']:
        '''
        Creates list of Experiment objects from file containing
        shifts, widths, temperatures, and labels

        Parameter
        ---------
        file_names: str | list[str]
            Name of file(s) to read

        Returns
        -------
        list[Experiment]
            Experiments, one per file

        Raises
        ------
        ValueError
            If file_names is an empty string or list
        '''

        if not len(file_names):
            raise ValueError(ut.cstr('No files provided', 'red'))

        if isinstance(file_names, str):
            file_names = [file_names]

        # Standardise column names
        name_convertor = {
            'shifts': 'shift',
            'shifts (ppm)': 'shift',
            'shift (ppm)': 'shift',
            'ppm': 'shift',
            'assignment': 'assignment',
            'assignments': 'assignmen',
            'assignments ()': 'assignment',
            'assignment ()': 'assignment',
            'widths': 'width',
            'width ()': 'width',
            'width(Hz)': 'width',
            'width (Hz)': 'width',
            'widths ()': 'width',
            'areas': 'area',
            'area ()': 'area',
            'areas ()': 'area',
            'integral': 'area',
            'integral ()': 'area',
            'integral': 'area',
            'integrals ()': 'area',
            'L/G ()': 'L/G'
        }
        others = {}
        for key, val in name_convertor.items():
            others[key.capitalize()] = val
            others[val.capitalize()] = val
        name_convertor.update(others)

        # Read each file
        final = []
        for file_name in file_names:
            _data = pd.read_csv(file_name, comment='#', skipinitialspace=True)
            _data.rename(columns=name_convertor, inplace=True)
            _temperature, _larmor, _isotope = ut.read_exp_metadata(file_name)
            _data['temperature'] = _temperature
            _data['larmor'] = _larmor
            _data['isotope'] = _isotope
            final.append(_data)

        # combine into a single dataframe
        data = pd.concat(final)
        data.reset_index(inplace=True)

        # Sort by temperature
        data = data.sort_values('temperature')

        # Add linewidth ratio if missing
        if 'L/G' not in data.columns:
            data['L/G'] = 1.

        # Split by mean temperature
        split_indices = ut.find_mean_values(data['temperature'], thresh=0.1)

        if len(split_indices):
            split_indices = [0] + split_indices
            split_indices.append(len(data))
            _exp = [
                data.iloc[split_indices[n]: split_indices[n + 1]]
                for n in range(len(split_indices) - 1)
            ]
        else:
            _exp = [data]

        # Then sort by shift
        for _e in _exp:
            _e.sort_values('shift')
            _e.reset_index(inplace=True)

        # and create experiments
        experiments = [
            cls(
                _e['temperature'][0],
                _e['larmor'][0],
                _e['isotope'][0],
                [
                    Signal(
                        signal['shift'],
                        signal['width'],
                        signal['area'],
                        signal['assignment'],
                        l_to_g=signal['L/G']
                    )
                    for _, signal in _e.iterrows()
                ]
            )
            for _e in _exp
        ]

        return experiments

    def to_csv(self, file_name: str, delimiter: str = ',',
               comment: str = '', verbose: bool = True) -> None:
        '''
        Saves Experiment to .csv file

        Parameters
        ----------
        file_name: str
            File name to use
        delimiter: str, default ','
            CSV Delimiter to use
        comment: str, optional
            Additional comment line added to file
        verbose: bool, default True
            If True, prints filename to screen

        Returns
        -------
        None
        '''

        data = {
            'assignment ()': [signal.assignment for signal in self.signals],
            'shift (ppm)': [signal.shift for signal in self.signals],
            'width (Hz)': [signal.width for signal in self.signals],
            'area ()': [signal.area for signal in self.signals],
            'L/G ()': [signal.l_to_g for signal in self.signals]
        }

        df = pd.DataFrame(data=data)
        df.sort_values(['shift (ppm)'], inplace=True)

        _comment = (
            f'#This file was generated with SimpNMR v{__version__}'
            ' at {}\n'.format(
                datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y ')
            )
        )
        _comment += f'#temperature = {self.temperature:.3f}\n'
        _comment += f'#larmor = {self.larmor:.3f}\n'
        _comment += f'#isotope = {self.isotope}\n'

        _comment += comment + '\n'

        with open(file_name, 'w') as _f:
            _f.write(_comment)

            df.to_csv(
                _f,
                sep=delimiter,
                header=True,
                float_format='%.5f',
                index=None
            )

        if verbose:
            ut.cprint(
                f'\n Assigned experiment saved to \n {file_name}\n',
                'cyan'
            )

        return

    @classmethod
    def generate_permutations(cls, experiment: 'Experiment',
                              groups: list[list[str]] = []) -> list['Experiment']: # noqa
        '''
        Generates a set of new experiments in which signal assignments
        are permuted based on a given list of groups

        Parameters
        ----------
        experiment: Experiment
            Experiment used as basis for assignment permutation
        groups: list[list[str]], optional
            Groups of experimental signal labels (assignments) which are to be
            permuted.

        Returns
        -------
        list[Experiment]
            Permuted experiments
        '''

        # Add on fixed assignments by treating each as it were a group
        # of its own
        fixed = [
            [label]
            for label in experiment.keys()
            if label not in np.concatenate(groups)
        ]
        groups += fixed

        # Find all permutations subject to grouping constraints
        _tmp = [
            permutations(group)
            for group in groups
        ]
        perms = [
            list(chain.from_iterable(e))
            for e in product(*_tmp, repeat=1)
        ]

        # Convert label groups into indices of experimental signals
        l2i = {
            label: it
            for it, label in enumerate(experiment.keys())
        }
        group_to_exp = [
            l2i[lab] for lab in np.concatenate(groups)
        ]

        # Order which returns signals listed in groups
        # back to that of original experiment
        order = np.argsort(group_to_exp)

        # Reorder to match original experiment
        all_new_assgn = [
            [new_assgns[o] for o in order]
            for new_assgns in perms
        ]

        return all_new_assgn


class Hyperfine():
    '''
    Contains all information on Hyperfine Coupling of a single atom

    Parameters
    ----------
    tensor: ndarray of floats, default = np.zeros([3,3])
        Hyperfine tensor as 3x3 numpy array of floats
        Assumes units of [ppm Angstrom^-3]

    Attributes
    ----------
    tensor: ndarray of floats, default = np.zeros([3,3])
        Hyperfine tensor as 3x3 numpy array of floats in
        units of [ppm Angstrom^-3]
    iso: float
        Isotropic Hyperfine Coupling Constant in units of [ppm Angstrom^-3]
    dip: ndarray of floats
        Dipolar Hyperfine Coupling array (3x3) in units of [ppm Angstrom^-3]
    eigvals: ndarray of floats
        Eigenvalues of Hyperfine tensor (iso+dip) sorted low to high as 3x1
        np.array in units of [ppm Angstrom^-3]
    eigvecs: ndarray of floats
        Eigenvectors of Hyperfine tensor (iso+dip) in same order as eigenvalues
    '''

    def __init__(self, tensor: NDArray = np.zeros([3, 3])) -> None:
        self._iso = None
        self._dip = None
        self._eigvals = None
        self._eigvecs = None
        self.tensor = copy.deepcopy(tensor)
        pass

    @property
    def tensor(self) -> NDArray:
        '''
        Hyperfine coupling tensor as 3x3 np.array of floats
        Units of [ppm Angstrom^-3]
        '''
        return self._tensor

    @tensor.setter
    def tensor(self, intensor: NDArray):
        if not isinstance(intensor, np.ndarray):
            raise TypeError('A must be np.array (3x3) of floats')
        elif intensor.shape != (3, 3):
            raise TypeError('A must be np.array (3x3) of floats')
        self._tensor = intensor

        # Recalculate isotropic hyperfine
        self.calc_iso()
        # and dipolar hyperfine
        self.calc_dip()

        # and reset eigenvalues and eigenvectors to None
        self._eigvals = None
        self._eigvecs = None
        return

    @property
    def iso(self) -> float:
        '''
        Isotropic Hyperfine coupling value as float
        Units of [ppm Angstrom^-3]
        '''
        return self._iso

    @iso.setter
    def iso(self, val: float):
        self._iso = val
        return

    def calc_iso(self):
        '''
        Calculates isotropic Hyperfine coupling value from self.tensor
        '''
        self.iso = self._calc_iso(self.tensor)
        return

    @staticmethod
    def _calc_iso(tensor: NDArray) -> float:
        '''
        Calculates isotropic Hyperfine coupling value from a given tensor
        '''
        return np.trace(tensor) / 3.

    @property
    def dip(self) -> NDArray:
        '''
        Dipolar Hyperfine coupling tensor as 3x3 np.array of floats
        Units of [ppm Angstrom^-3]
        '''
        return self._dip

    @dip.setter
    def dip(self, tensor: NDArray):
        self._dip = tensor
        return

    def calc_dip(self):
        '''
        Calculates dipolar Hyperfine coupling tensor using self.tensor
        '''
        self.dip = self._calc_dip(self.tensor)
        return

    @staticmethod
    def _calc_dip(tensor: NDArray) -> NDArray:
        '''
        Calculates dipolar Hyperfine coupling tensor from a given tensor
        '''
        return tensor - np.eye(3) * Hyperfine._calc_iso(tensor)

    @property
    def eigvals(self) -> NDArray:
        '''
        Eigenvalues of Hyperfine coupling tensor
        Units of [ppm Angstrom^-3]
        '''
        # Recalculate if not populated
        if self._eigvals is None:
            self.eigvals, self.eigvecs = self.calc_eig()
        return self._eigvals

    @eigvals.setter
    def eigvals(self, value: ArrayLike):
        value = np.asarray(value)
        if np.size(value) != 3 or np.shape(value) != (3,):
            raise TypeError('Values must be 3 element arraylike')
        self._eigvals = value
        return

    @property
    def eigvecs(self) -> NDArray:
        '''
        Eigenvectors of Hyperfine coupling tensor
        Dimensionless
        '''
        # Recalculate if not populated
        if self._eigvecs is None:
            self.eigvals, self.eigvecs = self.calc_eig()
        return self._eigvecs

    @eigvecs.setter
    def eigvecs(self, intensor: NDArray):
        if not isinstance(intensor, np.ndarray):
            raise TypeError('Vectors must be np.array (3x3) of floats')
        elif intensor.shape != (3, 3):
            raise TypeError('Vectors must be np.array (3x3) of floats')
        self._eigvecs = intensor
        return

    @staticmethod
    def calc_pdip(r_nuc: ArrayLike, r_elec: ArrayLike = np.zeros(3)):
        '''
        Calculates point dipole approximation of Dipolar Hyperfine Tensor
        for coupling of a nucleus at r_nuc to an electron at r_elec

        Parameters
        ----------
        r_nuc: array_like
            Coordinates of Nucleus
        r_elec: array_like, default = np.zeros(3)
            Coordinates of electron to which the nucleus couples

        Returns
        -------
        ndarray of floats
            Dipolar Hyperfine Tensor as 3x3 array
        '''

        r_nuc = np.asarray(r_nuc)
        r_elec = np.asarray(r_elec)

        r = r_nuc - r_elec

        rnorm = la.norm(r)

        pdip = (3 * np.outer(r, r) / rnorm**5 - np.eye(3) / rnorm**3)
        pdip /= 4 * np.pi

        return pdip


class Susceptibility:
    '''
    Susceptibility tensor object

    Parameters
    ----------
    tensor: ndarray of floats, default = np.zeros([3,3])
        Susceptibility tensor as 3x3 numpy array of floats
        Assumes units of [Angstrom^3]
    temperature: float
        Temperature that this tensor corresponds to [Kelvin]

    Attributes
    ----------
    tensor: ndarray of floats, default = np.zeros([3,3])
        susceptibility tensor as 3x3 numpy array of floats in units of
        [Angstrom^3]
    iso: float
        Isotropic susceptibility in units of [Angstrom^3]
    dtensor: ndarray of floats
        Delta susceptibility tensor (3x3) in units of [ppm Angstrom^-3]
    eigvals: ndarray of floats
        Eigenvalues of susceptibility tensor sorted low to high as\n
        3x1 np.array in units of [Angstrom^3]
    eigvecs: ndarray of floats
        Eigenvectors of susceptibility tensor in same order as eigenvalues
    alpha: float
        Alpha angle between eigenframe and original tensor frame (ZYZ) [rad]
    beta: float
        Beta angle between eigenframe and original tensor frame (ZYZ) [rad]
    gamma: float
        Gamma angle between eigenframe and original tensor frame (ZYZ) [rad]
    axiality: float
        Axiality value of dtensor [Angstrom^3]
    rhombicity: float
        Axiality value of dtensor [Angstrom^3]
    irred: ndarray of floats
        Irreducible spherical components of Susceptibility Tensor\n
        as 5x1 np.array of floats ordered chi_-2, chi_-1, chi_0, chi_1, chi_2
    temperature: float
        Temperature that this tensor corresponds to [Kelvin]
    '''
    def __init__(self, tensor: NDArray = np.zeros([3, 3]),
                 temperature: float = 0.) -> None:

        self._dtensor = None
        self._iso = None
        self._eigvals = None
        self._eigvecs = None
        self._axiality = None
        self._rhombicity = None
        self._irred = None
        self._alpha = None
        self._beta = None
        self._gamma = None

        self.temperature = temperature
        self.tensor = copy.deepcopy(tensor)
        pass

    @property
    def tensor(self) -> NDArray:
        '''
        Susceptibility Tensor as 3x3 np.array of floats
        Units of [Å^3]
        '''
        return self._tensor

    @tensor.setter
    def tensor(self, intensor: NDArray):
        if not isinstance(intensor, np.ndarray):
            raise TypeError('Chi must be np.array (3x3) of floats')
        elif intensor.shape != (3, 3):
            raise TypeError('Chi must be np.array (3x3) of floats')
        self._tensor = intensor

        # Recalculate isotropic susceptibility
        self.calc_iso()
        # and delta susceptibility
        self.calc_dtensor()

        # and reset eigenvalues and eigenvectors to None
        self._eigvals = None
        self._eigvecs = None
        self._axiality = None
        self._rhombicity = None
        self._irred = None
        self._alpha = None
        self._beta = None
        self._gamma = None
        return

    @property
    def iso(self) -> float:
        '''
        Isotropic Susceptibility value as float
        Units of [Å^3]
        '''
        return self._iso

    @iso.setter
    def iso(self, val: float):
        self._iso = val
        return

    def calc_iso(self):
        '''
        Calculates isotropic susceptibility value from self.tensor
        '''
        self.iso = self._calc_iso(self.tensor)
        return

    @staticmethod
    def _calc_iso(tensor: NDArray) -> float:
        '''
        Calculates isotropic susceptibility value from a given susceptibility
        tensor
        '''
        return np.trace(tensor) / 3.

    @property
    def dtensor(self) -> NDArray:
        '''
        Delta Susceptibility tensor as 3x3 np.array of floats
        Units of [Å^3]
        '''
        if self._dtensor is None:
            self.calc_dtensor()
        return self._dtensor

    @dtensor.setter
    def dtensor(self, tensor: NDArray):
        self._dtensor = tensor
        return

    def calc_dtensor(self):
        '''
        Calculates dtensor Susceptibility tensor using self.tensor
        '''
        self.dtensor = self._calc_dtensor(self.tensor)
        return

    @staticmethod
    def _calc_dtensor(tensor: NDArray) -> NDArray:
        '''
        Calculates dtensor Susceptibility tensor from a given susceptibility
        tensor
        '''
        return tensor - (np.eye(3) * Susceptibility._calc_iso(tensor))

    @property
    def eigvals(self) -> NDArray:
        '''
        Eigenvalues of Susceptibility Tensor
        Units of [Å^3]
        '''
        # Recalculate if not populated
        if self._eigvals is None:
            self.eigvals, self.eigvecs = self.calc_eig()
        return self._eigvals

    @eigvals.setter
    def eigvals(self, value: ArrayLike):
        value = np.asarray(value)
        if np.size(value) != 3 or np.shape(value) != (3,):
            raise TypeError('Values must be 3 element arraylike')
        self._eigvals = value
        return

    @property
    def eigvecs(self) -> NDArray:
        '''
        Eigenvectors of Susceptibility Tensor
        Dimensionless
        '''
        # Recalculate if not populated
        if self._eigvecs is None:
            self.eigvals, self.eigvecs = self.calc_eig()
        return self._eigvecs

    @eigvecs.setter
    def eigvecs(self, intensor: NDArray):
        if not isinstance(intensor, np.ndarray):
            raise TypeError('Vectors must be np.array (3x3) of floats')
        elif intensor.shape != (3, 3):
            raise TypeError('Vectors must be np.array (3x3) of floats')
        self._eigvecs = intensor
        return

    def calc_eig(self):
        '''
        Calculates Eigenvalues and Eigenvectors of Susceptibility Tensor
        using self.tensor
        '''
        vals, vecs = la.eigh(self.tensor)

        self._eigvals = vals[np.argsort(np.abs(vals))]
        self._eigvecs = vecs[np.argsort(np.abs(vals))]
        return vals, vecs

    @property
    def axiality(self) -> float:
        if self._axiality is None:
            self.calc_axiality()
        return self._axiality

    @axiality.setter
    def axiality(self, value: float):
        if not isinstance(value, np.floating):
            raise ValueError('Axiality must be a float')
        else:
            self._axiality = value
        return

    def calc_axiality(self):
        devals = la.eigvalsh(self.dtensor)
        self.axiality = 1.5 * devals[np.argmax(np.abs(devals))]
        return

    @property
    def rhombicity(self) -> float:
        if self._rhombicity is None:
            self.calc_rhombicity()
        return self._rhombicity

    @rhombicity.setter
    def rhombicity(self, value: float):
        if not isinstance(value, np.floating):
            raise ValueError('Rhombicity must be a float')
        else:
            self._rhombicity = value
        return

    def calc_rhombicity(self):
        devals = la.eigvalsh(self.dtensor)
        order = np.argsort(np.abs(devals))
        devals = devals[order]
        self.rhombicity = 0.5 * (devals[0] - devals[1])
        return

    @property
    def alpha(self) -> float:
        '''
        Alpha Euler angle for active ZYZ rotation between tensor input frame
        and tensor eigenframe
        Units of Radians
        '''
        # Calculate if unpopulated
        if self._alpha is None:
            self.calc_euler()
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, np.floating):
            raise ValueError('Alpha must be a float')
        else:
            self._alpha = value
        return

    @property
    def beta(self) -> float:
        '''
        Beta Euler angle for active ZYZ rotation between tensor input frame
        and tensor eigenframe
        Units of Radians
        '''
        # Calculate if unpopulated
        if self._beta is None:
            self.calc_euler()
        return self._beta

    @beta.setter
    def beta(self, value):
        if not isinstance(value, np.floating):
            raise ValueError('Beta must be a float')
        else:
            self._beta = value
        return

    @property
    def gamma(self) -> float:
        '''
        Gamma Euler angle for active ZYZ rotation between tensor input frame
        and tensor eigenframe
        Units of Radians
        '''
        # Calculate if unpopulated
        if self._gamma is None:
            self.calc_euler()
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if not isinstance(value, np.floating):
            raise ValueError('Gamma must be a float')
        else:
            self._gamma = value
        return

    def calc_euler(self):
        '''
        Calculates Euler angles for active ZYZ rotation between\n
        tensor input frame and tensor eigenframe\n
        Units of Radians
        '''
        _ev = np.abs(self.eigvals - self.iso)
        order = np.argsort(_ev)
        _vecs = self.eigvecs[:, order]

        self.alpha = np.rad2deg(np.arctan2(_vecs[2, 1], - _vecs[0, 1]))
        self.beta = np.rad2deg(np.arccos(_vecs[1, 1]))
        self.gamma = np.rad2deg(np.arctan2(-_vecs[1, 2], _vecs[1, 0]))
        return

    @property
    def irred(self) -> NDArray:
        '''
        Irreducible spherical components of Susceptibility Tensor\n
        as 5x1 np.array of floats ordered chi_-2, chi_-1, chi_0, chi_1, chi_2
        '''
        # Calculate if unpopulated
        if self._irred is None:
            self.calc_irred()
        return self._irred

    @irred.setter
    def irred(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                'Irreducible Spherical Components must be 5 element arraylike of complex numbers' # noqa
            )
        elif value.shape != (5,):
            raise TypeError(
                'Irreducible Spherical Components must be 5 element arraylike of complex numbers' # noqa
            )
        elif not np.iscomplexobj(value):
            raise TypeError(
                'Irreducible Spherical Components must be 5 element arraylike of complex numbers' # noqa
            )
        self._irred = value
        return

    def calc_irred(self):
        '''
        Calculates irreducible spherical components of susceptibility tensor\n
        using self.tensor
        '''
        self.irred = self._calc_irred(self.tensor)
        return

    @staticmethod
    def _calc_irred(tensor: NDArray) -> NDArray:
        '''
        Calculates irreducible spherical components of susceptiblity tensor\
        Note this does not include any isotropic contributions

        Parameters
        ----------
        tensor: ndarray of floats
            3x3 real susceptibility tensor

        Returns
        -------
        ndarray of floats
            complex128 array with 5 elements ordered\n
            chi_-2, chi_-1, chi_0, chi_1, chi_2
        '''
        irred = np.zeros(5, dtype=np.complex128)
        # chi_-2
        irred[0] = + np.sqrt(2 * np.pi / 15) * (tensor[0, 0] - tensor[1, 1] + 1j*(tensor[0, 1] + tensor[1, 0])) # noqa
        # chi_-1
        irred[1] = - np.sqrt(2 * np.pi / 15) * (tensor[0, 2] - tensor[2, 0] + 1j*(tensor[1, 2] + tensor[2, 1])) # noqa
        # chi_0
        irred[2] = + np.sqrt(4 * np.pi/45)*(2*tensor[2, 2] - tensor[0, 0] - tensor[1, 1]) # noqa
        # chi_+1
        irred[3] = - np.sqrt(2 * np.pi / 15) * (tensor[0, 2] - tensor[2, 0] - 1j*(tensor[1, 2] + tensor[2, 1])) # noqa
        # chi_2
        irred[4] = + np.sqrt(2 * np.pi / 15) * (tensor[0, 0] - tensor[1, 1] - 1j*(tensor[0, 1] + tensor[1, 0])) # noqa

        return irred

    @classmethod
    def from_csv(cls, file_name: str) -> list['Susceptibility']: # noqa
        '''
        Loads susceptibility information from .csv file\n
        Headers must match those of the file generated by\n
        Susceptibility.save_susc() method
        '''

        data = pd.read_csv(
            file_name,
            skipinitialspace=True,
            index_col=False,
            comment='#'
        )

        # Forward conversion, A^3 --> Key
        convs = {
            '(A^3)': 1.,
            '(Å^3 mol^-1)': constants.Avogadro,
            '(A^3 mol^-1)': constants.Avogadro,
            '(cm^3)': 1E-24,
            '(cm^3 mol^-1)': 1E-24 * constants.Avogadro / (4 * np.pi),
        }

        # Check names of columns and convert units to angstrom cubed
        renamer = {}
        for name in data.keys():
            for unit in convs.keys():
                # Apply conversion backwards
                if unit in name:
                    data[name] /= convs[unit]
                    renamer[name] = name.replace(unit, '(Å^3)')

        # Rename column headers to angstrom cubed
        data.rename(
            renamer,
            inplace=True,
            axis=1
        )

        # Read susceptibility tensor
        suscs = [
            cls(
                np.array([
                    [row['chi_xx (Å^3)'], row['chi_xy (Å^3)'], row['chi_xz (Å^3)']], # noqa
                    [row['chi_xy (Å^3)'], row['chi_yy (Å^3)'], row['chi_yz (Å^3)']], # noqa
                    [row['chi_xz (Å^3)'], row['chi_yz (Å^3)'], row['chi_zz (Å^3)']], # noqa
                ]), temperature=row['Temperature (K)']
            )
            for _, row in data.iterrows()
        ]

        return suscs

    @classmethod
    def from_orca(cls, file_name: str, section: str) -> list['Susceptibility']:
        '''
        Loads susceptibility information from ORCA output file

        Parameters
        ----------
        file_name: str
            ORCA output file containing susceptibility section
        section: str, {'cas', 'nev'}
            Section to extract susceptibility from

        Returns
        -------
        list[Susceptibility]
            Susceptibility tensors, one per temperature
        '''

        # Extract all possible susceptibility tensors from ORCA output file
        tensors = rdrs.read_orca_susceptibility(file_name, section)

        # Orca units of XT are cm3 mol-1 K, so convert to Angstrom^3 K
        conv = 1E-24 * constants.Avogadro / (4 * np.pi)
        conv = 1 / conv

        suscs = [
            cls(tensor / temperature * conv, temperature=temperature)
            for temperature, tensor in tensors.items()
        ]

        return suscs

    def save_pcs_isosurface(self, labels: ArrayLike, coords: ArrayLike,
                            center_atom: str,
                            file_name: str = 'pcs_isosurf.cube',
                            comment='', verbose: bool = True) -> None:
        '''
        Saves the PCS isosurface using the dipolar hyperfine tensor
        calculated in the same way as in `calc_pdip`, incorporating the
        susceptibility tensor.

        Parameters
        ----------
        labels: array_like
            Atomic labels with indexing numbers
        coords: array_like
            Atomic coordinates in Angstrom
        center_atom: str
            Atomic label of the central atom (including indexing number)
        file_name: str, default 'pcs_isosurf.cube'
            Output cube file name
        comment: str, default ''
            Comment line added to the cube file
        verbose: bool, default True
            If True, echo file_name to terminal

        Returns
        -------
        None
        '''

        labels = np.asarray(labels)
        coords = np.asarray(coords)

        upper = 15
        step = 0.5
        lower = -15

        # Create a 3D grid of points where PCS values will be evaluated
        x, y, z = np.meshgrid(
            np.arange(lower, upper + step, step),
            np.arange(lower, upper + step, step),
            np.arange(lower, upper + step, step)
        )

        # Convert coordinates to atomic units (Bohr)
        coords *= 1.88973  

        # Find the central atom's index
        center_idx = np.where(labels == center_atom)[0]
        if len(center_idx) == 0:
            raise ValueError(f'Center atom {center_atom} not found in labels')

        # Shift all coordinates so that the central atom is at the origin
        coords -= coords[center_idx[0]]

        # Ensure that the susceptibility tensor is initialized
        if self.dtensor is None:
            raise ValueError("Susceptibility tensor (dtensor) is not initialized!")

        chi_tensor = self.dtensor  # Load the deviatoric susceptibility tensor (in Å³)

        # Initialize the isosurface array
        isosurf = np.zeros_like(x, dtype=float)

        # Compute PCS values for each grid point
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    r_point = np.array([x[i, j, k], y[i, j, k], z[i, j, k]])

                    # Avoid division by zero by skipping points at the origin
                    if np.allclose(r_point, np.zeros(3)):
                        continue  # Skip calculation if r is zero

                    # Compute the dipolar hyperfine tensor
                    pdip = Hyperfine.calc_pdip(r_point)

                    # Compute the paramagnetic chemical shift (PCS)
                    pcs_value = (1 / 3) * np.trace(chi_tensor @ pdip)
                    isosurf[i, j, k] = pcs_value

        # Scale the PCS values (convert to ppb)
        isosurf *= 1E7  

        # Write the computed PCS isosurface to a cube file
        with open(file_name, 'w') as f:
            f.write(f'{comment}\n')
            f.write('Comment line\n')
            f.write('{:d}   {:.6f} {:.6f} {:.6f}\n'.format(len(labels), lower, lower, lower))
            f.write('{:d}   {:.6f}    0.000000    0.000000\n'.format(x.shape[0], step))
            f.write('{:d}   0.000000    {:.6f}    0.000000\n'.format(y.shape[1], step))
            f.write('{:d}   0.000000    0.000000    {:.6f}\n'.format(z.shape[2], step))

            # Write atomic labels and coordinates
            for lbl, c in zip(labels, coords):
                f.write('{:d}   0.000000  {:.6f} {:.6f} {:.6f}\n'.format(xyzp.lab_to_num(lbl), *c))

            # Write PCS values into the cube file
            a = 0
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        a += 1
                        f.write('{:.5e} '.format(isosurf[i, j, k]))
                        if a == 6:
                            f.write('\n')
                            a = 0
                    f.write('\n')
                    a = 0

        if verbose:
            ut.cprint(f'\n PCS Isosurface written to \n {file_name}\n', 'cyan')
            
        return


class Shift():
    '''
    Contains information on calculated chemical shifts for a given
    nucleus

    Attributes
    ----------
    dia: float
        Diagmagnetic chemical shift (ppm)
    fc: float
        Fermi contact chemical shift (ppm)
    pc: float
        Pseudocontact chemical shift (ppm)
    hf: float
        Hyperfine chemical shift (ppm) = Fermi+Pseduo
    total: float
        Total chemical shift (ppm), sum of diamagnetic, and hyperfine (pc+fc)\n
        contributions
    avg: float
        Averaged total shift (ppm) - defaults to total and is reset to total \n
        if any shift component values are modified
    linewidth: float
        Linewidth of shift signal
    '''
    def __init__(self, dia: float = 0., pc: float = 0.,
                 fc: float = 0., lw: float = 1.) -> None:
        self._pc = dia  # Pseudocontact
        self._fc = pc  # Fermi Contact
        self._dia = fc  # Diamagnetic
        self._lw = lw
        self._avg = copy.copy(self.total)
        pass

    @property
    def total(self) -> float:
        return self.dia + self.hf

    @property
    def hf(self) -> float:
        return self.pc + self.fc

    @property
    def avg(self) -> float:
        return self._avg

    @avg.setter
    def avg(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError('Chemical shift must be a float')
        self._avg = float(val)
        return

    @property
    def pc(self) -> float:
        return self._pc

    @pc.setter
    def pc(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError('Chemical shift must be a float')
        self._pc = float(val)
        self.avg = copy.copy(self.total)
        return

    @property
    def fc(self) -> float:
        return self._fc

    @fc.setter
    def fc(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError('Chemical shift must be a float')
        self._fc = float(val)
        self.avg = copy.copy(self.total)
        return

    @property
    def dia(self) -> float:
        return self._dia

    @dia.setter
    def dia(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError('Diamagnetic chemical shift must be a float')
        self._dia = float(val)
        self.avg = copy.copy(self.total)
        return

    @property
    def lw(self) -> float:
        return self._lw

    @lw.setter
    def lw(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError('Linewidth must be a float')
        self._lw = float(val)
        return

    @staticmethod
    def calc_pcs(A: Hyperfine, chi: 'Susceptibility') -> float:
        '''
        Calculate Pseudocontact part of chemical shift
        '''
        shift = 1. / 3. * np.trace(chi.dtensor @ A.dip)
        return shift

    @staticmethod
    def calc_fcs(A: Hyperfine, chi: 'Susceptibility') -> float:
        '''
        Calculate Fermi Contact part of chemical shift
        # '''
        shift = chi.iso * A.iso

        return shift

    @staticmethod
    def calc_hfs(A: Hyperfine, chi: 'Susceptibility') -> float:
        '''
        Calculate Hyperfine chemical shift
        Sum of Fermi and Pseudocontact terms
        '''

        return Shift.calc_fcs(A, chi) + Shift.calc_pcs(A, chi)


class Nucleus():
    r'''
    Contains all information on a given nucleus of a molecule

    Parameters
    ----------
    label: str
        Atomic label with indexing number e.g. H2
    coords: array_like
        Coordinates of atom
    A: Hyperfine
        Hyperfine object
    shift: Shift, optional
        Shift object. Default is Shift object with all zeros
    chem_label: str, optional
        Optional chemical label for this atom, e.g. tBu3
    chem_math_label: str, optional
        Optional chemical label for this atom, e.g. $\mathregular{tBu_3}$\n
        with mathtext (LaTeX) formatting - used in plots.
    isotope: str
        Isotope of element formatted as nucleon number then symbol e.g. 13C

    Attributes
    ----------
    label: str
        Atomic label with indexing number e.g. H2
    label_nn: str
        Atomic label without indexing number e.g. H
    chem_label: str
        Optional chemical label for this atom, e.g. tBu3
    chem_math_label: str, optional
        Optional chemical label for this atom, e.g. $\mathregular{tBu_3}$\n
        with mathtext (LaTeX) formatting - used in plots.
    coord: ndarray of floats
        Coordinates of nucleus as (1x3) array
    A: Hyperfine
        Hyperfine object containing this Nucleus' hyperfine information
    shift: Shift
        Shift object containing this Nucleus' chemical shift information
    isotope: str
        Isotope of element formatted as nucleon number then symbol e.g. 13C
    ''' # noqa
    def __init__(self, label: str, coord: list[float], A: Hyperfine,
                 shift: Shift = Shift(), chem_label: str = None,
                 chem_math_label: str = None, isotope: str = None) -> None:

        # Label with and without indexing
        self.label = label
        self.label_nn = xyzp.remove_label_indices(self.label)

        # Hyperfine coupling tensor for current nucleus
        self.A = copy.deepcopy(A)

        # Chemical shift
        self.shift = copy.deepcopy(shift)

        # Coordinates of nucleus
        self.coord = coord

        # Chemical labels, normal and mathtext
        if chem_label is None:
            self._chem_label = None
        else:
            self.chem_label = chem_label
        if chem_math_label is None:
            self._chem_math_label = None
        else:
            self.chem_math_label = chem_math_label

        # If isotope is provided then set, else set as default
        if isotope is None:
            self.isotope = ut.DEFAULT_ISOTOPES[self.label_nn]

        return

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, inlabel: str):
        if not isinstance(inlabel, str):
            raise TypeError('label must be string')
        self._label = str(inlabel)
        return

    @property
    def chem_label(self) -> str:
        if self._chem_label is None:
            return self.label
        return self._chem_label

    @chem_label.setter
    def chem_label(self, inchem_label: str):
        if not isinstance(inchem_label, (type(None), str)):
            raise TypeError('chem_label must be string')
        self._chem_label = inchem_label
        return

    @property
    def chem_math_label(self) -> str:
        if self._chem_math_label is None:
            return self.chem_label
        return self._chem_math_label

    @chem_math_label.setter
    def chem_math_label(self, inchem_math_label: str):
        if not isinstance(inchem_math_label, (type(None), str)):
            raise TypeError('chem_math_label must be string')
        self._chem_math_label = inchem_math_label
        return

    @property
    def coord(self) -> NDArray:
        return self._coord

    @coord.setter
    def coord(self, incoord: ArrayLike):

        incoord = np.asarray(incoord)

        if len(incoord.shape) > 1:
            raise ValueError(
                ut.cstr('Nucleus coordinates must be (1x3) array', 'red')
            )

        elif incoord.shape[0] != 3:
            raise ValueError(
                ut.cstr('Nucleus coordinates must be (1x3) array', 'red')
            )
        self._coord = incoord
        return

    @property
    def A(self) -> Hyperfine:
        return self._A

    @A.setter
    def A(self, inA: Hyperfine):
        if not isinstance(inA, Hyperfine):
            raise TypeError('A must be Hyperfine object')
        self._A = inA
        return

    @property
    def shift(self) -> Shift:
        return self._shift

    @shift.setter
    def shift(self, inShift: Shift):
        if not isinstance(inShift, Shift):
            raise TypeError('shift must be a Shift object')
        self._shift = inShift

    @property
    def isotope(self) -> str:
        return self._isotope

    @isotope.setter
    def isotope(self, value: str):
        if re.sub('[0-9]', '', value) != self.label_nn:
            raise ValueError('Isotope label does not match atomic label')
        elif value not in ut.SUPPORTED_ISOTOPES:
            raise ValueError(f'Unsupported isotope {value}')
        else:
            self._isotope = value

    @classmethod
    def from_a_values(cls, a_isos: dict[str, float],
                      a_dips: dict[str, NDArray],
                      coords: NDArray) -> list['Nucleus']:
        '''
        Creates list of Nuclei from dictionaries of isotropic and dipolar
        hyperfine coupling values

        Parameters
        ----------
        a_isos: dict[str, float]
            Keys are atom labels, values are isotropic Hyperfine coupling
            values in units of ppm Angstrom^-3
        a_dips: dict[str, ndarray of floats]
            Keys are atom labels, values are dipolar Hyperfine coupling
            tensors as 3x3 array of floats in units of ppm Angstrom^-3
        coords: ndarray of floats
            Coordinates of each nucleus, order matches that of dicts
        '''

        tensors = {
            label: Hyperfine(a_dips[label] + np.eye(3) * a_isos[label])
            for label in a_dips
        }

        nuclei = [
            cls(key, coord, value)
            for (key, value), coord
            in zip(tensors.items(), coords)
        ]

        if not len(nuclei):
            raise ValueError('No Nuclei selected!')

        return nuclei


class Molecule():
    '''
    Stores information on a given molecule

    Parameters
    ----------
    labels: NDArray[str]
        Atomic labels with no indexing numbers
    coords: NDArray
        Atomic coordinates (xyz) as N_atoms,3 array in Angstrom
    nuclei: list[Nucleus]
        NMR active nuclei in molecule

    Attributes
    ----------
    labels: NDArray[str]
        Atomic labels with no indexing numbers
    coords: NDArray
        Atomic coordinates (xyz) as N_atoms,3 array in Angstrom
    n_atoms: int
        Number of atoms in molecule
    nuclei: list[Nucleus]
        NMR active nuclei in molecule
    susc: Susceptibility
        Susceptibility object for molecule
    '''
    def __init__(self, labels: NDArray[np.str_], coords: NDArray,
                 nuclei: list[Nucleus]) -> None:

        self.labels = xyzp.add_label_indices(labels)
        self.coords = coords

        # List of Nucleus objects
        self.nuclei = nuclei

        # Susceptibility object
        self.susc = copy.deepcopy(Susceptibility())

    pass

    @property
    def n_atoms(self):
        return len(self.labels)

    def __str__(self):

        string = ''

        string += st.title('Molecule Hyperfine Data')

        string += st.subtitle('Isotropic A values (ppm Å^-3)')

        for nuc in self.nuclei:

            if not len(nuc.chem_label):
                label = nuc.label
            else:
                label = f'{nuc.chem_label} ({nuc.label})'

            string += f'{label} {nuc.A.iso: .6f}\n'

        string += st.subtitle('Anisotropic (dipolar) A Tensor (ppm Å^-3)')

        for nuc in self.nuclei:

            if not len(nuc.chem_label):
                label = nuc.label
            else:
                label = f'{nuc.chem_label} ({nuc.label})'

            string += '\n{:} {: .6f} {: .6f} {: .6f}\n'.format(
                ' ' * len(label), *nuc.A.dip[0]
            )
            string += '{:} {: .6f} {: .6f} {: .6f}\n'.format(
                label, *nuc.A.dip[1]
            )
            string += '{:} {: .6f} {: .6f} {: .6f}\n'.format(
                ' ' * len(label), *nuc.A.dip[2]
            )

        return string

    @classmethod
    def from_xyz(cls, xyz_file: str,
                 elements: list[str] | str = 'all') -> 'Molecule':
        '''
        Creates Molecule from an xyz file

        Parameters
        ----------
        xyz_file: str
            File containing xyz coordinates of entire molecule
        elements: list[str] | str
            Elements to include, numbered labels, or just string 'all'

        Returns
        -------
        Molecule
            Molecule object containing specified data
        '''

        # Load xyz file
        labels, coords = xyzp.load_xyz(xyz_file)

        # Generate Molecule using ALL labels and coords
        base = cls.from_labels_coords(labels, coords, elements)

        return base

    @classmethod
    def from_labels_coords(cls, labels: list[str], coords: ArrayLike,
                           elements: list[str] | str = 'all') -> 'Molecule':
        '''
        Creates Molecule from an xyz file

        Parameters
        ----------
        labels: list[str]
            atomic labels
        coords: array_like
            (n_atoms,3) array_like containing xyz coordinates of each atom
        elements: list[str] | str
            Elements to include, numbered labels, or just string 'all'

        Returns
        -------
        Molecule
            Molecule object containing specified data
        '''

        if isinstance(elements, str):
            elements = [elements]

        coords = np.asarray(coords)

        elements_to_include = []
        for ele in elements:
            if ele == 'all':
                elements_to_include = labels
                break
            elif 'all_' in ele or ele in atomic.elements:
                if 'all_' in ele:
                    _e = ele[4:]
                else:
                    _e = ele
                tmp = [
                    la
                    for la in labels
                    if _e == xyzp.remove_label_indices(la)
                ]
                elements_to_include += tmp
            else:
                elements_to_include.append(ele)

        # Generate list of Nuclei, one for each atom
        # selecting only those elements requested by user
        nuclei = [
            Nucleus(label, coord, Hyperfine())
            for label, coord in zip(labels, coords)
            if label in elements_to_include
        ]

        # Generate Molecule using ALL labels and coords
        base = cls(labels, coords, nuclei)

        return base

    @classmethod
    def from_csv(cls, file_name: str,
                 elements: list[str] | str = 'all') -> 'Molecule':
        '''
        Creates Molecule from csv file containing structure, labels,\n
        and tensors

        Parameters
        ----------
        file_name: str
            CSV file contianing structure, labels, and hyperfine data
        elements: str
            Elements to include, numbered labels, or just string 'all'

        Returns
        -------
        Molecule
            Molecule object
        '''

        data = pd.read_csv(
            file_name,
            skipinitialspace=True,
            index_col=False,
            engine='python',
            comment='#'
        )

        required_cols = [
            'atom_label ()',
            'x (Å)',
            'y (Å)',
            'z (Å)'
        ]
        split_hyperfine_cols = [
            'Aiso (ppm Å^-3)',
            'Adip_xx (ppm Å^-3)',
            'Adip_xy (ppm Å^-3)',
            'Adip_xz (ppm Å^-3)',
            'Adip_yy (ppm Å^-3)',
            'Adip_yz (ppm Å^-3)',
            'Adip_zz (ppm Å^-3)'
        ]
        full_hyperfine_cols = [
            'A_xx (ppm Å^-3)',
            'A_xy (ppm Å^-3)',
            'A_xz (ppm Å^-3)',
            'A_yy (ppm Å^-3)',
            'A_yz (ppm Å^-3)',
            'A_zz (ppm Å^-3)'
        ]

        # Standardise column names
        name_convertor = {
            'atom_labels': 'atom_label ()',
            'atom_labels ()': 'atom_label ()',
            'chem_label': 'chem_label ()',
            'chem_labels ()': 'chem_label ()',
            'chem_math_label': 'chem_math_label ()',
            'chem_math_labels ()': 'chem_math_label ()',
            'x': 'x (Å)',
            'x (A)': 'x (Å)',
            'y': 'y (Å)',
            'y (A)': 'y (Å)',
            'z': 'z (Å)',
            'z (A)': 'z (Å)'
        }
        others = {}
        for key, val in name_convertor.items():
            others[key.capitalize()] = val
            others[val.capitalize()] = val
        name_convertor.update(others)
        data.rename(columns=name_convertor, inplace=True)

        missing = [col for col in required_cols if col not in data.columns]
        if any(missing):
            raise ValueError(f'Missing header(s) {missing} in {file_name}')

        if all([col in data.columns for col in split_hyperfine_cols]):
            split = True
        elif all([col in data.columns for col in full_hyperfine_cols]):
            split = False
        else:
            raise ValueError(f'Incomplete hyperfine headers in {file_name}')

        labels = data['atom_label ()']

        elements_to_include = []
        for ele in elements:
            if ele == 'all':
                elements_to_include = labels
                break
            elif 'all_' in ele or ele in atomic.elements:
                if 'all_' in ele:
                    _e = ele[4:]
                else:
                    _e = ele
                tmp = [
                    la
                    for la in labels
                    if _e == xyzp.remove_label_indices(la)
                ]
                elements_to_include += tmp
            else:
                elements_to_include.append(ele)

        # Generate list of Nuclei, one for each atom
        # selecting only those elements requested by user
        coords = np.array([
            data['x (Å)'],
            data['y (Å)'],
            data['z (Å)']
        ])

        if split:
            tensors = [
                np.array([
                    [row['Adip_xx (ppm Å^-3)'], row['Adip_xy (ppm Å^-3)'], row['Adip_xz (ppm Å^-3)']], # noqa
                    [row['Adip_xy (ppm Å^-3)'], row['Adip_yy (ppm Å^-3)'], row['Adip_yz (ppm Å^-3)']], # noqa
                    [row['Adip_xz (ppm Å^-3)'], row['Adip_yz (ppm Å^-3)'], row['Adip_zz (ppm Å^-3)']] # noqa
                ]) + np.eye(3) * row['Aiso (ppm Å^-3)']
                for _, row in data.iterrows()
            ]
        else:
            tensors = [
                np.array([
                    [row['A_xx (ppm Å^-3)'], row['A_xy (ppm Å^-3)'], row['A_xz (ppm Å^-3)']], # noqa
                    [row['A_xy (ppm Å^-3)'], row['A_yy (ppm Å^-3)'], row['A_yz (ppm Å^-3)']], # noqa
                    [row['A_xz (ppm Å^-3)'], row['A_yz (ppm Å^-3)'], row['A_zz (ppm Å^-3)']] # noqa
                ])
                for _, row in data.iterrows()
            ]

        coords = coords.T

        nuclei = [
            Nucleus(label, coord, Hyperfine(tensor))
            for label, coord, tensor in zip(labels, coords, tensors)
            if label in elements_to_include
        ]

        # Add chem labels if present
        if 'chem_label ()' in data.columns:
            for nucleus, (_, row) in zip(nuclei, data.iterrows()):
                nucleus.chem_label = row['chem_label ()']
        if 'chem_math_label ()' in data.columns:
            for nucleus, (_, row) in zip(nuclei, data.iterrows()):
                nucleus.chem_math_label = row['chem_math_label ()']

        # Generate Molecule using ALL labels and coords
        base = cls(labels, coords, nuclei)

        return base

    @classmethod
    def from_QCA(cls, ab_initio: rdrs.QCA, converter: str = 'Null',
                 elements: list[str] | str = 'all') -> 'Molecule':
        '''
        Creates Molecule from ab_initio data, converting A values if requested

        Parameters
        ----------
        ab_initio: rdrs.QCA
            QCA containing ab_initio hyperfine data4
        converter: str {'null', 'mhz_to_ang-3'}
            Unit converter to for ab initio A values (dipolar and isotropic),
            null applies no conversion
        elements: list[str] | str
            Elements to include, numbered labels, ranges, \n
            'all_H', 'H', or just 'all'
        Returns
        -------
        Molecule
            Molecule object containing specified data
        '''

        if isinstance(elements, str):
            elements = [elements]

        elements_to_include = []
        for ele in elements:
            if ele == 'all':
                elements_to_include = copy.copy(ab_initio.labels)
                break
            elif 'all_' in ele or ele in atomic.elements:
                if 'all_' in ele:
                    _e = ele[4:]
                else:
                    _e = ele
                tmp = [
                    la
                    for la in ab_initio.labels
                    if _e == xyzp.remove_label_indices(la)
                ]
                elements_to_include += tmp
            else:
                elements_to_include.append(ele)

        # Convert units
        if converter.lower() != 'null':
            a_isos = ut.a_tensor_mhz_to_angstrom(ab_initio.a_iso)
            a_dips = ut.a_tensor_mhz_to_angstrom(ab_initio.a_dip)
        else:
            a_isos = ab_initio.a_iso
            a_dips = ab_initio.a_dip

        a_isos = {
            key: val
            for key, val in a_isos.items()
            if key in elements_to_include
        }
        a_dips = {
            key: val
            for key, val in a_dips.items()
            if key in elements_to_include
        }

        coords = [
            coord
            for label, coord in zip(ab_initio.labels, ab_initio.coords)
            if label in elements_to_include
        ]

        # Generate list of Nuclei, one for each atom
        nuclei = Nucleus.from_a_values(a_isos, a_dips, coords)

        # Create molecule - uses all atoms, regardless of user labels
        base = cls(ab_initio.labels, ab_initio.coords, nuclei)

        return base

    @property
    def susc(self) -> Susceptibility:
        return self._susc

    @susc.setter
    def susc(self, new_susc: Susceptibility):
        if not isinstance(new_susc, Susceptibility):
            raise TypeError('Molecule.susc must be of type Susceptibility')
        self._susc = new_susc
        return

    def load_diamagnetic_shifts(self, file_name: str, file_type: str = 'csv',
                                ref_file_name: str = '',
                                ref_file_type: str = 'csv') -> None:
        '''
        Load experimental diamagnetic shifts from csv

        Parameters
        ----------
        file_name: str
            Name of csv file from which diamagnetic shifts are read
        file_type: str {'csv', 'dft'}
            Type of file to read.
        ref_file_name: str, optional
            Name of csv file from which reference shifts are read.\n
            If not specified then no reference is subtracted.
        ref_file_type: str, optional {'csv', 'dft'}
            Type of file to read for reference.
        '''

        if file_type == 'csv':
            dia = pd.read_csv(
                file_name, skipinitialspace=True, index_col=False
            )

            if 'atom_label' in dia.keys():
                dia.set_index('atom_label', inplace=True)
                for nuc in self.nuclei:
                    nuc.shift.dia = dia['shift'][nuc.label]
            elif 'chem_label' in dia.keys():
                dia.set_index('chem_label', inplace=True)
                for nuc in self.nuclei:
                    nuc.shift.dia = dia['shift'][nuc.chem_label]
            else:
                raise KeyError('atom_label or chem_label not present in diamagnetic shift file')  # noqa
        elif file_type == 'dft':
            data = rdrs.QCCS.guess_from_file(file_name)

            _relabel = {
                new: old
                for old, new in zip(data.cs_iso.keys(), xyzp.add_label_indices(data.cs_iso.keys())) # noqa
            }

            for nuc in self.nuclei:
                try:
                    nuc.shift.dia = data.cs_iso[_relabel[nuc.label]]
                except KeyError:
                    raise KeyError(f'Cannot find {nuc.label} in reference diamagnetic shift file') # noqa
        else:
            raise ValueError('Unknown file_type')

        if len(ref_file_name):
            if ref_file_type == 'csv':
                ref = pd.read_csv(
                    ref_file_name, skipinitialspace=True, index_col=False
                )

                # Average by nucleus
                ref['atom_label'] = xyzp.remove_label_indices(
                    ref['atom_label']
                )

                ref = ref.groupby('atom_label').mean().reset_index()

                for nuc in self.nuclei:
                    try:
                        nuc.shift.dia = ref['shift'][nuc.label_nn] - nuc.shift.dia # noqa
                    except KeyError:
                        raise KeyError(f'Cannot find {nuc.label_nn} in reference diamagnetic shift file') # noqa

            elif ref_file_type == 'dft':
                ref_data = rdrs.QCCS.guess_from_file(ref_file_name)

                ref_labels = list(ref_data.cs_iso.keys())
                ref_labels_nn = xyzp.remove_label_indices(ref_labels)

                avg_ref_iso = dict.fromkeys(ref_labels_nn, 0)

                for lab, lab_nn in zip(ref_labels, ref_labels_nn):
                    avg_ref_iso[lab_nn] += ref_data.cs_iso[lab]

                for lab_nn in np.unique(ref_labels_nn):
                    avg_ref_iso[lab_nn] /= ref_labels_nn.count(lab_nn)

                for nuc in self.nuclei:
                    try:
                        nuc.shift.dia = avg_ref_iso[nuc.label_nn] - nuc.shift.dia # noqa
                    except KeyError:
                        raise KeyError(f'Cannot find {nuc.label_nn} in reference diamagnetic shift file') # noqa
            else:
                raise ValueError('Unknown file_type')
        return

    def average_shifts(self):
        '''
        Averages total shifts of Nuclei according to their chem_label.\n
        Average values are then stored in Nucleus.avg_shift attribute
        '''

        cl_to_shifts = {nuc.chem_label for nuc in self.nuclei}
        cl_to_shifts = {
            cl: []
            for cl in cl_to_shifts
        }
        for nuc in self.nuclei:
            cl_to_shifts[nuc.chem_label].append(nuc.shift.total)

        cl_to_shifts = {
            cl: np.mean(shifts)
            for cl, shifts in cl_to_shifts.items()
        }
        for nuc in self.nuclei:
            nuc.shift.avg = cl_to_shifts[nuc.chem_label]

        return

    def average_hyperfine(self, av_chemlabels: list[str] | list[list[str]]):
        '''
        Averages hyperfines for specified nuclei

        Parameters
        ----------
        av_chemlabels: list[str] or list[list[str]]
            List of chemical labels specifying nuclei for which averaging\n
            will take place.\n
            list - entries are averaged separately\n
            list of lists - sublists group dissimilar labels which will be \n
            averaged together
        '''

        # Convert all entries into lists
        av_chemlabels = [
            [ent]
            if not isinstance(ent, list)
            else ent
            for ent in av_chemlabels
        ]

        # Check formatting - either list of lists or just list
        # list of lists - sublists group dissimilar labels which will be
        # averaged together
        # list - entries are averaged separately
        if not all(isinstance(ent, (list, str)) for ent in av_chemlabels):
            raise TypeError('Unknown type passed to average_hyperfine, labels should be list[list[str]] or list[str]') # noqa

        # Check sublists are all string
        if any([not isinstance(subent, str) for ent in av_chemlabels for subent in ent]): # noqa
            raise TypeError('Unknown type passed to average_hyperfine, labels should be list[list[str]] or list[str]') # noqa

        # Check labels exist in molecule
        _fl_av_chemlabels = ut.flatten(av_chemlabels)
        all_chemlabels = [nuc.chem_label for nuc in self.nuclei]
        if any([cl not in all_chemlabels for cl in _fl_av_chemlabels]):
            print(set(all_chemlabels).difference(set(_fl_av_chemlabels)))
            raise ValueError('Attempted average using unknown chem_label')

        # Average hyperfines and diamagnetic shifts
        for ents in av_chemlabels:
            avg_atens = np.mean(
                [
                    nuc.A.tensor
                    for nuc in self.nuclei
                    if nuc.chem_label in ents], # noqa
                axis=0
            )
            for nuc in self.nuclei:
                if nuc.chem_label in ents:
                    nuc.A.tensor = avg_atens

        return

    def rotate_hyperfines(self, rot_mat: ArrayLike):
        '''
        Rotates all hyperfine tensors of this molecule using a specified
        rotation matrix.

        This applies the standard second-rank tensor rotation

            A' = R . A . R^T

        where R maps components from the *old* frame into the *new* frame.

        Parameters
        ----------
        rot_mat: array_like
            Rotation matrix R applied to hyperfine tensors as:
            A_new = R @ A_old @ R.T

        Returns
        -------
        None
        '''

        rot_mat = np.asarray(rot_mat)
        if rot_mat.shape != (3, 3):
            raise ValueError('rot_mat must be a (3x3) rotation matrix')

        for nuc in self.nuclei:
            nuc.A.tensor = rot_mat @ nuc.A.tensor @ rot_mat.T

        return

    def calc_pdip(self, centre_labels: list[str]):
        '''
        Calculate point dipole approximation to A_dip for all nuclei
        of the current Molecule

        Parameters
        ----------
        centre_labels: list[str]
            List of paramagnetic centres to which other nuclei couple

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If centre_labels is empty, if multiple of the same label found,\n
            or if no matching labels found
        '''

        if not len(centre_labels):
            raise ValueError(
                ut.cstr(
                    'Error: No paramagnetic centres specified for point dipole', # noqa
                    'red'
                )
            )

        # Find user specified centre(s)
        for centre in centre_labels:
            it = [i for i, x in enumerate(self.labels) if x == centre]

            if len(it) > 1:
                raise ValueError(
                    ut.cstr(
                        'Error: More than one of specified label found',
                        'red'
                    )
                )
            elif not len(it):
                raise ValueError(
                    ut.cstr(f'Cant find {centre} in labels', 'red')
                )

            for nuc in self.nuclei:
                if nuc.label in centre_labels:
                    continue
                val = Hyperfine.calc_pdip(nuc.coord, self.coords[it[0]])
                val *= 1E6 / len(centre_labels)
                nuc.A.tensor += val
        return

    def calculate_shifts(self, shift_terms='full'):
        '''
        Calculates paramagentic chemical shift components for this
        Molecule's Nuclei using susceptibility tensor

        Parameters
        ----------
        shift_terms: list[str] | str {'full', 'pc', 'fc'}
            Shift terms to calculate

        Returns
        -------
        None
        '''

        if isinstance(shift_terms, str):
            shift_terms = [shift_terms]

        # Swap full for actual terms
        shift_terms = [
            nst for st in shift_terms
            for nst in
            (st if st != 'full' else ['pc', 'fc'])
        ]

        if 'pc' in shift_terms:
            for nuc in self.nuclei:
                nuc.shift.pc = Shift.calc_pcs(nuc.A, self.susc)
        if 'fc' in shift_terms:
            for nuc in self.nuclei:
                nuc.shift.fc = Shift.calc_fcs(nuc.A, self.susc)

        if 'fc' not in shift_terms and 'pc' not in shift_terms:
            raise ValueError(ut.cstr('Unknown shift specified', 'red'))

        return

    def add_chem_labels_from_file(self, file_name: str) -> None:
        '''
        Adds chemical label to each nucleus of this molecule
        using the provided csv file

        Parameters
        ----------
        file_name: str
            Name of csv file containing chem_labels\n
            format is 1 header line containing titles atom_label, chem_label\n
            then a label pair on each line

        Returns
        -------
        None
        '''

        _tmp = pd.read_csv(file_name, skipinitialspace=True, comment='#')

        # Check for duplicate atom labels
        if any([val > 1 for val in _tmp['atom_label'].value_counts()]):
            _dupes = _tmp['atom_label'].value_counts().gt(1)
            raise KeyError(
                f'Duplicate Atom label(s) {_dupes} in chemlabels file'
            )

        # Check for missing/empty entries in chem_label
        if any(_tmp['chem_label'].isnull()):
            raise KeyError(
                'Missing chem_label for {}'.format(
                    _tmp[_tmp['chem_label'].isnull()]['atom_label'][0]
                )
            )

        # Check for missing/empty entries in chem_math_label
        if 'chem_math_label' in _tmp.keys():
            if any(_tmp['chem_math_label'].isnull()):
                raise KeyError(
                    'Missing chem_math_label for {}'.format(
                        _tmp[_tmp['chem_math_label'].isnull()]['atom_label'][0]
                    )
                )

        al_to_cl = {
            al: cl
            for al, cl in zip(
                _tmp['atom_label'], _tmp['chem_label']
            )
            if al in [nuc.label for nuc in self.nuclei]
        }

        # Add chem label to each atom
        for nuc in self.nuclei:
            if nuc.label in al_to_cl.keys():
                nuc.chem_label = al_to_cl[nuc.label]

        # Add math label to each atom
        # from supplied math labels
        if 'chem_math_label' in _tmp.keys():
            al_to_cml = {
                al: cl
                for al, cl in zip(_tmp['atom_label'], _tmp['chem_math_label'])
            }
            for nuc in self.nuclei:
                if nuc.label in al_to_cl.keys():
                    nuc.chem_math_label = al_to_cml[nuc.label].lstrip().rstrip() # noqaq
        # or if math labels are not provided, set to the same as math labels
        else:
            for nuc in self.nuclei:
                if not len(nuc.chem_math_label):
                    nuc.chem_math_label = copy.deepcopy(nuc.chem_label)

        # If coordinates are provided in chem_labels file, then check these
        # against the current molecular coordinates
        if all(clab in _tmp.keys() for clab in ['x', 'y', 'z']):
            _tmp.set_index('atom_label')
            for nuc in self.nuclei:
                _coord = [
                    _tmp.loc[nuc.label]['x'],
                    _tmp.loc[nuc.label]['y'],
                    _tmp.loc[nuc.label]['z']
                ]
                diff = np.sum(_coord - nuc.coord)
                if diff > 1e-8:
                    raise ValueError(
                        ut.cstr(
                            f'Coordinates of {nuc.label} in chem_labels file\n'
                            ' do not match those of molecule.',
                            'red'
                        )
                    )

        return

    def save_hyperfines_to_csv(self, file_name: str = 'dft_hyperfines.csv',
                               verbose: bool = True, comment: str = '',
                               delimiter: str = ',') -> None:
        '''
        Saves Hyperfine data to file

        Parameters
        ----------
        file_name: str
            File to which hyperfine data is written in CSV format
        verbose: bool, default True
            If True, echo filename to screen
        comment: str, optional
            Additional comment line INCLUDING comment character
        delimiter, str, default = ','
            CSV delimiter to use
        '''

        data = {
            'atom_label ()': [nuc.label for nuc in self.nuclei],
            'chem_label ()': [nuc.chem_label for nuc in self.nuclei],
            'Aiso (ppm Å^-3)': [nuc.A.iso for nuc in self.nuclei],
            'Adip_xx (ppm Å^-3)': [nuc.A.iso for nuc in self.nuclei],
            'Adip_xy (ppm Å^-3)': [nuc.A.iso for nuc in self.nuclei],
            'Adip_xz (ppm Å^-3)': [nuc.A.iso for nuc in self.nuclei],
            'Adip_yy (ppm Å^-3)': [nuc.A.iso for nuc in self.nuclei],
            'Adip_yz (ppm Å^-3)': [nuc.A.iso for nuc in self.nuclei],
            'Adip_zz (ppm Å^-3)': [nuc.A.iso for nuc in self.nuclei]
        }

        df = pd.DataFrame(data=data)

        _comment = (
            f'#This file was generated with SimpNMR v{__version__}'
            ' at {}\n'.format(
                datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y ')
            )
        )

        _comment += comment + '\n'

        with open(file_name, 'w') as _f:
            _f.write(_comment)

            df.to_csv(
                _f,
                sep=delimiter,
                header=True,
                float_format='%.5f',
                index=None
            )

        if verbose:
            ut.cprint(
                f'\n Molecule data written to \n {file_name}\n',
                'cyan'
            )
        if verbose:
            ut.cprint(
                f'\n Converted Hyperfine data written to \n {file_name}\n',
                'cyan'
            )

        return

    def to_csv(self, file_name: str = 'molecule.csv',
               verbose: bool = True, comment: str = '',
               delimiter: str = ',') -> None:
        '''
        Saves Molecule data to file

        Parameters
        ----------
        file_name: str, defualt 'molecule.csv'
            File to which Molecule data is written in CSV format
        verbose: bool, default True
            If True, echo filename to screen
        comment: str, optional
            Additional comment line INCLUDING comment character
        delimiter, str, default = ','
            CSV delimiter to use
        '''

        data = {
            'atom_label ()': [nuc.label for nuc in self.nuclei],
            'chem_label ()': [nuc.chem_label for nuc in self.nuclei],
            'x (Å)': [nuc.coord[0] for nuc in self.nuclei],
            'y (Å)': [nuc.coord[1] for nuc in self.nuclei],
            'z (Å)': [nuc.coord[2] for nuc in self.nuclei],
            'Aiso (ppm Å^-3)': [nuc.A.iso for nuc in self.nuclei],
            'Adip_xx (ppm Å^-3)': [nuc.A.dip[0, 0] for nuc in self.nuclei],
            'Adip_xy (ppm Å^-3)': [nuc.A.dip[0, 1] for nuc in self.nuclei],
            'Adip_xz (ppm Å^-3)': [nuc.A.dip[0, 2] for nuc in self.nuclei],
            'Adip_yy (ppm Å^-3)': [nuc.A.dip[1, 1] for nuc in self.nuclei],
            'Adip_yz (ppm Å^-3)': [nuc.A.dip[1, 2] for nuc in self.nuclei],
            'Adip_zz (ppm Å^-3)': [nuc.A.dip[2, 2] for nuc in self.nuclei],
            'δ_total_avg (ppm)': [nuc.shift.avg for nuc in self.nuclei],
            'δ_total (ppm)': [nuc.shift.total for nuc in self.nuclei],
            'δ_dia (ppm)': [nuc.shift.dia for nuc in self.nuclei],
            'δ_fc (ppm)': [nuc.shift.fc for nuc in self.nuclei],
            'δ_pc (ppm)': [nuc.shift.pc for nuc in self.nuclei],
            'linewidth (Hz)': [1 for _ in self.nuclei]
        }

        df = pd.DataFrame(data=data)

        _comment = (
            f'# This file was generated with SimpNMR v{__version__}'
            ' at {}\n'.format(
                datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y ')
            )
        )

        _comment += comment + '\n'

        with open(file_name, 'w') as _f:
            _f.write(_comment)

            df.to_csv(
                _f,
                sep=delimiter,
                header=True,
                float_format='%.5f',
                index=None
            )

        if verbose:
            ut.cprint(
                f'\n Molecule data written to \n {file_name}\n',
                'cyan'
            )

        return

    def save_chemcraft_xyz(self, file_name: str, verbose: bool = True):
        '''
        Save xyz file of current structure with chemlabels in format
        used by chemcraft

        Parameters
        ----------
        file_name: str, defualt 'molecule.csv'
            File to which Molecule data is written in CSV format
        verbose: bool, default True
            If True, echo filename to screen
        '''

        _clabs = {nuc.label: nuc.chem_label for nuc in self.nuclei}
        with open(file_name, 'w') as f:
            for lab, trio in zip(self.labels, self.coords):
                f.write('{:5} {:15.7f} {:15.7f} {:15.7f}'.format(xyzp.lab_to_num(lab), *trio)) # noqa
                if lab in _clabs.keys():
                    f.write('      "{}"\n'.format(_clabs[lab]))
                else:
                    f.write('\n')

        if verbose:
            ut.cprint(
                f'\n Molecule CHEMCRAFT .xyz file written to \n {file_name}\n',
                'cyan'
            )
        return

    def save_xyz(self, file_name: str, verbose: bool = True,
                 comment: str = ''):
        '''
        Save xyz file of current structure

        Parameters
        ----------
        file_name: str, defualt 'molecule.csv'
            File to which Molecule data is written in CSV format
        verbose: bool, default True
            If True, echo filename to screen
        comment: str, optional
            Additional comment line
        '''

        _comment = (
            f'This file was generated with SimpNMR v{__version__}'
            ' at {}. '.format(
                datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y')
            )
        )
        _comment += comment

        xyzp.save_xyz(
            file_name,
            labels=self.labels,
            coords=self.coords,
            verbose=False,
            comment=_comment
        )

        if verbose:
            ut.cprint(
                f'\n Molecule .xyz file written to \n {file_name}\n',
                'cyan'
            )
        return
