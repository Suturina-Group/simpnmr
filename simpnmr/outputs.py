'''
This module contains objects and methods for working with paramagnetic nmr
spectra and susceptibility tensors
'''

import datetime
import pandas as pd
import scipy.constants as constants
import numpy as np

from . import utils as ut
from .__version__ import __version__
from . import main
from . import models


def save_susc(molecules: list[main.Molecule],
              file_name: str = 'susceptibility.csv',
              verbose: bool = True,
              susc_models: list[models.SusceptibilityModel] = [],
              susc_units: str = 'A3', delimiter: str = ',', comment: str = ''):
    '''
    Saves susceptibility tensors of a set of Molecules to csv file

    molecules: list[main.Molecule]
        List of molecules, each with an associated susceptibility object
    file_name: str, default 'susceptibility.csv'
        Name of file to save to
    verbose: bool, default True
        If True, print filename to screen
    susc_models: list[models.SusceptibilityModel], optional
        If included, parameter uncertainties will be added to output file
    delimiter: str, default ','
        Delimiter used in csv file
    susc_units: str, {'A3', 'A3 mol-1', 'cm3 mol-1', 'cm3'}
        Units to use for susceptibility
    comment: str, ''
        Additional comment line added to file - must begin with #
    '''

    if susc_units == 'A3':
        conv = 1.
        unit_label = r'Å^3'
    elif susc_units == 'A3 mol-1':
        conv = constants.Avogadro
        unit_label = r'Å^3 mol^-1'
    elif susc_units == 'cm3 ':
        conv = 1E-24
        unit_label = r'cm^3'
    elif susc_units == 'cm3 mol-1':
        conv = 1E-24 * constants.Avogadro / (4 * np.pi)
        unit_label = r'cm^3 mol^-1'

    # And write susceptibility tensor to csv
    out = {
        'Temperature (K)': [molecule.susc.temperature for molecule in molecules], # noqa
        f'chi_iso ({unit_label})': [molecule.susc.iso * conv for molecule in molecules], # noqa
        f'chi_iso-s-dev ({unit_label})': None,
        f'chi_xx ({unit_label})': [molecule.susc.tensor[0, 0] * conv for molecule in molecules], # noqa
        f'chi_xx-s-dev ({unit_label})': None,
        f'chi_xy ({unit_label})': [molecule.susc.tensor[0, 1] * conv for molecule in molecules], # noqa
        f'chi_xy-s-dev ({unit_label})': None,
        f'chi_xz ({unit_label})': [molecule.susc.tensor[0, 2] * conv for molecule in molecules], # noqa
        f'chi_xz-s-dev ({unit_label})': None,
        f'chi_yy ({unit_label})': [molecule.susc.tensor[1, 1] * conv for molecule in molecules], # noqa
        f'chi_yy-s-dev ({unit_label})': None,
        f'chi_yz ({unit_label})': [molecule.susc.tensor[1, 2] * conv for molecule in molecules], # noqa
        f'chi_yz-s-dev ({unit_label})': None,
        f'chi_zz ({unit_label})': [molecule.susc.tensor[2, 2] * conv for molecule in molecules], # noqa
        f'chi_zz-s-dev ({unit_label})': None,
        f'dchi_xx ({unit_label})': [molecule.susc.dtensor[0, 0] * conv for molecule in molecules], # noqa
        f'dchi_xx-s-dev ({unit_label})': None,
        f'dchi_xy ({unit_label})': [molecule.susc.dtensor[0, 1] * conv for molecule in molecules], # noqa
        f'dchi_xy-s-dev ({unit_label})': None,
        f'dchi_xz ({unit_label})': [molecule.susc.dtensor[0, 2] * conv for molecule in molecules], # noqa
        f'dchi_xz-s-dev ({unit_label})': None,
        f'dchi_yy ({unit_label})': [molecule.susc.dtensor[1, 1] * conv for molecule in molecules], # noqa
        f'dchi_yy-s-dev ({unit_label})': None,
        f'dchi_yz ({unit_label})': [molecule.susc.dtensor[1, 2] * conv for molecule in molecules], # noqa
        f'dchi_yz-s-dev ({unit_label})': None,
        f'dchi_zz ({unit_label})': [molecule.susc.dtensor[2, 2] * conv for molecule in molecules], # noqa
        f'dchi_zz-s-dev ({unit_label})': None,
        f'chi_x ({unit_label})': [molecule.susc.eigvals[0] * conv for molecule in molecules], # noqa
        f'chi_x-s-dev ({unit_label})': None,
        f'chi_y ({unit_label})': [molecule.susc.eigvals[1] * conv for molecule in molecules], # noqa
        f'chi_y-s-dev ({unit_label})': None,
        f'chi_z ({unit_label})': [molecule.susc.eigvals[2] * conv for molecule in molecules], # noqa
        f'chi_z-s-dev ({unit_label})': None,
        f'chi_ax ({unit_label})': [molecule.susc.axiality * conv for molecule in molecules], # noqa
        f'chi_ax-s-dev ({unit_label})': None,
        f'chi_rho ({unit_label})': [molecule.susc.rhombicity * conv for molecule in molecules], # noqa
        f'chi_rho-s-dev ({unit_label})': None,
        'alpha (degrees)': [molecule.susc.alpha for molecule in molecules], # noqa
        'alpha-s-dev (degrees)': None,
        'beta (degrees)': [molecule.susc.beta for molecule in molecules], # noqa
        'beta-s-dev (degrees)': None,
        'gamma (degrees)': [molecule.susc.gamma for molecule in molecules], # noqa
        'gamma-s-dev (degrees)': None,
        'r2 ()': None,
        'r2_adjusted ()': None,
        'MAE (ppm)': None,
        'RMSE (ppm)': None,
    }

    if len(susc_models):
        out['r2 ()'] = [model.r2 for model in susc_models]
        out['r2_adjusted ()'] = [model.adj_r2 for model in susc_models]
        out['MAE (ppm)'] = [model.mae for model in susc_models]
        out['RMSE (ppm)'] = [model.rmse for model in susc_models]
        for key in susc_models[0].fit_stdev:
            out[f'chi_{key}-s-dev ({unit_label})'] = [
                model.fit_stdev[key] * conv for model in susc_models
            ]

    to_pop = [key for key, value in out.items() if value is None]
    for pop in to_pop:
        out.pop(pop)

    df = pd.DataFrame(data=out)

    _comment = (
        f'#This file was generated with SimpNMR v{__version__}'
        ' at {}\n'.format(
            datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y ')
        )
    )

    if len(comment):
        if comment[0] != '#':
            comment = f'#{comment}'
        _comment += f'{comment}\n'

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
            f'\n Susceptibility data written to \n {file_name}\n',
            'cyan'
        )

    return

def save_relaxation_decomposition(
        avg_r1_by_chem_label: dict[str, float],
        avg_r2_by_chem_label: dict[str, float],
        avg_lw_by_chem_label: dict[str, float],
        file_name: str,
        avg_dipolar_by_chem_label: dict[str, float] | None = None,
        avg_contact_by_chem_label: dict[str, float] | None = None,
        avg_curie_by_chem_label: dict[str, float] | None = None,
        delimiter: str = ',',
        comment: str = '',
        verbose: bool = True
) -> None:
    '''
    Saves per-chemical-label relaxation rates and linewidths to csv file.

    avg_r1_by_chem_label: dict[str, float]
        Average total R1 rates by chemical label (s^-1)
    avg_r2_by_chem_label: dict[str, float]
        Average total R2 rates by chemical label (s^-1)
    avg_lw_by_chem_label: dict[str, float]
        Average linewidths by chemical label (Hz)
    avg_dipolar_by_chem_label: dict[str, float], optional
        Average SBM dipolar R1 contribution (s^-1)
    avg_contact_by_chem_label: dict[str, float], optional
        Average SBM contact R1 contribution (s^-1)
    avg_curie_by_chem_label: dict[str, float], optional
        Average Curie R1 contribution (s^-1)
    file_name: str
        Path to file to save to
    delimiter: str, default ','
        Delimiter used in csv file
    comment: str, ''
        Additional comment line added to file - must begin with #
    verbose: bool, default True
        If True, print filename to screen
    '''

    # Collect the union of all chemical labels that appear in any dict
    chem_labels: set[str] = set(avg_r1_by_chem_label.keys())
    chem_labels |= set(avg_r2_by_chem_label.keys())
    chem_labels |= set(avg_lw_by_chem_label.keys())

    if avg_dipolar_by_chem_label is not None:
        chem_labels |= set(avg_dipolar_by_chem_label.keys())
    if avg_contact_by_chem_label is not None:
        chem_labels |= set(avg_contact_by_chem_label.keys())
    if avg_curie_by_chem_label is not None:
        chem_labels |= set(avg_curie_by_chem_label.keys())

    chem_labels = sorted(chem_labels)

    # Base columns
    out: dict[str, list] = {
        'chem_label': chem_labels,
        'R1_total (s^-1)': [
            avg_r1_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ],
        'R2_total (s^-1)': [
            avg_r2_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ],
        'linewidth (Hz)': [
            avg_lw_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ],
    }

    # Optional decompositions
    if avg_dipolar_by_chem_label is not None:
        out['R1_sbm_dipolar (s^-1)'] = [
            avg_dipolar_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_contact_by_chem_label is not None:
        out['R1_sbm_contact (s^-1)'] = [
            avg_contact_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_curie_by_chem_label is not None:
        out['R1_curie (s^-1)'] = [
            avg_curie_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]

    df = pd.DataFrame(data=out)

    _comment = (
        f'#This file was generated with SimpNMR v{__version__}'
        ' at {}\n'.format(
            datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y ')
        )
    )

    if len(comment):
        if comment[0] != '#':
            comment = f'#{comment}'
        _comment += f'{comment}\n'

    with open(file_name, 'w') as _f:
        _f.write(_comment)
        df.to_csv(
            _f,
            sep=delimiter,
            header=True,
            float_format='%.5e',
            index=None
        )

    if verbose:
        ut.cprint(
            f'\n Relaxation decomposition written to \n {file_name}\n',
            'cyan'
        )

    return