from . import utils as ut
import sys
import multiprocessing as mp
import yaml
import yaml_include
from abc import ABC, abstractmethod
import copy
import os
from glob import glob
import numpy as np
import csv


class Config(ABC):

    @property
    @abstractmethod
    def REQ_KEYWORDS() -> dict[str, list[str]]:
        'Required keywords and subkeywords'
        raise NotImplementedError

    @property
    @abstractmethod
    def KEYWORDS() -> dict[str, list[str]]:
        'All keywords and subkeywords'
        raise NotImplementedError

    @property
    @abstractmethod
    def KEYWORD_PARTNERS() -> dict[str, list[str]]:
        'Specifies groups of subkeywords which are mututally required'
        raise NotImplementedError

    @classmethod
    def from_file(cls, file_name) -> 'Config':
        '''
        Creates class from input file

        Parameters
        ----------
        file_name: str
            Name of file to read

        Returns
        -------
        Config
            Configuration object
        '''

        yaml.add_constructor(
            "!inc", yaml_include.Constructor(base_dir='.')
        )

        f = open(file_name, 'r')
        parsed = yaml.full_load(f)
        if 'master' in parsed:
            for key, value in parsed['master'].items():
                parsed[key] = value
            parsed.pop('master')

        # Check for unsupported keywords
        unsupported = [key for key in parsed if key not in cls.KEYWORDS]
        # and subkeywords
        unsupported += [
            subkey
            for key in parsed
            for subkey in parsed[key]
            if subkey not in cls.KEYWORDS[key]
        ]
        if any(unsupported):
            for us in unsupported:
                ut.cprint(
                    f'Error: input keyword {us} unknown',
                    'black_yellowbg'
                )
                parsed.pop(us)

        # missing required (mandatory) keywords
        for keyword in cls.REQ_KEYWORDS:
            if keyword not in parsed:
                raise KeyError(f'Error: missing keyword {keyword}')
            for subkeyword in cls.REQ_KEYWORDS[keyword]:
                if subkeyword not in parsed[keyword]:
                    # Allow nuclei:include to be omitted if nuclei:include_groups is provided
                    if keyword == 'nuclei' and subkeyword == 'include':
                        nuclei_block = parsed.get('nuclei', {}) if isinstance(parsed, dict) else {}
                        if isinstance(nuclei_block, dict):
                            include_groups_val = nuclei_block.get('include_groups', [])
                            if include_groups_val not in (None, [], ''):
                                continue
                    raise KeyError(
                        f'Error: missing keyword {keyword}:{subkeyword}'
                    )

        # and missing partner keywords
        # for keyword in parsed:
        #     if keyword not in cls.KEYWORD_PARTNERS:
        #         continue
        #     for subkeyword in cls.KEYWORD_PARTNERS[keyword]:
        #         if subkeyword not in parsed[keyword]:
        #             raise KeyError(
        #                 f'Error: missing keyword {keyword}:{subkeyword}'
        #             )
        _parsed = copy.copy(parsed)
        for key, value in parsed.items():
            if value is None:
                _parsed.pop(key)
        parsed = _parsed

        parsed_to_cls = {
            f'{keyword}_{subkeyword}': parsed[keyword][subkeyword]
            for keyword in parsed
            for subkeyword in parsed[keyword]
        }

        config = cls(**parsed_to_cls)

        return config


class FitSuscConfig(Config):

    REQ_KEYWORDS = {
        'hyperfine': [
            'method',
            'file'
        ],
        'experiment': [
            'files'
        ],
        'assignment': [
            'method',
        ],
        'nuclei': [
            'include'
        ],
        'susc_fit': [
            'type',
            'variables'
        ],
        'project': [
            'name'
        ],
        'chem_labels': [
            'file'
        ]
    }

    KEYWORDS = {
        'hyperfine': [
            'method',
            'file',
            'average',
            'pdip_centres',
            'spin'
        ],
        'experiment': [
            'files'
        ],
        'assignment': [
            'method',
            'groups'
        ],
        'nuclei': [
            'include',
            'include_groups'
        ],
        'susc_fit': [
            'type',
            'variables',
            'average_shifts'
        ],
        'project': [
            'name'
        ],
        'chem_labels': [
            'file'
        ],
        'diamagnetic': [
            'method',
            'file',
        ],
        'diamagnetic_ref': [
            'method',
            'file'
        ]
    }

    KEYWORD_PARTNERS = {
        'hyperfine': [
            'method',
            'file'
        ],
        'assignment': [
            'method',
            'groups',
        ],
        'susc_fit': [
            'type',
            'variables'
        ],
        'diamagnetic': [
            'method',
            'file',
        ],
        'diamagnetic_ref': [
            'method',
            'file'
        ]
    }

    def __init__(self, **kwargs) -> None:

        self._num_threads = 'auto'
        self._hyperfine_method = ''
        self._hyperfine_file = ''
        self._hyperfine_average = []
        self._hyperfine_pdip_centres = []
        self._hyperfine_rotate = []
        self._project_name = ''
        self._experiment_files = []
        self._experiment_spectrum_files = []
        self._diamagnetic_file = ''
        self._diamagnetic_method = ''
        self._diamagnetic_ref_method = ''
        self._diamagnetic_ref_file = ''
        self._assignment_method = ''
        self._assignment_groups = []
        self._nuclei_include = ''
        self._nuclei_include_groups = []
        self._susc_fit_type = ''
        self._susc_fit_variables = ''
        self._susc_fit_average_shifts = []
        self._chem_labels_file = ''
        self._spin_S = None
        self._spin_multiplicity = None
        self._spin_file = ''

        for key in kwargs:
            setattr(self, key, kwargs[key])

        self._resolve_nuclei_include_groups()

        pass
    @property
    def nuclei_include_groups(self) -> list | str:
        return self._nuclei_include_groups

    @nuclei_include_groups.setter
    def nuclei_include_groups(self, values: list | str):
        # Accept a single string or a list of strings representing chem_labels
        if isinstance(values, str):
            self._nuclei_include_groups = [values]
        else:
            self._nuclei_include_groups = list(values)
        # Do not expand here because chem_labels_file may not yet be set; expansion happens in _resolve_nuclei_include_groups()
        return
    def _resolve_nuclei_include_groups(self):
        """Expand nuclei include groups (chem_labels) into atom labels using chem_labels_file.
        Merge the expanded atoms into self._nuclei_include. Remove duplicates preserving order.
        Safe to call multiple times.
        """
        groups = getattr(self, '_nuclei_include_groups', []) or []
        if not groups:
            return
        # If chem_labels_file is not set yet, skip silently
        chem_file = getattr(self, '_chem_labels_file', '')
        if not chem_file:
            return
        expanded_atoms: list[str] = []
        try:
            with open(chem_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    clabel = row.get('chem_label')
                    alabel = row.get('atom_label')
                    if clabel in groups and alabel:
                        expanded_atoms.append(alabel)
        except FileNotFoundError:
            raise FileNotFoundError(f'chem_labels_file not found: {chem_file}')
        except Exception as e:
            raise e
        # Merge with existing nuclei_include
        current = self._nuclei_include
        if isinstance(current, str) and current:
            merged = [current] + expanded_atoms
        elif isinstance(current, list):
            merged = current + expanded_atoms
        elif not current:
            merged = expanded_atoms
        else:
            merged = expanded_atoms
        # Deduplicate preserving order
        seen = set()
        deduped = []
        for x in merged:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        self._nuclei_include = deduped

    @property
    def hyperfine_rotate(self) -> str:
        return self._hyperfine_rotate

    @hyperfine_rotate.setter
    def hyperfine_rotate(self, value: str):
        if isinstance(value, list):
            self._hyperfine_rotate = value[0]
        elif isinstance(value, str):
            self._hyperfine_rotate = value
        else:
            raise ValueError

    @property
    def project_name(self) -> str:
        return self._project_name

    @project_name.setter
    def project_name(self, value: str):
        if isinstance(value, list):
            self._project_name = value[0]
        elif isinstance(value, str):
            self._project_name = value
        else:
            raise ValueError
        return None

    @property
    def hyperfine_file(self) -> list[str]:
        return self._hyperfine_file

    @hyperfine_file.setter
    def hyperfine_file(self, value: list[str]):
        self._hyperfine_file = os.path.abspath(value)
        return None

    @property
    def hyperfine_method(self) -> list[str]:
        return self._hyperfine_method

    @hyperfine_method.setter
    def hyperfine_method(self, value: str):
        if value not in ['dft', 'pdip', 'csv']:
            raise ValueError(f'Unknown hyperfine:method {value}')
        else:
            self._hyperfine_method = value
        return None

    @property
    def hyperfine_average(self) -> list[list[str]]:
        return self._hyperfine_average

    @hyperfine_average.setter
    def hyperfine_average(self, values: list[list[str]]):
        self._hyperfine_average = values
        return

    @property
    def hyperfine_pdip_centres(self) -> list[str]:
        return self._hyperfine_pdip_centres

    @hyperfine_pdip_centres.setter
    def hyperfine_pdip_centres(self, value: list[str]):
        self._hyperfine_pdip_centres = value

    @property
    def susc_fit_type(self) -> bool:
        return self._susc_fit_type

    @susc_fit_type.setter
    def susc_fit_type(self, value: bool):
        self._susc_fit_type = value
        return

    @property
    def num_threads(self) -> int:
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value: list[float]):
        value = int(value[0])
        if value > mp.cpu_count():
            ut.cprint(
                'Warning: Number of threads > system number, resetting',
                'black_yellowbg'
            )
            self._num_threads = mp.cpu_count() - 1
        else:
            self._num_threads = value
        return

    @property
    def assignment_method(self) -> str:
        return self._assignment_method

    @assignment_method.setter
    def assignment_method(self, value: str):
        if value not in ['fixed', 'permute']:
            ut.cprint(f'Unknown assignment:method {value}', 'red')
            sys.exit()
        self._assignment_method = value
        return None

    @property
    def assignment_groups(self) -> list[list[str]]:
        return self._assignment_groups

    @assignment_groups.setter
    def assignment_groups(self, value: list[list[str]]):
        self._assignment_groups = value
        return None

    @property
    def chem_labels_file(self) -> str:
        return self._chem_labels_file

    @chem_labels_file.setter
    def chem_labels_file(self, value: str):
        if not isinstance(value, str):
            raise ValueError('chem_labels_file file should be string')
        self._chem_labels_file = os.path.abspath(value)
        return None

    @property
    def susc_fit_variables(self) -> dict[str, dict[str, float]]:
        return self._susc_fit_variables

    @susc_fit_variables.setter
    def susc_fit_variables(self, value):
        self._susc_fit_variables = value
        return

    @property
    def susc_fit_average_shifts(self) -> list[str]:
        return self._susc_fit_average_shifts

    @susc_fit_average_shifts.setter
    def susc_fit_average_shifts(self, values: list[str]):
        if isinstance(values, str):
            self.susc_fit_average_shifts = [values]
        self._susc_fit_average_shifts = values
        return

    @property
    def nuclei_include(self) -> list | str:
        return self._nuclei_include

    @nuclei_include.setter
    def nuclei_include(self, values: list | str):
        self._nuclei_include = values
        return

    @property
    def experiment_files(self) -> list[str]:
        return self._experiment_files

    @experiment_files.setter
    def experiment_files(self, value: list[str]):
        # Use glob to expand wildcards
        if isinstance(value, list):
            self._experiment_files = [
                glob(os.path.abspath(val))
                if '*' in val
                else os.path.abspath(val)
                for val in value
            ]
            self._experiment_files = np.concatenate(
                [self._experiment_files]
            ).flatten().tolist()

        elif isinstance(value, str):
            if '*' in value:
                value = glob(os.path.abspath(value))
            self._experiment_files = [os.path.abspath(value)]
        else:
            raise ValueError
        return

    @property
    def experiment_spectrum_files(self) -> list[str]:
        return self._experiment_spectrum_files

    @experiment_spectrum_files.setter
    def experiment_spectrum_files(self, value: list[str]):
        if isinstance(value, list):
            self._experiment_spectrum_files = [
                os.path.abspath(val) for val in value
            ]
        elif isinstance(value, str):
            self._experiment_spectrum_files = [os.path.abspath(value)]
        else:
            raise ValueError
        return

    @property
    def diamagnetic_file(self) -> str:
        return self._diamagnetic_file

    @diamagnetic_file.setter
    def diamagnetic_file(self, value: str):
        if not isinstance(value, str):
            raise ValueError('Diamagnetic file should be string')
        self._diamagnetic_file = os.path.abspath(value)
        return

    @property
    def diamagnetic_method(self) -> str:
        return self._diamagnetic_method

    @diamagnetic_method.setter
    def diamagnetic_method(self, value: str):
        if value not in ['dft', 'csv']:
            raise ValueError(f'Unknown diamagnetic:method {value}')
        else:
            self._diamagnetic_method = value
        return

    @property
    def diamagnetic_ref_method(self) -> str:
        return self._diamagnetic_ref_method

    @diamagnetic_ref_method.setter
    def diamagnetic_ref_method(self, value: str):
        if value not in ['dft', 'csv']:
            raise ValueError(f'Unknown diamagnetic_reference:method {value}')
        else:
            self._diamagnetic_ref_method = value
        return

    @property
    def diamagnetic_ref_file(self) -> str:
        return self._diamagnetic_ref_file

    @diamagnetic_ref_file.setter
    def diamagnetic_ref_file(self, value: str):
        if not isinstance(value, str):
            raise ValueError('Diamagnetic reference file should be string')
        self._diamagnetic_ref_file = os.path.abspath(value)
        return

    @property
    def spin_S(self) -> float | None:
        return self._spin_S

    @spin_S.setter
    def spin_S(self, value: float | None):
        self._spin_S = value

    @property
    def spin_multiplicity(self) -> float | None:
        return self._spin_multiplicity

    @spin_multiplicity.setter
    def spin_multiplicity(self, value: float | None):
        self._spin_multiplicity = value
        
    @property
    def spin_file(self) -> str:
        return self._spin_file

    @spin_file.setter
    def spin_file(self, value: str):
        self._spin_file = os.path.abspath(value)

    @property
    def hyperfine_spin(self) -> float | None:
        return self._spin_S
    
    @hyperfine_spin.setter
    def hyperfine_spin(self, value):
        if isinstance(value, (list, tuple)):
            value = value[0]
        try:
            self._spin_S = float(value)
        except Exception:
            raise ValueError(f'Cannot convert hyperfine: spin={value} to float')
        
    @classmethod
    def from_file(cls, file_name) -> 'FitSuscConfig':
        '''
        Creates class from input file

        Parameters
        ----------
        file_name: str
            Name of file to read

        Returns
        -------
        Config
            Configuration object
        '''

        config = super().from_file(file_name)

        if config.assignment_method == 'permute':
            if not len(config.assignment_groups):
                ut.cprint(
                    'Warning, Missing permutation groups in input',
                    'black_yellowbg'
                )
        elif config.assignment_method == 'fixed':
            if len(config.assignment_groups):
                ut.cprint(
                    'Warning, groups provided with fixed assignment',
                    'black_yellowbg'
                )

        return config


class PredictConfig(FitSuscConfig):

    REQ_KEYWORDS = {
        'hyperfine': [
            'method',
            'file'
        ],
        'nuclei': [
            'include',
        ],
        'susceptibility': [
            'file',
            'format',
            'temperatures'
        ],
        'project': [
            'name'
        ]
    }

    KEYWORDS = {
        'hyperfine': [
            'method',
            'file',
            'average',
            'pdip_centres',
            'spin'
        ],
        'experiment': [
            'files',
            'spectrum_files'
        ],
        'nuclei': [
            'include'
        ],
        'project': [
            'name'
        ],
        'chem_labels': [
            'file'
        ],
        'diamagnetic': [
            'method',
            'file',
        ],
        'diamagnetic_ref': [
            'method',
            'file'
        ],
        'susceptibility': [
            'file',
            'format',
            'temperatures'
        ],
        'relaxation': [
            'model',
            'electron_coords',
            'magnetic_field_tesla',
            'temperature',
            'T1e',
            'T2e',
            'tR'
        ]
    }

    def __init__(self, **kwargs):

        self._susceptibility_file = ''
        self._susceptibility_format = ''
        self._susceptibility_temperatures = []
        self._relaxation_model = ''
        self._relaxation_electron_coords = None
        self._relaxation_magnetic_field_tesla = None
        self._relaxation_T1e = None
        self._relaxation_T2e = None
        self._relaxation_tR = None

        super().__init__(**kwargs)

    @property
    def susceptibility_file(self) -> str:
        return self._susceptibility_file

    @susceptibility_file.setter
    def susceptibility_file(self, value: str):
        self._susceptibility_file = os.path.abspath(value)
        return None

    @property
    def susceptibility_format(self) -> str:
        return self._susceptibility_format

    @susceptibility_format.setter
    def susceptibility_format(self, value: str):
        # if value not in ['csv', 'txt', 'orca', 'molcas']:
        if value not in ['csv', 'txt', 'orca_nev', 'orca_cas', 'molcas']:
            raise ValueError(f'Unknown susceptibility_format: {value}')
        else:
            self._susceptibility_format = value
        return None

    @property
    def susceptibility_temperatures(self) -> list[float]:
        return self._susceptibility_temperatures

    @susceptibility_temperatures.setter
    def susceptibility_temperatures(self, value: list[float] | float):
        if isinstance(value, int):
            self._susceptibility_temperatures = [float(value)]
        elif isinstance(value, float):
            self._susceptibility_temperatures = [value]
        elif isinstance(value, list):
            self._susceptibility_temperatures = [float(val) for val in value]
        else:
            raise ValueError(f'Cannot set temperature to {value}')
        return None
    
    @property
    def relaxation_model(self) -> str:
        return self._relaxation_model

    @relaxation_model.setter
    def relaxation_model(self, value: str):
        if value.lower() not in ['sbm', 'curie', 'sbm curie', 'curie sbm']:
            raise ValueError(f'Unknown relaxation: model {value}')
        else:
            self._relaxation_model = value.lower()
        return None
    
    @property
    def relaxation_electron_coords(self) -> list[float]:
        return self._relaxation_electron_coords

    # Relaxation electron coordinates are a list of floats

    @relaxation_electron_coords.setter
    def relaxation_electron_coords(self, value: list[float] | float):
        if value is None:
            raise ValueError(
                f"If 'relaxation' is specified, Cartesian 'electron_coords' must be set")
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                self._relaxation_electron_coords = [
                    float(val) for val in value]
            except Exception:
                raise ValueError(
                    f"Cannot convert electron coordinates {value} to list of floats")
        else:
            raise ValueError(
                f"Electron coordinates must be a list of 3 floats")
        return None

    @property
    def relaxation_magnetic_field_tesla(self) -> float | None:
        return self._relaxation_magnetic_field_tesla

    @relaxation_magnetic_field_tesla.setter
    def relaxation_magnetic_field_tesla(self, value: float | None):
        if value is None:
            self._relaxation_magnetic_field_tesla = float(0.0)
        else:
            try:
                if float(value) < 0:
                    raise ValueError(
                        f'Magnetic field must be zero or positive')
                self._relaxation_magnetic_field_tesla = float(value)
            except:
                raise ValueError(
                    f'Cannot convert magnetic field value {value} to float')
        return None

    @property
    def relaxation_temperature(self) -> float | None:
        return self._relaxation_temperature

    @relaxation_temperature.setter
    def relaxation_temperature(self, value: float | None):
        # Only require temperature if 'curie' is in the relaxation model
        if hasattr(self, '_relaxation_model') and 'curie' in self._relaxation_model:
            if value is None:
                raise ValueError(
                    f"If 'curie' relaxation is specified, 'temperature' must be set")
            try:
                if float(value) <= 0:
                    raise ValueError(f'Temperature must be positive')
                self._relaxation_temperature = float(value)
            except Exception:
                raise ValueError(
                    f"Cannot convert temperature value {value} to float")
        else:
            # If 'curie' is not in the model, temperature is not required
            self._relaxation_temperature = None
        return None

    @property
    def relaxation_T1e(self) -> float | None:
        return self._relaxation_T1e

    @relaxation_T1e.setter
    def relaxation_T1e(self, value: float | None):
        if value is None:
            raise ValueError(
                f"If 'relaxation' is specified, 'T1e' must be set")
        try:
            if float(value) <= 0:
                raise ValueError(f'T1e must be positive')
            self._relaxation_T1e = float(value)
        except Exception:
            raise ValueError(f"Cannot convert T1e value {value} to float")
        return None

    @property
    def relaxation_T2e(self) -> float | None:
        return self._relaxation_T2e

    @relaxation_T2e.setter
    def relaxation_T2e(self, value: float | None):
        if value is None:
            raise ValueError(
                f"If 'relaxation' is specified, 'T2e' must be set")
        try:
            if float(value) <= 0:
                raise ValueError(f'T2e must be positive')
            self._relaxation_T2e = float(value)
        except Exception:
            raise ValueError(f"Cannot convert T2e value {value} to float")
        return None

    @property
    def relaxation_tR(self) -> float | None:
        return self._relaxation_tR

    @relaxation_tR.setter
    def relaxation_tR(self, value: float | None):
        if value is None:
            raise ValueError(f"If 'relaxation' is specified, 'tR' must be set")
        try:
            if float(value) <= 0:
                raise ValueError(f'tR must be positive')
            self._relaxation_tR = float(value)
        except Exception:
            raise ValueError(f"Cannot convert tR value {value} to float")
        return None

    @classmethod
    def from_file(cls, file_name: str) -> 'PredictConfig':
        cls: PredictConfig = super().from_file(file_name)
        return cls


class PlotAConfig(FitSuscConfig):

    REQ_KEYWORDS = {
        'hyperfine': [
            'method',
            'file'
        ],
        'nuclei': [
            'include',
        ],
        'project': [
            'name'
        ]
    }

    KEYWORDS = {
        'hyperfine': [
            'method',
            'file',
            'average',
            'pdip_centres'
        ],
        'nuclei': [
            'include',
            'include_groups'
        ],
        'project': [
            'name'
        ],
        'chem_labels': [
            'file'
        ]
    }

    @property
    def hyperfine_rotate(self) -> str:
        return self._hyperfine_rotate

    @hyperfine_rotate.setter
    def hyperfine_rotate(self, value: str):
        if isinstance(value, list):
            self._hyperfine_rotate = value[0]
        elif isinstance(value, str):
            self._hyperfine_rotate = value
        else:
            raise ValueError

    @property
    def project_name(self) -> str:
        return self._project_name

    @project_name.setter
    def project_name(self, value: str):
        if isinstance(value, list):
            self._project_name = value[0]
        elif isinstance(value, str):
            self._project_name = value
        else:
            raise ValueError
        return None

    @property
    def hyperfine_file(self) -> list[str]:
        return self._hyperfine_file

    @hyperfine_file.setter
    def hyperfine_file(self, value: list[str]):
        self._hyperfine_file = os.path.abspath(value)
        return None

    @property
    def hyperfine_method(self) -> list[str]:
        return self._hyperfine_method

    @hyperfine_method.setter
    def hyperfine_method(self, value: str):
        if value not in ['dft', 'pdip', 'csv']:
            raise ValueError(f'Unknown hyperfine:method {value}')
        else:
            self._hyperfine_method = value
        return None

    @property
    def hyperfine_average(self) -> list[list[str]]:
        return self._hyperfine_average

    @hyperfine_average.setter
    def hyperfine_average(self, values: list[list[str]]):
        self._hyperfine_average = values
        return

    @property
    def hyperfine_pdip_centres(self) -> list[str]:
        return self._hyperfine_pdip_centres

    @hyperfine_pdip_centres.setter
    def hyperfine_pdip_centres(self, value: list[str]):
        self._hyperfine_pdip_centres = value

    @property
    def nuclei_include(self) -> list | str:
        return self._nuclei_include

    @nuclei_include.setter
    def nuclei_include(self, values: list | str):
        self._nuclei_include = values
        return
