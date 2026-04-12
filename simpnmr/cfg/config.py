# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define YAML-backed configuration schemas for application workflows.

Loads and validates input files and exposes typed config objects.
"""

import copy
import csv
import logging
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from glob import glob

import numpy as np
import yaml
import yaml_include

logger = logging.getLogger(__name__)


class Config(ABC):
    @property
    @abstractmethod
    def REQ_KEYWORDS() -> dict[str, list[str]]:
        """Required keywords and subkeywords."""
        raise NotImplementedError

    @property
    @abstractmethod
    def KEYWORDS() -> dict[str, list[str]]:
        """All keywords and subkeywords."""
        raise NotImplementedError

    @property
    @abstractmethod
    def KEYWORD_PARTNERS() -> dict[str, list[str]]:
        """Specifies groups of subkeywords which are mutually required."""
        raise NotImplementedError

    @classmethod
    def from_file(cls, file_name) -> "Config":
        """Creates a configuration object from a YAML input file.

        Args:
            file_name: Path to the YAML file to read.

        Returns:
            A configuration object of type `cls`.

        Raises:
            yaml.YAMLError: If the input YAML file has invalid syntax or structure.
            KeyError: If a required keyword or subkeyword is missing.
            Exception: Propagates any other unexpected I/O or parsing errors.
        """

        yaml.add_constructor("!inc", yaml_include.Constructor(base_dir="."))

        try:
            with open(file_name, "r") as f:
                parsed = yaml.full_load(f)

        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Invalid YAML structure in input file '{file_name}'."
            ) from e

        except Exception as e:
            raise e

        if "master" in parsed:
            for key, value in parsed["master"].items():
                parsed[key] = value
            parsed.pop("master")

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
                logger.warning("Input keyword %s unknown", us)
                parsed.pop(us)

        # missing required (mandatory) keywords
        for keyword in cls.REQ_KEYWORDS:
            if keyword not in parsed:
                raise KeyError(f"Error: missing keyword {keyword}")
            for subkeyword in cls.REQ_KEYWORDS[keyword]:
                if subkeyword not in parsed[keyword]:
                    # Allow nuclei: include to be omitted if
                    # nuclei: include_groups is provided
                    if keyword == "nuclei" and subkeyword == "include":
                        nuclei_block = (
                            parsed.get("nuclei", {}) if isinstance(parsed, dict) else {}
                        )
                        if isinstance(nuclei_block, dict):
                            include_groups_val = nuclei_block.get("include_groups", [])
                            if include_groups_val not in (None, [], ""):
                                continue
                    raise KeyError(f"Error: missing keyword {keyword}:{subkeyword}")

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
            f"{keyword}_{subkeyword}": parsed[keyword][subkeyword]
            for keyword in parsed
            for subkeyword in parsed[keyword]
        }

        config = cls(**parsed_to_cls)

        return config


class FitSuscConfig(Config):
    REQ_KEYWORDS = {
        "hyperfine": ["method", "file"],
        "experiment": ["files"],
        "assignment": [
            "method",
        ],
        "nuclei": ["include"],
        "susc_fit": ["type", "variables"],
        "project": ["name"],
        "chem_labels": ["file"],
    }

    KEYWORDS = {
        "hyperfine": [
            "method",
            "file",
            "average",
            "spin",
            "orbit",
            "total_momentum_J",
            "orbital_contribution",
            "paramagnetic_centre",
        ],
        "experiment": ["files", "spectrum_files", "exp_reference"],
        "assignment": [
            "method",
            "groups",
            "search",
            "area_weight",
            "width_weight",
            "r1_weight",
        ],
        "nuclei": ["include", "include_groups"],
        "susc_fit": ["type", "variables", "input_units", "average_shifts"],
        "project": ["name"],
        "chem_labels": ["file"],
        "diamagnetic": [
            "method",
            "file",
        ],
        "diamagnetic_ref": ["method", "file"],
        "susc_vt": [
            "method",
            "variables",
            "tip_type",
            "ab_initio_file",
            "ab_initio_format",
        ],
        "fit_relaxation": [
            "tau_e_range",
            "tau_r_range",
            "tau_e",
            "tau_r_fixed",
            "tau_r_method",
            "tau_r_solvent",
            "tau_r_eta",
            "tau_r_shell",
            "tau_r_sigma",
        ],
    }

    KEYWORD_PARTNERS = {
        "hyperfine": ["method", "file"],
        "assignment": ["method"],
        "susc_fit": ["type", "variables"],
        "diamagnetic": [
            "method",
            "file",
        ],
        "diamagnetic_ref": ["method", "file"],
    }

    def __init__(self, **kwargs) -> None:
        self._num_threads = "auto"
        self._hyperfine_method = ""
        self._hyperfine_file = ""
        self._hyperfine_average = []
        self._hyperfine_rotate = []
        self._project_name = ""
        self._experiment_files = []
        self._experiment_spectrum_files = []
        self._experiment_exp_reference = None
        self._diamagnetic_file = ""
        self._diamagnetic_method = ""
        self._diamagnetic_ref_method = ""
        self._diamagnetic_ref_file = ""
        self._assignment_method = ""
        self._assignment_groups = []
        self._assignment_search = ""
        self._assignment_n_attempts = None
        self._assignment_max_iter = None
        self._assignment_rmse_threshold = None
        self._assignment_area_weight = 0.0
        self._assignment_width_weight = 0.0
        self._assignment_r1_weight = 0.0
        self._nuclei_include = ""
        self._nuclei_include_groups = []
        self._susc_fit_type = ""
        self._susc_fit_variables = ""
        self._susc_fit_input_units = "A3"
        self._susc_fit_average_shifts = []
        self._chem_labels_file = ""
        self._spin_S = None
        self._spin_multiplicity = None
        self._spin_file = ""
        self._orbit = None
        self._total_momentum_J = None
        self._hyperfine_orbital_contribution = "auto"
        self._hyperfine_paramagnetic_centre = None
        self._susc_vt_method = None
        self._susc_vt_tip_type = None
        self._susc_vt_variables = None
        self._susc_vt_ab_initio_file = None
        self._susc_vt_ab_initio_format = None
        self._fit_relaxation_tau_e_range = None
        self._fit_relaxation_tau_r_range = None
        self._fit_relaxation_tau_e = None
        self._fit_relaxation_tau_r_fixed = None
        self._fit_relaxation_tau_r_method = None
        self._fit_relaxation_tau_r_solvent = None
        self._fit_relaxation_tau_r_eta = None
        self._fit_relaxation_tau_r_shell = None
        self._fit_relaxation_tau_r_sigma = None

        for key in kwargs:
            setattr(self, key, kwargs[key])

        self._resolve_nuclei_include_groups()

        pass

    @property
    def hyperfine_paramagnetic_centre(self) -> list[float] | None:
        return self._hyperfine_paramagnetic_centre

    @hyperfine_paramagnetic_centre.setter
    def hyperfine_paramagnetic_centre(
        self, value: list[float] | tuple[float, float, float] | None
    ):
        if value is None or value == "":
            self._hyperfine_paramagnetic_centre = None
            return None
        if isinstance(value, str):
            value = yaml.safe_load(value)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                self._hyperfine_paramagnetic_centre = [float(val) for val in value]
            except Exception as exc:
                raise ValueError(
                    f"Cannot convert hyperfine:paramagnetic_centre={value} to "
                    "list of 3 floats"
                ) from exc
            return None
        raise ValueError("hyperfine:paramagnetic_centre must be a list of 3 floats")

    @property
    def hyperfine_orbital_contribution(self) -> str:
        """Controls whether ORCA A(ORB) contributions are included in A-tensors.

        Allowed values:
            - 'auto': include A(ORB) only if present in the QC output.
            - 'on': require and include A(ORB); raise if not present.
            - 'off': ignore A(ORB) even if present.
        """
        return self._hyperfine_orbital_contribution

    @hyperfine_orbital_contribution.setter
    def hyperfine_orbital_contribution(self, value: str):
        if isinstance(value, (list, tuple)):
            value = value[0] if value else "auto"
        if value is None or value == "":
            value = "auto"
        if not isinstance(value, str):
            raise ValueError("hyperfine:orbital_contribution must be a string")

        mode = value.strip().lower()
        allowed = {"auto", "on", "off"}
        if mode not in allowed:
            raise ValueError(
                "Unknown hyperfine:orbital_contribution "
                f"{value!r}. Allowed values are: 'auto', 'on', 'off'."
            )

        self._hyperfine_orbital_contribution = mode

    @property
    def experiment_exp_reference(self) -> float | None:
        """Experimental reference position in ppm."""
        return self._experiment_exp_reference

    @experiment_exp_reference.setter
    def experiment_exp_reference(self, value: float | None):
        if value is None or value == "":
            self._experiment_exp_reference = None
            return None
        if isinstance(value, (list, tuple)):
            value = value[0]
        try:
            self._experiment_exp_reference = float(value)
        except Exception:
            raise ValueError(
                f"Cannot convert experiment:exp_reference={value} to float (ppm)"
            )
        return None

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
        return

    def _resolve_nuclei_include_groups(self):
        """Expands `nuclei:include_groups` into atom labels.

        Uses `chem_labels_file` to map `chem_label` values to `atom_label` values.
        The expanded atoms are merged into `self._nuclei_include` with duplicates
        removed while preserving order.

        This method is safe to call multiple times.

        Raises:
            FileNotFoundError: If `chem_labels_file` does not exist.
        """
        raw_groups = getattr(self, "_nuclei_include_groups", [])
        if raw_groups is None:
            raw_groups = []
        if isinstance(raw_groups, str):
            raw_groups = [raw_groups]
        # Normalise groups to stripped strings to avoid whitespace mismatches.
        groups = [str(g).strip() for g in raw_groups if str(g).strip()]
        if not groups:
            return
        # If chem_labels_file is not set yet, skip silently
        chem_file = getattr(self, "_chem_labels_file", "")
        if not chem_file:
            return
        expanded_atoms: list[str] = []
        try:
            with open(chem_file, newline="") as csvfile:
                reader = csv.DictReader(csvfile, skipinitialspace=True)

                # Strip header whitespace by matching keys after .strip().
                def _get(row: dict, key: str):
                    for k, v in row.items():
                        if k is not None and k.strip() == key:
                            return v
                    return None

                for row in reader:
                    clabel = (_get(row, "chem_label") or "").strip()
                    alabel = (_get(row, "atom_label") or "").strip()
                    if clabel in groups and alabel:
                        expanded_atoms.append(alabel)
        except FileNotFoundError:
            raise FileNotFoundError(f"chem_labels_file not found: {chem_file}")
        except Exception as e:
            raise e
        if not expanded_atoms:
            raise ValueError(
                "No nuclei selected: nuclei:include_groups did not match any "
                "chem_label entries in chem_labels_file. "
                f"Requested groups={groups}."
            )
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
        if value not in ["dft", "pdip", "csv"]:
            raise ValueError(f"Unknown hyperfine:method {value}")
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
            logger.error("Number of threads > system number, resetting")
            self._num_threads = mp.cpu_count() - 1
        else:
            self._num_threads = value
        return

    @property
    def assignment_method(self) -> str:
        return self._assignment_method

    @assignment_method.setter
    def assignment_method(self, value: str):
        if value not in ["fixed", "permute", "hungarian"]:
            raise ValueError(f"Unknown assignment:method {value}")
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
    def assignment_search(self) -> str:
        return self._assignment_search

    @assignment_search.setter
    def assignment_search(self, value: dict | None):
        if value is None or value == "":
            self._assignment_search = ""
            self._assignment_n_attempts = None
            self._assignment_max_iter = None
            self._assignment_rmse_threshold = None
            return None

        if not isinstance(value, dict):
            raise ValueError(
                "assignment:search must be a mapping, e.g. search: {mode: balanced}"
            )

        mode_value = value.get("mode")
        if mode_value is None:
            raise ValueError("assignment:search must define 'mode'")
        if not isinstance(mode_value, str):
            raise ValueError("assignment:search:mode must be a string")

        mode = mode_value.strip().lower()
        allowed = {"fast", "balanced", "robust", "custom"}
        if mode not in allowed:
            raise ValueError(
                "Invalid assignment:search:mode '"
                + str(mode_value)
                + "'. Allowed values are: 'fast', 'balanced', 'robust', 'custom'."
            )

        allowed_keys = {"mode", "n_attempts", "max_iter", "rmse_threshold"}
        unknown = set(value) - allowed_keys
        if unknown:
            raise ValueError(
                "assignment:search contains unknown key(s): "
                + ", ".join(sorted(unknown))
            )

        if mode != "custom":
            unexpected = []
            if "n_attempts" in value:
                unexpected.append("n_attempts")
            if "max_iter" in value:
                unexpected.append("max_iter")
            if "rmse_threshold" in value:
                unexpected.append("rmse_threshold")
            if unexpected:
                raise ValueError(
                    "assignment:search only allows n_attempts, max_iter, and "
                    "rmse_threshold when mode is 'custom'; unexpected key(s): "
                    + ", ".join(unexpected)
                )

        self._assignment_search = mode
        self.assignment_n_attempts = value.get("n_attempts")
        self.assignment_max_iter = value.get("max_iter")
        self.assignment_rmse_threshold = value.get("rmse_threshold")
        return None

    @property
    def assignment_n_attempts(self) -> int | None:
        return self._assignment_n_attempts

    @assignment_n_attempts.setter
    def assignment_n_attempts(self, value: int | None):
        if value is None or value == "":
            self._assignment_n_attempts = None
            return None
        if isinstance(value, (list, tuple)):
            value = value[0]
        try:
            ivalue = int(value)
        except Exception as exc:
            raise ValueError(
                f"Cannot convert assignment:n_attempts={value} to int"
            ) from exc
        if ivalue <= 0:
            raise ValueError("assignment:n_attempts must be positive")
        self._assignment_n_attempts = ivalue
        return None

    @property
    def assignment_max_iter(self) -> int | None:
        return self._assignment_max_iter

    @assignment_max_iter.setter
    def assignment_max_iter(self, value: int | None):
        if value is None or value == "":
            self._assignment_max_iter = None
            return None
        if isinstance(value, (list, tuple)):
            value = value[0]
        try:
            ivalue = int(value)
        except Exception as exc:
            raise ValueError(
                f"Cannot convert assignment:max_iter={value} to int"
            ) from exc
        if ivalue <= 0:
            raise ValueError("assignment:max_iter must be positive")
        self._assignment_max_iter = ivalue
        return None

    @property
    def assignment_rmse_threshold(self) -> float | None:
        return self._assignment_rmse_threshold

    @assignment_rmse_threshold.setter
    def assignment_rmse_threshold(self, value: float | None):
        if value is None or value == "":
            self._assignment_rmse_threshold = None
            return None
        if isinstance(value, (list, tuple)):
            value = value[0]
        try:
            fvalue = float(value)
        except Exception as exc:
            raise ValueError(
                f"Cannot convert assignment:rmse_threshold={value} to float"
            ) from exc
        self._assignment_rmse_threshold = fvalue
        return None

    @property
    def assignment_area_weight(self) -> float:
        return self._assignment_area_weight

    @assignment_area_weight.setter
    def assignment_area_weight(self, value: float | None):
        if value is None or value == "":
            self._assignment_area_weight = 0.0
            return None
        if isinstance(value, (list, tuple)):
            value = value[0]
        try:
            fvalue = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot convert assignment:area_weight={value} to float"
            ) from exc
        if fvalue < 0.0:
            raise ValueError("assignment:area_weight must be non-negative")
        self._assignment_area_weight = fvalue
        return None

    @property
    def assignment_width_weight(self) -> float:
        return self._assignment_width_weight

    @assignment_width_weight.setter
    def assignment_width_weight(self, value: float | None):
        if value is None or value == "":
            self._assignment_width_weight = 0.0
            return None
        if isinstance(value, (list, tuple)):
            value = value[0]
        try:
            fvalue = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot convert assignment:width_weight={value} to float"
            ) from exc
        if fvalue < 0.0:
            raise ValueError("assignment:width_weight must be non-negative")
        self._assignment_width_weight = fvalue
        return None

    @property
    def assignment_r1_weight(self) -> float:
        return self._assignment_r1_weight

    @assignment_r1_weight.setter
    def assignment_r1_weight(self, value: float | None):
        if value is None or value == "":
            self._assignment_r1_weight = 0.0
            return None
        if isinstance(value, (list, tuple)):
            value = value[0]
        try:
            fvalue = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot convert assignment:r1_weight={value} to float"
            ) from exc
        if fvalue < 0.0:
            raise ValueError("assignment:r1_weight must be non-negative")
        self._assignment_r1_weight = fvalue
        return None

    @property
    def chem_labels_file(self) -> str:
        return self._chem_labels_file

    @chem_labels_file.setter
    def chem_labels_file(self, value: str):
        if not isinstance(value, str):
            raise ValueError("chem_labels_file file should be string")
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
    def susc_fit_input_units(self) -> str:
        return self._susc_fit_input_units

    @susc_fit_input_units.setter
    def susc_fit_input_units(self, value):
        if value is None or value == "":
            self._susc_fit_input_units = "A3"
            return
        if isinstance(value, (list, tuple)):
            value = value[0] if value else "A3"
        if not isinstance(value, str):
            raise ValueError("susc_fit:input_units must be a string")
        self._susc_fit_input_units = value
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
                glob(os.path.abspath(val)) if "*" in val else os.path.abspath(val)
                for val in value
            ]
            self._experiment_files = (
                np.concatenate([self._experiment_files]).flatten().tolist()
            )

        elif isinstance(value, str):
            if "*" in value:
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
            self._experiment_spectrum_files = [os.path.abspath(val) for val in value]
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
            raise ValueError("Diamagnetic file should be string")
        self._diamagnetic_file = os.path.abspath(value)
        return

    @property
    def diamagnetic_method(self) -> str:
        return self._diamagnetic_method

    @diamagnetic_method.setter
    def diamagnetic_method(self, value: str):
        if value not in ["dft", "csv"]:
            raise ValueError(f"Unknown diamagnetic:method {value}")
        else:
            self._diamagnetic_method = value
        return

    @property
    def diamagnetic_ref_method(self) -> str:
        return self._diamagnetic_ref_method

    @diamagnetic_ref_method.setter
    def diamagnetic_ref_method(self, value: str):
        if value not in ["dft", "csv"]:
            raise ValueError(f"Unknown diamagnetic_reference:method {value}")
        else:
            self._diamagnetic_ref_method = value
        return

    @property
    def diamagnetic_ref_file(self) -> str:
        return self._diamagnetic_ref_file

    @diamagnetic_ref_file.setter
    def diamagnetic_ref_file(self, value: str):
        if not isinstance(value, str):
            raise ValueError("Diamagnetic reference file should be string")
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
            raise ValueError(f"Cannot convert hyperfine: spin={value} to float")

    @property
    def orbit(self) -> float | None:
        return self._orbit

    @orbit.setter
    def orbit(self, value: float | None):
        self._orbit = value

    @property
    def hyperfine_orbit(self) -> float | None:
        return self._orbit

    @hyperfine_orbit.setter
    def hyperfine_orbit(self, value: float | None):
        if value is None:
            self._orbit = None
            return
        try:
            self._orbit = float(value)
        except Exception:
            raise ValueError(f"Cannot convert hyperfine: orbit={value} to float")

    @property
    def total_momentum_J(self) -> float | None:
        return self._total_momentum_J

    @total_momentum_J.setter
    def total_momentum_J(self, value: float | None):
        self._total_momentum_J = value

    @property
    def hyperfine_total_momentum_J(self) -> float | None:
        return self._total_momentum_J

    @hyperfine_total_momentum_J.setter
    def hyperfine_total_momentum_J(self, value: float | None):
        if value is None:
            self._total_momentum_J = None
            return
        try:
            self._total_momentum_J = float(value)
        except Exception:
            raise ValueError(
                f"Cannot convert hyperfine: total momentum J={value} to float"
            )

    @property
    def susc_vt_method(self) -> str | None:
        return self._susc_vt_method

    @susc_vt_method.setter
    def susc_vt_method(self, value: str | None):
        if value is None or value == "":
            self._susc_vt_method = None
            return
        if not isinstance(value, str):
            raise ValueError("susc_vt: method must be a string or None")

        method = value.strip().lower()
        allowed = {"ht_limit", "vt_2nd_order"}
        if method not in allowed:
            raise ValueError(
                "Invalid susc_vt:method '"
                + str(value)
                + "'. Allowed values are: 'vt_2nd_order' or 'ht_limit'."
            )

        self._susc_vt_method = method

    @property
    def susc_vt_tip_type(self) -> str | None:
        return self._susc_vt_tip_type

    @susc_vt_tip_type.setter
    def susc_vt_tip_type(self, value: str | None):
        if value is None or value == "":
            self._susc_vt_tip_type = None
            return
        if not isinstance(value, str):
            raise ValueError("susc_vt: type must be a string or None")

        type = value.strip().lower()
        allowed = {"fit", "fix_tip_from_ab_initio"}
        if type not in allowed:
            raise ValueError(
                "Invalid susc_vt:type '"
                + str(value)
                + "'. Allowed values are: 'fit', or 'fix_tip_from_ab_initio'."
            )

        self._susc_vt_tip_type = type

    @property
    def susc_vt_variables(self) -> dict[str, dict[str, list[object]]] | None:
        return self._susc_vt_variables

    @susc_vt_variables.setter
    def susc_vt_variables(self, value: dict[str, object] | None):
        if value is None or value == "":
            self._susc_vt_variables = None
            return
        if not isinstance(value, dict):
            raise ValueError("susc_vt: variables must be a dict or None")

        required_components = {"iso", "ax", "rh"}
        unknown_components = set(value) - required_components
        if unknown_components:
            raise ValueError(
                "susc_vt: variables contains unknown component(s): "
                + ", ".join(sorted(unknown_components))
            )

        missing_components = required_components - set(value)
        if missing_components:
            raise ValueError(
                "susc_vt: variables is missing component(s): "
                + ", ".join(sorted(missing_components))
            )

        normalised: dict[str, dict[str, list[object]]] = {}
        for comp in required_components:
            block = value.get(comp)
            if not isinstance(block, dict):
                raise ValueError(
                    f"susc_vt: variables component '{comp}' must be a mapping with keys"
                    " 'intercept' and 'slope' (and optional 'tip')"
                )

            missing = {"intercept", "slope"} - set(block)
            if missing:
                raise ValueError(
                    "susc_vt: variables component '"
                    + str(comp)
                    + "' is missing key(s): "
                    + ", ".join(sorted(missing))
                )

            allowed_keys = {"intercept", "slope", "tip"}
            unknown_keys = set(block) - allowed_keys
            if unknown_keys:
                raise ValueError(
                    "susc_vt: variables component '"
                    + str(comp)
                    + "' contains unknown key(s): "
                    + ", ".join(sorted(unknown_keys))
                )

            # 'tip' is only meaningful in TIP fit mode.
            if "tip" in block and getattr(self, "_susc_vt_tip_type", None) != "fit":
                raise ValueError(
                    "susc_vt: variables component '"
                    + str(comp)
                    + "' provides 'tip' but susc_vt:tip_type is not 'fit'. "
                    "Remove the 'tip' entry or set tip_type: fit."
                )

            comp_vars: dict[str, list[object]] = {}
            keys_to_parse = ["intercept", "slope"]
            if "tip" in block:
                keys_to_parse.append("tip")
            for key in keys_to_parse:
                entry = block.get(key)
                if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                    raise ValueError(
                        "susc_vt: variables entries must be 2-item sequences like "
                        "['fit'|'fix', value]; bad entry for '"
                        + str(comp)
                        + ":"
                        + str(key)
                        + "': "
                        + repr(entry)
                    )

                mode, val = entry
                if not isinstance(mode, str):
                    raise ValueError(
                        "susc_vt: variables mode must be a string 'fit' or 'fix'; "
                        "bad mode for '"
                        + str(comp)
                        + ":"
                        + str(key)
                        + "': "
                        + repr(mode)
                    )

                mode_norm = mode.strip().lower()
                if mode_norm not in {"fit", "fix"}:
                    raise ValueError(
                        "susc_vt: variables mode must be 'fit' or 'fix'; bad mode for '"
                        + str(comp)
                        + ":"
                        + str(key)
                        + "': "
                        + repr(mode)
                    )

                try:
                    fval = float(val)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "susc_vt: variables value must be numeric; bad value for '"
                        + str(comp)
                        + ":"
                        + str(key)
                        + "': "
                        + repr(val)
                    ) from exc

                comp_vars[key] = [mode_norm, fval]

            normalised[comp] = comp_vars

        self._susc_vt_variables = normalised

    @property
    def susc_vt_ab_initio_file(self) -> str:
        """Optional susceptibility file used by VT workflows."""
        return self._susc_vt_ab_initio_file

    @susc_vt_ab_initio_file.setter
    def susc_vt_ab_initio_file(self, value: str | None):
        if value is None or value == "":
            self._susc_vt_ab_initio_file = ""
            return None
        if not isinstance(value, str):
            raise ValueError("susc_vt:ab_initio_file must be a string")
        self._susc_vt_ab_initio_file = os.path.abspath(value)
        return None

    @property
    def susc_vt_ab_initio_format(self) -> str:
        """Format of the optional VT susceptibility file."""
        return self._susc_vt_ab_initio_format

    @susc_vt_ab_initio_format.setter
    def susc_vt_ab_initio_format(self, value: str | None):
        if value is None or value == "":
            self._susc_vt_ab_initio_format = ""
            return None
        if not isinstance(value, str):
            raise ValueError("susc_vt:ab_initio_format must be a string")
        fmt = value.strip()
        if fmt not in ["csv", "txt", "orca_nev", "orca_cas", "molcas"]:
            raise ValueError(f"Unknown susc_vt:ab_initio_format {fmt}")
        self._susc_vt_ab_initio_format = fmt
        return None

    @property
    def fit_relaxation_tau_e_range(self) -> list[float] | None:
        """τe plot range [min, max] in seconds, or None for defaults."""
        return self._fit_relaxation_tau_e_range

    @fit_relaxation_tau_e_range.setter
    def fit_relaxation_tau_e_range(self, value):
        if value is None or value == "":
            self._fit_relaxation_tau_e_range = None
            return
        if isinstance(value, str):
            import yaml as _yaml
            value = _yaml.safe_load(value)
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            raise ValueError(
                "fit_relaxation:tau_e_range must be [min, max]"
            )
        lo, hi = float(value[0]), float(value[1])
        if lo <= 0 or hi <= 0 or lo >= hi:
            raise ValueError(
                "fit_relaxation:tau_e_range values must be positive "
                "and min < max"
            )
        self._fit_relaxation_tau_e_range = [lo, hi]

    @property
    def fit_relaxation_tau_r_range(self) -> list[float] | None:
        """τR plot range [min, max] in seconds, or None for defaults."""
        return self._fit_relaxation_tau_r_range

    @fit_relaxation_tau_r_range.setter
    def fit_relaxation_tau_r_range(self, value):
        if value is None or value == "":
            self._fit_relaxation_tau_r_range = None
            return
        if isinstance(value, str):
            import yaml as _yaml
            value = _yaml.safe_load(value)
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            raise ValueError(
                "fit_relaxation:tau_r_range must be [min, max]"
            )
        lo, hi = float(value[0]), float(value[1])
        if lo <= 0 or hi <= 0 or lo >= hi:
            raise ValueError(
                "fit_relaxation:tau_r_range values must be positive "
                "and min < max"
            )
        self._fit_relaxation_tau_r_range = [lo, hi]

    @property
    def fit_relaxation_tau_e(self) -> float | None:
        """Fixed τe (s) for contact-contribution subtraction before r^-6 fit."""
        return self._fit_relaxation_tau_e

    @fit_relaxation_tau_e.setter
    def fit_relaxation_tau_e(self, value):
        if value is None:
            self._fit_relaxation_tau_e = None
            return
        v = float(value)
        if v <= 0:
            raise ValueError("fit_relaxation:tau_e must be positive")
        self._fit_relaxation_tau_e = v

    @property
    def fit_relaxation_tau_r_fixed(self) -> float | None:
        """Fixed τR (s) to overlay as a horizontal line on τ-space plots."""
        return self._fit_relaxation_tau_r_fixed

    @fit_relaxation_tau_r_fixed.setter
    def fit_relaxation_tau_r_fixed(self, value):
        if value is None:
            self._fit_relaxation_tau_r_fixed = None
            return
        v = float(value)
        if v <= 0:
            raise ValueError("fit_relaxation:tau_r_fixed must be positive")
        self._fit_relaxation_tau_r_fixed = v

    @property
    def fit_relaxation_tau_r_method(self) -> str | None:
        """Hydrodynamic model for τ_R calculation ('ellipsoid' or 'beadshell')."""
        return self._fit_relaxation_tau_r_method

    @fit_relaxation_tau_r_method.setter
    def fit_relaxation_tau_r_method(self, value):
        if value is None or value == "":
            self._fit_relaxation_tau_r_method = None
            return
        if value not in ("ellipsoid", "beadshell"):
            raise ValueError(
                "fit_relaxation:tau_r_method must be 'ellipsoid' or 'beadshell'"
            )
        self._fit_relaxation_tau_r_method = value

    @property
    def fit_relaxation_tau_r_solvent(self) -> str | None:
        """Solvent name for viscosity lookup when computing τ_R."""
        return self._fit_relaxation_tau_r_solvent

    @fit_relaxation_tau_r_solvent.setter
    def fit_relaxation_tau_r_solvent(self, value):
        if value is None or value == "":
            self._fit_relaxation_tau_r_solvent = None
            return
        self._fit_relaxation_tau_r_solvent = str(value)

    @property
    def fit_relaxation_tau_r_eta(self) -> float | None:
        """Explicit solvent viscosity (Pa·s) for τ_R calculation, overrides solvent."""
        return self._fit_relaxation_tau_r_eta

    @fit_relaxation_tau_r_eta.setter
    def fit_relaxation_tau_r_eta(self, value):
        if value is None or value == "":
            self._fit_relaxation_tau_r_eta = None
            return
        v = float(value)
        if v <= 0:
            raise ValueError("fit_relaxation:tau_r_eta must be positive")
        self._fit_relaxation_tau_r_eta = v

    @property
    def fit_relaxation_tau_r_shell(self) -> float | None:
        """Solvent shell thickness (Å) added to vdW radii when computing τ_R."""
        return self._fit_relaxation_tau_r_shell

    @fit_relaxation_tau_r_shell.setter
    def fit_relaxation_tau_r_shell(self, value):
        if value is None or value == "":
            self._fit_relaxation_tau_r_shell = None
            return
        v = float(value)
        if v < 0:
            raise ValueError("fit_relaxation:tau_r_shell must be non-negative")
        self._fit_relaxation_tau_r_shell = v

    @property
    def fit_relaxation_tau_r_sigma(self) -> float | None:
        """Minibead radius (Å) for the bead-shell τ_R model."""
        return self._fit_relaxation_tau_r_sigma

    @fit_relaxation_tau_r_sigma.setter
    def fit_relaxation_tau_r_sigma(self, value):
        if value is None or value == "":
            self._fit_relaxation_tau_r_sigma = None
            return
        v = float(value)
        if v <= 0:
            raise ValueError("fit_relaxation:tau_r_sigma must be positive")
        self._fit_relaxation_tau_r_sigma = v

    @classmethod
    def from_file(cls, file_name) -> "FitSuscConfig":
        """Creates a `FitSuscConfig` from a YAML input file.

        Args:
            file_name: Path to the YAML file to read.

        Returns:
            A populated `FitSuscConfig` instance.
        """

        config = super().from_file(file_name)

        # If an optional VT susceptibility file is provided, require a ab_initio_format.
        if getattr(config, "susc_vt_ab_initio_file", ""):
            if not getattr(config, "susc_vt_ab_initio_format", ""):
                raise ValueError(
                    " Invalid VT configuration: 'susc_vt:ab_initio_file' was provided "
                    "but 'susc_vt:ab_initio_format' is missing."
                )

        if config.susc_vt_method == "ht_limit" and config.susc_vt_variables is not None:
            raise ValueError(
                " Invalid VT configuration: method 'ht_limit' "
                "does not use 'susc_vt:variables' "
                "or the optional susceptibility input ('susc_vt:ab_initio_file/"
                "ab_initio_format'). "
                "Remove the 'variables' block (no linear intercept/slope "
                "fitting is performed in ht_limit).\n"
            )

        if config.susc_vt_method == "vt_2nd_order" and config.susc_vt_variables is None:
            if config.susc_vt_tip_type == "fit":
                logger.warning(
                    "'susc_vt:variables' not provided. Using defaults: "
                    "VT Intercept / Slope and TIP set to ['fit', 0.0]."
                )
                config.susc_vt_variables = {
                    "iso": {
                        "intercept": ["fit", 0.0],
                        "slope": ["fit", 0.0],
                        "tip": ["fit", 0.0],
                    },
                    "ax": {
                        "intercept": ["fit", 0.0],
                        "slope": ["fit", 0.0],
                        "tip": ["fit", 0.0],
                    },
                    "rh": {
                        "intercept": ["fit", 0.0],
                        "slope": ["fit", 0.0],
                        "tip": ["fit", 0.0],
                    },
                }
            else:
                logger.warning(
                    "'susc_vt:variables' not provided. Using defaults: "
                    "VT Intercept / Slope set to ['fit', 0.0]."
                )
                config.susc_vt_variables = {
                    "iso": {
                        "intercept": ["fit", 0.0],
                        "slope": ["fit", 0.0],
                    },
                    "ax": {
                        "intercept": ["fit", 0.0],
                        "slope": ["fit", 0.0],
                    },
                    "rh": {
                        "intercept": ["fit", 0.0],
                        "slope": ["fit", 0.0],
                    },
                }

        if (
            config.susc_vt_tip_type is None
            and config.susc_vt_ab_initio_file is not None
        ):
            raise ValueError(
                " Invalid VT configuration: TIP type is not provided, "
                "Therefore 'susc_vt:ab_initio_file' variable can not be used"
            )

        # exp_reference requires spectrum_files
        if getattr(
            config, "experiment_exp_reference", None
        ) is not None and not getattr(config, "experiment_spectrum_files", []):
            raise ValueError(
                "Invalid experiment configuration: 'experiment:exp_reference' was "
                "provided but no 'experiment:spectrum_files' were specified."
            )

        if config.assignment_method == "permute":
            if not len(config.assignment_groups):
                logger.warning("Missing permutation groups in input")
            if config.assignment_search:
                logger.warning(
                    "Ignoring Hungarian-only assignment:search mapping for "
                    "assignment method 'permute'"
                )

        elif config.assignment_method == "fixed":
            if len(config.assignment_groups):
                logger.info("Chemical groups (signals) provided with fixed assignment")
            if config.assignment_search:
                logger.warning(
                    "Ignoring Hungarian-only assignment:search mapping for "
                    "assignment method 'fixed'"
                )

        elif config.assignment_method == "hungarian":
            if len(config.assignment_groups):
                raise ValueError(
                    "assignment:groups is not supported when "
                    "assignment:method is 'hungarian'"
                )

        return config


class PredictConfig(FitSuscConfig):
    REQ_KEYWORDS = {
        "hyperfine": ["method", "file"],
        "nuclei": [
            "include",
        ],
        "susceptibility": ["temperatures"],
        "project": ["name"],
    }

    KEYWORDS = {
        "hyperfine": [
            "method",
            "file",
            "average",
            "spin",
            "orbit",
            "total_momentum_J",
            "orbital_contribution",
            "paramagnetic_centre",
        ],
        "experiment": ["files", "spectrum_files", "exp_reference"],
        "nuclei": ["include", "include_groups"],
        "project": ["name"],
        "chem_labels": ["file"],
        "diamagnetic": [
            "method",
            "file",
        ],
        "diamagnetic_ref": ["method", "file"],
        "susceptibility": ["file", "format", "temperatures"],
        "relaxation": [
            "model",
            "temperature",
            "magnetic_field_tesla",
            "T1e",
            "T2e",
            "tR",
            "min_linewidth_hz",
        ],
    }

    def __init__(self, **kwargs):
        self._susceptibility_file = None
        self._susceptibility_format = None
        self._susceptibility_temperatures = []
        self._relaxation_model = ""
        self._hyperfine_paramagnetic_centre = None
        self._relaxation_temperature = None
        self._relaxation_magnetic_field_tesla = None
        self._relaxation_T1e = None
        self._relaxation_T2e = None
        self._relaxation_tR = None
        self._relaxation_min_linewidth_hz = 0.0

        super().__init__(**kwargs)

    @property
    def susceptibility_file(self) -> str | None:
        return self._susceptibility_file

    @susceptibility_file.setter
    def susceptibility_file(self, value: str | None):
        if value is None or value == "":
            self._susceptibility_file = None
            return None
        if not isinstance(value, str):
            raise ValueError("susceptibility:file must be a string or None")
        self._susceptibility_file = os.path.abspath(value)
        return None

    @property
    def susceptibility_format(self) -> str | None:
        return self._susceptibility_format

    @susceptibility_format.setter
    def susceptibility_format(self, value: str | None):
        if value is None or value == "":
            self._susceptibility_format = None
            return None
        fmt = value.strip()
        if fmt not in ["csv", "txt", "orca_nev", "orca_cas", "molcas"]:
            raise ValueError(f"Unknown susceptibility_format: {value}")
        self._susceptibility_format = fmt
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
            raise ValueError(f"Cannot set temperature to {value}")
        return None

    @property
    def relaxation_model(self) -> str:
        return self._relaxation_model

    @relaxation_model.setter
    def relaxation_model(self, value: str):
        if value.lower() not in ["sbm", "curie", "sbm curie", "curie sbm"]:
            raise ValueError(f"Unknown relaxation: model {value}")
        else:
            self._relaxation_model = value.lower()
        return None

    @property
    def hyperfine_paramagnetic_centre(self) -> list[float] | None:
        return self._hyperfine_paramagnetic_centre

    @hyperfine_paramagnetic_centre.setter
    def hyperfine_paramagnetic_centre(
        self, value: list[float] | tuple[float, float, float] | None
    ):
        if value is None or value == "":
            self._hyperfine_paramagnetic_centre = None
            return None
        if isinstance(value, str):
            value = yaml.safe_load(value)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                self._hyperfine_paramagnetic_centre = [float(val) for val in value]
            except Exception as exc:
                raise ValueError(
                    f"Cannot convert hyperfine:paramagnetic_centre={value} to "
                    "list of 3 floats"
                ) from exc
            return None
        raise ValueError("hyperfine:paramagnetic_centre must be a list of 3 floats")

    @property
    def relaxation_temperature(self) -> float | None:
        return self._relaxation_temperature

    @relaxation_temperature.setter
    def relaxation_temperature(self, value: float | None):
        if value is None or value == "":
            self._relaxation_temperature = None
            return None
        if isinstance(value, (list, tuple)):
            value = value[0]
        try:
            temperature = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot convert relaxation:temperature value {value} to float"
            ) from exc
        if temperature <= 0:
            raise ValueError("relaxation:temperature must be positive")
        self._relaxation_temperature = temperature
        return None

    @property
    def relaxation_magnetic_field_tesla(self) -> float | None:
        return self._relaxation_magnetic_field_tesla

    @relaxation_magnetic_field_tesla.setter
    def relaxation_magnetic_field_tesla(self, value: float | None):
        if value is None or value == "":
            self._relaxation_magnetic_field_tesla = None
            return None
        if isinstance(value, (list, tuple)):
            value = value[0]
        try:
            field = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot convert relaxation:magnetic_field_tesla value {value} to float"
            ) from exc
        if field < 0:
            raise ValueError("relaxation:magnetic_field_tesla must be zero or positive")
        self._relaxation_magnetic_field_tesla = field
        return None

    @property
    def relaxation_T1e(self) -> float | None:
        return self._relaxation_T1e

    @relaxation_T1e.setter
    def relaxation_T1e(self, value: float | None):
        if value is None:
            raise ValueError("If 'relaxation' is specified, 'T1e' must be set")
        try:
            if float(value) <= 0:
                raise ValueError("T1e must be positive")
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
            raise ValueError("If 'relaxation' is specified, 'T2e' must be set")
        try:
            if float(value) <= 0:
                raise ValueError("T2e must be positive")
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
            raise ValueError("If 'relaxation' is specified, 'tR' must be set")
        try:
            if float(value) <= 0:
                raise ValueError("tR must be positive")
            self._relaxation_tR = float(value)
        except Exception:
            raise ValueError(f"Cannot convert tR value {value} to float")
        return None

    @property
    def relaxation_min_linewidth_hz(self) -> float:
        return self._relaxation_min_linewidth_hz

    @relaxation_min_linewidth_hz.setter
    def relaxation_min_linewidth_hz(self, value):
        if value is None:
            self._relaxation_min_linewidth_hz = 0.0
        else:
            v = float(value)
            if v < 0:
                raise ValueError("min_linewidth_hz must be non-negative")
            self._relaxation_min_linewidth_hz = v

    @classmethod
    def from_file(cls, file_name: str) -> "PredictConfig":
        """Creates a `PredictConfig` from a YAML input file.

        Args:
            file_name: Path to the YAML file to read.

        Returns:
            A populated `PredictConfig` instance.
        """
        config: PredictConfig = super().from_file(file_name)

        if config.relaxation_model and config.hyperfine_paramagnetic_centre is None:
            raise ValueError(
                "If 'relaxation' is specified, 'hyperfine:paramagnetic_centre' "
                "must be set"
            )

        if config.susceptibility_format and not config.susceptibility_file:
            logger.warning(
                "Ignoring susceptibility:format because no susceptibility:file was "
                "provided."
            )

        return config


class FitCorrTimeConfig(FitSuscConfig):
    REQ_KEYWORDS = {
        "hyperfine": ["method", "file"],
        "nuclei": [
            "include",
        ],
        "experiment": ["files"],
        "fit_corr_time": [
            "tau_R",
            "tau_E",
        ],
        "relaxation": [
            "model",
        ],
        "project": ["name"],
        "chem_labels": ["file"],
    }

    KEYWORDS = {
        "hyperfine": [
            "method",
            "file",
            "average",
            "spin",
            "orbit",
            "total_momentum_J",
            "paramagnetic_centre",
        ],
        "nuclei": ["include", "include_groups"],
        "experiment": ["files"],
        "fit_corr_time": [
            "tau_R",
            "tau_E",
        ],
        "relaxation": [
            "model",
        ],
        "project": ["name"],
        "chem_labels": ["file"],
    }

    def __init__(self, **kwargs):
        self._fit_corr_time_tau_R = None
        self._fit_corr_time_tau_E = None
        self._fit_corr_time_fix = ""
        self._relaxation_model = ""
        self._hyperfine_paramagnetic_centre = None

        super().__init__(**kwargs)

    @property
    def fit_corr_time_tau_R(self) -> list:
        return self._fit_corr_time_tau_R

    @fit_corr_time_tau_R.setter
    # Accept value as a list: [fit/fix, guess, [upper-bound, lower-bound]]
    def fit_corr_time_tau_R(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                "tau_R must take the form: [fit/fix, guess, "
                "[lower-bound, upper-bound]], with bounds optional"
            )
        if len(value) < 2:
            raise ValueError(
                "tau_R must take the form: [fit/fix, guess, "
                "[lower-bound, upper-bound]], with bounds optional"
            )
        mode = value[0].lower()
        if mode not in ["fit", "fix"]:
            raise ValueError('tau_R first element must be "fit" or "fix"')
        try:
            guess = float(value[1])
        except Exception:
            raise ValueError(f"Cannot convert tau_R guess value {value[1]} to float")
        if guess <= 0:
            raise ValueError("tau_R guess must be positive")
        bounds = value[2] if len(value) == 3 else None
        if bounds is not None:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(
                    "tau_R bounds must be a list: [upper-bound, lower-bound]"
                )
            try:
                lower = float(bounds[0])
                upper = float(bounds[1])
            except Exception:
                raise ValueError(f"Cannot convert tau_R bounds {bounds} to floats")
            if upper <= lower:
                raise ValueError("tau_R upper bound must be greater than lower bound")
            if lower <= 0 or upper <= 0:
                raise ValueError("tau_R bounds must be positive")
            self._fit_corr_time_tau_R = [mode, guess, [lower, upper]]
        if mode == "fix" and bounds is not None:
            raise ValueError("Remove bounds if correlation time is fixed.")
        else:
            self._fit_corr_time_tau_R = [mode, guess]
        return None

    @property
    def fit_corr_time_tau_E(self) -> list:
        return self._fit_corr_time_tau_E

    @fit_corr_time_tau_E.setter
    # Accept value as a list: [fit/fix, guess, [upper-bound, lower-bound]]
    def fit_corr_time_tau_E(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                "tau_E must take the form: "
                "[fit/fix, guess, [upper-bound, lower-bound]], with bounds optional"
            )
        if len(value) < 2:
            raise ValueError(
                "tau_E must take the form: "
                "[fit/fix, guess, [upper-bound, lower-bound]], with bounds optional"
            )
        mode = value[0].lower()
        if mode not in ["fit", "fix"]:
            raise ValueError('tau_E: first element must be "fit" or "fix"')
        try:
            guess = float(value[1])
        except Exception:
            raise ValueError(f"Cannot convert {value[1]} to float")
        if guess <= 0:
            raise ValueError(f"{value[1]} is negative; tau_E must be positive")
        bounds = value[2] if len(value) == 3 else None
        if bounds is not None:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(
                    "tau_E bounds must be a list: [upper-bound, lower-bound]"
                )
            try:
                lower = float(bounds[0])
                upper = float(bounds[1])
            except Exception:
                raise ValueError(f"Cannot convert tau_E bounds {bounds} to floats")
            if upper <= lower:
                raise ValueError("tau_E upper bound must be greater than lower bound")
            if lower <= 0 or upper <= 0:
                raise ValueError("tau_E bounds must be positive")
            self._fit_corr_time_tau_E = [mode, guess, [lower, upper]]
        if mode == "fix" and bounds is not None:
            raise ValueError("Remove bounds if correlation time is fixed.")
        else:
            self._fit_corr_time_tau_E = [mode, guess]
        return None

    @property
    def relaxation_model(self) -> str:
        return self._relaxation_model

    @relaxation_model.setter
    def relaxation_model(self, value: str):
        if value.lower() not in ["sbm", "curie", "sbm curie", "curie sbm"]:
            raise ValueError(f"Unknown relaxation: model {value}")
        else:
            self._relaxation_model = value.lower()
        return None

    @property
    def hyperfine_paramagnetic_centre(self) -> list[float] | None:
        return self._hyperfine_paramagnetic_centre

    @hyperfine_paramagnetic_centre.setter
    def hyperfine_paramagnetic_centre(
        self, value: list[float] | tuple[float, float, float] | None
    ):
        if value is None or value == "":
            self._hyperfine_paramagnetic_centre = None
            return None
        if isinstance(value, str):
            value = yaml.safe_load(value)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                self._hyperfine_paramagnetic_centre = [float(val) for val in value]
            except Exception as exc:
                raise ValueError(
                    f"Cannot convert hyperfine:paramagnetic_centre={value} "
                    "to list of 3 floats"
                ) from exc
            return None
        raise ValueError("hyperfine:paramagnetic_centre must be a list of 3 floats")

    @classmethod
    def from_file(cls, file_name: str) -> "FitCorrTimeConfig":
        """Creates a `FitCorrTimeConfig` from a YAML input file.

        Args:
            file_name: Path to the YAML file to read.

        Returns:
            A populated `FitCorrTimeConfig` instance.
        """
        cls: FitCorrTimeConfig = super().from_file(file_name)
        if cls.relaxation_model and cls.hyperfine_paramagnetic_centre is None:
            raise ValueError(
                "If 'relaxation' is specified, 'hyperfine:paramagnetic_centre' "
                "must be set"
            )
        return cls


class PlotHFCConfig(FitSuscConfig):
    REQ_KEYWORDS = {
        "hyperfine": ["method", "file"],
        "nuclei": [
            "include",
        ],
        "project": ["name"],
    }

    KEYWORDS = {
        "hyperfine": [
            "method",
            "file",
            "average",
            "orbital_contribution",
        ],
        "nuclei": ["include", "include_groups"],
        "project": ["name"],
        "chem_labels": ["file"],
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
        if value not in ["dft", "pdip", "csv"]:
            raise ValueError(f"Unknown hyperfine:method {value}")
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
    def nuclei_include(self) -> list | str:
        return self._nuclei_include

    @nuclei_include.setter
    def nuclei_include(self, values: list | str):
        self._nuclei_include = values
        return
