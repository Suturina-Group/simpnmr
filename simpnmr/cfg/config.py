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
        # and subkeywords (only for top-level keys that are known)
        unsupported += [
            subkey
            for key in parsed
            if key in cls.KEYWORDS
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
                    # Allow nuclei: isotope (or legacy include) to be omitted
                    # if the other form or include_groups is provided.
                    if keyword == "nuclei" and subkeyword in ("isotope", "include"):
                        nuclei_block = (
                            parsed.get("nuclei", {}) if isinstance(parsed, dict) else {}
                        )
                        if isinstance(nuclei_block, dict):
                            # Accept either key as satisfying the requirement.
                            alt = "include" if subkeyword == "isotope" else "isotope"
                            if nuclei_block.get(alt) not in (None, [], ""):
                                continue
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
        "nuclei": ["isotope"],
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
            "shared",
            "correlations",
        ],
        "nuclei": ["isotope", "include", "include_groups", "exclude_groups"],
        "susc_fit": ["type", "variables", "input_units", "average_shifts", "covariance_params", "figures", "spectra_break", "shifts_format", "shifts_width_scale", "shifts_labels"],
        "project": ["name"],
        "chem_labels": ["file"],
        "diamagnetic": [
            "method",
            "file",
        ],
        "diamagnetic_ref": ["method", "file", "values"],
        "susc_vt": [
            "method",
            "variables",
            "tip_type",
            "ab_initio_file",
            "ab_initio_format",
            "zeta_eff",
            "zeta_eff_tol",
            "evans_g_iso",
            "evans_g_iso_err",
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
            "distance_power",
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
        self._diamagnetic_ref_values = None
        self._assignment_method = ""
        self._assignment_groups = []
        self._assignment_search = ""
        self._assignment_n_attempts = None
        self._assignment_max_iter = None
        self._assignment_rmse_threshold = None
        self._assignment_area_weight = 0.0
        self._assignment_width_weight = 0.0
        self._assignment_r1_weight = 0.0
        self._assignment_shared = False
        self._assignment_correlations: list[dict] = []
        self._nuclei_include = ""
        self._nuclei_include_groups = []
        self._nuclei_isotope_order: list[str] = []
        self._nuclei_exclude_groups = []
        self._nuclei_exclude: list[str] = []
        self._susc_fit_type = ""
        self._susc_fit_variables = ""
        self._susc_fit_input_units = "A3"
        self._susc_fit_average_shifts = []
        self._susc_fit_covariance_params: list[str] = []
        self._susc_fit_figures: dict[str, bool] = {}
        self._susc_fit_spectra_break: dict | None = None
        self._susc_fit_shifts_format = "standard"
        self._susc_fit_shifts_width_scale = 1.0
        self._susc_fit_shifts_labels = True
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
        self._susc_vt_zeta_eff: float | None = None
        self._susc_vt_zeta_eff_tol: float = 0.15
        self._susc_vt_evans_g_iso: float | None = None
        self._susc_vt_evans_g_iso_err: float = 0.0
        self._fit_relaxation_tau_e_range = None
        self._fit_relaxation_tau_r_range = None
        self._fit_relaxation_tau_e = None
        self._fit_relaxation_tau_r_fixed = None
        self._fit_relaxation_tau_r_method = None
        self._fit_relaxation_tau_r_solvent = None
        self._fit_relaxation_tau_r_eta = None
        self._fit_relaxation_tau_r_shell = None
        self._fit_relaxation_tau_r_sigma = None
        self._fit_relaxation_distance_power = 0.0

        for key in kwargs:
            setattr(self, key, kwargs[key])

        self._resolve_nuclei_include_groups()

        pass

    @property
    def hyperfine_paramagnetic_centre(self) -> list[float] | None:
        return self._hyperfine_paramagnetic_centre

    @hyperfine_paramagnetic_centre.setter
    def hyperfine_paramagnetic_centre(
        self, value: list[float] | tuple[float, float, float] | str | None
    ):
        if value is None or value == "":
            self._hyperfine_paramagnetic_centre = None
            return None
        if isinstance(value, str):
            value = yaml.safe_load(value)
        if isinstance(value, str):
            # Atom label (e.g. "Ni1") — resolved against molecule geometry at load time
            self._hyperfine_paramagnetic_centre = value
            return None
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                self._hyperfine_paramagnetic_centre = [float(val) for val in value]
            except Exception as exc:
                raise ValueError(
                    f"Cannot convert hyperfine:paramagnetic_centre={value} to "
                    "list of 3 floats"
                ) from exc
            return None
        raise ValueError(
            "hyperfine:paramagnetic_centre must be an atom label (e.g. Ni1) "
            "or a list of 3 floats [x, y, z]"
        )

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
        # Accept a single string, a flat list of strings, or a list of lists
        # (positional syntax pairing each sub-list with an isotope index).
        if isinstance(values, str):
            self._nuclei_include_groups = [values]
        else:
            self._nuclei_include_groups = list(values)
        return

    @property
    def nuclei_exclude_groups(self) -> list:
        return self._nuclei_exclude_groups

    @nuclei_exclude_groups.setter
    def nuclei_exclude_groups(self, values: list | str):
        # Same syntax as include_groups: flat list or list-of-lists.
        if isinstance(values, str):
            self._nuclei_exclude_groups = [values]
        else:
            self._nuclei_exclude_groups = list(values)
        return

    @property
    def nuclei_exclude(self) -> list[str]:
        return self._nuclei_exclude

    def _resolve_nuclei_include_groups(self):
        """Expands ``nuclei:include_groups`` into atom labels.

        Uses ``chem_labels_file`` to map ``chem_label`` values to
        ``atom_label`` values.

        **Flat syntax** (single list of strings):
            Each entry may be a bare chem_label (``"ring"``) or carry an
            isotope prefix (``"1H:ring"``).  Bare entries expand all matching
            atoms regardless of element; prefixed entries restrict to the
            stated element (``"1H"`` → ``"H"``).  The result is merged with
            the existing ``_nuclei_include`` content.

        **Positional syntax** (list of lists):
            Each sub-list is paired with the isotope at the same index in
            ``nuclei:isotope``.  Only atom labels whose element matches the
            paired isotope are selected.  The result *replaces* the broad
            element symbols that ``nuclei_isotope`` stored in
            ``_nuclei_include``, so only the explicitly listed groups are
            retained::

                nuclei:
                  isotope: [1H, 13C]
                  include_groups:
                    - [Me1, Me2]   # selects H atoms for 1H
                    - [Me1, Me3]   # selects C atoms for 13C

        This method is safe to call multiple times.

        Raises:
            FileNotFoundError: If ``chem_labels_file`` does not exist.
            ValueError: If no atoms are matched.
        """
        import re

        raw_groups = getattr(self, "_nuclei_include_groups", [])
        if raw_groups is None:
            raw_groups = []
        if isinstance(raw_groups, str):
            raw_groups = [raw_groups]

        raw_excl = getattr(self, "_nuclei_exclude_groups", [])
        if raw_excl is None:
            raw_excl = []
        if isinstance(raw_excl, str):
            raw_excl = [raw_excl]

        # Nothing to do if both lists are empty.
        if not raw_groups and not raw_excl:
            return

        # If chem_labels_file is not set yet, skip silently.
        chem_file = getattr(self, "_chem_labels_file", "")
        if not chem_file:
            return

        # ── helpers ───────────────────────────────────────────────────
        def _load_csv(path: str) -> list[tuple[str, str]]:
            pairs: list[tuple[str, str]] = []
            try:
                with open(path, newline="") as f:
                    reader = csv.DictReader(f, skipinitialspace=True)

                    def _get(row: dict, key: str):
                        for k, v in row.items():
                            if k is not None and k.strip() == key:
                                return v
                        return None

                    for row in reader:
                        cl = (_get(row, "chem_label") or "").strip()
                        al = (_get(row, "atom_label") or "").strip()
                        if cl and al:
                            pairs.append((cl, al))
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"chem_labels_file not found: {path}"
                )
            return pairs

        csv_pairs = _load_csv(chem_file)

        def _atom_element(alabel: str) -> str:
            m = re.match(r"[A-Za-z]+", alabel)
            return m.group(0) if m else ""

        def _expand(
            chem_labels: list[str],
            element_filter: str | None,
        ) -> list[str]:
            label_set = {str(c).strip() for c in chem_labels}
            result: list[str] = []
            for cl, al in csv_pairs:
                if cl not in label_set:
                    continue
                if element_filter and _atom_element(al) != element_filter:
                    continue
                result.append(al)
            return result

        def _expand_flat_list(
            entries: list,
            isotope_order: list[str],
        ) -> list[str]:
            """Expand a flat or positional group list to atom labels."""
            first = entries[0] if entries else None
            if isinstance(first, (list, tuple)):
                # Positional: pair sub-list[i] with isotope_order[i].
                atoms: list[str] = []
                for i, sub in enumerate(entries):
                    if not isinstance(sub, (list, tuple)):
                        sub = [sub]
                    el = isotope_order[i] if i < len(isotope_order) else None
                    atoms.extend(_expand(list(sub), el))
                return atoms
            # Flat: "1H:ring" prefix or bare "ring".
            atoms = []
            for entry in entries:
                entry = str(entry).strip()
                if ":" in entry:
                    iso, cl = entry.split(":", 1)
                    el = re.sub(r"^\d+", "", iso.strip()).strip() or None
                    atoms.extend(_expand([cl.strip()], el))
                else:
                    atoms.extend(_expand([entry], None))
            return atoms

        isotope_order = getattr(self, "_nuclei_isotope_order", [])

        # ── include_groups ────────────────────────────────────────────
        if raw_groups:
            first = raw_groups[0]
            positional = isinstance(first, (list, tuple))

            if positional:
                expanded = _expand_flat_list(raw_groups, isotope_order)
                if not expanded:
                    raise ValueError(
                        "No nuclei selected: positional "
                        "nuclei:include_groups did not match any "
                        "chem_label entries in chem_labels_file."
                    )
                seen: set[str] = set()
                self._nuclei_include = [
                    x for x in expanded
                    if not (x in seen or seen.add(x))
                ]
            else:
                # Flat syntax.
                # Unprefixed entries (e.g. "tBu1a") inherit the element
                # filter from isotope_order so that "isotope: 1H" +
                # "include_groups: [tBu1a]" selects only H atoms from
                # tBu1a rather than all atoms of any element.
                # Explicitly prefixed entries ("1H:tBu1a") override.
                expanded_flat: list[str] = []
                for entry in raw_groups:
                    entry = str(entry).strip()
                    if ":" in entry:
                        iso, cl = entry.split(":", 1)
                        el = (
                            re.sub(r"^\d+", "", iso.strip()).strip()
                            or None
                        )
                        expanded_flat.extend(_expand([cl.strip()], el))
                    elif isotope_order:
                        # Apply each isotope element as a filter.
                        for el in isotope_order:
                            expanded_flat.extend(_expand([entry], el))
                    else:
                        expanded_flat.extend(_expand([entry], None))

                if not expanded_flat:
                    raise ValueError(
                        "No nuclei selected: nuclei:include_groups did "
                        "not match any chem_label entries in "
                        f"chem_labels_file. Requested groups={raw_groups}."
                    )
                # Replace _nuclei_include: include_groups is the
                # authoritative selector, not a supplement to isotope:.
                seen2: set[str] = set()
                self._nuclei_include = [
                    x for x in expanded_flat
                    if not (x in seen2 or seen2.add(x))
                ]

        # ── exclude_groups ────────────────────────────────────────────
        if raw_excl:
            excluded = _expand_flat_list(raw_excl, isotope_order)
            excl_set: set[str] = set(excluded)
            self._nuclei_exclude = list(excl_set)

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
    def assignment_correlations(self) -> list[dict]:
        """List of HMBC/HSQC correlation constraints for permute assignment.

        Each entry is a dict with keys ``h`` (H experimental signal label),
        ``c`` (C experimental signal label), ``type`` (``"hsqc"`` or
        ``"hmbc"``), and optionally ``cutoff`` (Å, overrides default).
        """
        return self._assignment_correlations

    @assignment_correlations.setter
    def assignment_correlations(self, value) -> None:
        if value is None or value == "" or value == []:
            self._assignment_correlations = []
            return
        if not isinstance(value, list):
            raise ValueError(
                "assignment:correlations must be a list of {h, c, type} dicts"
            )
        parsed = []
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(
                    f"assignment:correlations[{i}] must be a mapping with keys "
                    "'h', 'c', and optionally 'type' and 'cutoff'"
                )
            if "h" not in item or "c" not in item:
                raise ValueError(
                    f"assignment:correlations[{i}] must have 'h' and 'c' keys"
                )
            corr = {
                "h": str(item["h"]),
                "c": str(item["c"]),
                "type": str(item.get("type", "hsqc")).lower(),
            }
            if corr["type"] not in ("hsqc", "hmbc"):
                raise ValueError(
                    f"assignment:correlations[{i}].type must be 'hsqc' or 'hmbc', "
                    f"got {corr['type']!r}"
                )
            if "cutoff" in item:
                corr["cutoff"] = float(item["cutoff"])
            parsed.append(corr)
        self._assignment_correlations = parsed

    @property
    def assignment_shared(self) -> bool:
        return self._assignment_shared

    @assignment_shared.setter
    def assignment_shared(self, value) -> None:
        if isinstance(value, bool):
            self._assignment_shared = value
        elif isinstance(value, str):
            self._assignment_shared = value.lower() in ("true", "yes", "1")
        else:
            self._assignment_shared = bool(value)

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
    def susc_fit_covariance_params(self) -> list[str]:
        """Two parameter names for the covariance contour plot."""
        return self._susc_fit_covariance_params

    @susc_fit_covariance_params.setter
    def susc_fit_covariance_params(self, value):
        if value is None or value == "":
            self._susc_fit_covariance_params = []
            return
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        self._susc_fit_covariance_params = list(value)

    # Valid figure-group keys
    _FIGURE_KEYS = frozenset([
        "fitted_shifts",
        "shift_components",
        "r6_fit",
        "tau_space",
        "bubble_plots",
        "spectra",
        "chi_t",
    ])

    @property
    def susc_fit_figures(self) -> dict[str, bool]:
        """Dict of figure-group key → enabled flag (missing key = enabled)."""
        return self._susc_fit_figures

    @susc_fit_figures.setter
    def susc_fit_figures(self, value):
        if value is None:
            self._susc_fit_figures = {}
            return
        if not isinstance(value, dict):
            raise ValueError("susc_fit:figures must be a mapping")
        self._susc_fit_figures = {k: bool(v) for k, v in value.items()}

    def susc_fit_figure(self, key: str) -> bool:
        """Return whether a figure group is enabled (default True)."""
        return self._susc_fit_figures.get(key, True)

    @property
    def susc_fit_shifts_format(self) -> str:
        """Figure-size variant for the fitted-shifts (``shifts_*_K``) plot."""
        return self._susc_fit_shifts_format

    @susc_fit_shifts_format.setter
    def susc_fit_shifts_format(self, value) -> None:
        if value is None or value == "":
            self._susc_fit_shifts_format = "standard"
            return
        variant = str(value).strip().lower()
        allowed = {"standard", "narrow", "vertical", "vertical_extended"}
        if variant not in allowed:
            raise ValueError(
                "susc_fit:shifts_format must be one of "
                + ", ".join(sorted(allowed))
                + f"; got '{value}'"
            )
        self._susc_fit_shifts_format = variant

    @property
    def susc_fit_shifts_width_scale(self) -> float:
        """Width multiplier for the fitted-shifts (``shifts_*_K``) figure."""
        return self._susc_fit_shifts_width_scale

    @susc_fit_shifts_width_scale.setter
    def susc_fit_shifts_width_scale(self, value) -> None:
        if value is None or value == "":
            self._susc_fit_shifts_width_scale = 1.0
            return
        try:
            scale = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "susc_fit:shifts_width_scale must be a number"
            ) from exc
        if scale <= 0.0:
            raise ValueError("susc_fit:shifts_width_scale must be positive")
        self._susc_fit_shifts_width_scale = scale

    @property
    def susc_fit_shifts_labels(self) -> bool:
        """Whether to draw per-point labels on the fitted-shifts figure."""
        return self._susc_fit_shifts_labels

    @susc_fit_shifts_labels.setter
    def susc_fit_shifts_labels(self, value) -> None:
        if value is None or value == "":
            self._susc_fit_shifts_labels = True
            return
        self._susc_fit_shifts_labels = bool(value)

    @property
    def susc_fit_spectra_break(self) -> dict | None:
        """Manual x-axis break spec for the pred/exp spectrum figure.

        ``None`` (default) keeps the automatic gap-based segmentation. When
        set, it is a normalised dict with keys:

        - ``after_labels`` (list[str]): break just below each named peak.
        - ``after_ppms`` (list[float]): break at each explicit ppm position.
        - ``segments`` (list[[lo, hi]] | None): explicit per-panel ppm limits,
          high→low ppm (left→right). When given, it overrides ``after_*`` and
          fully controls each panel's displayed range.
        - ``scales`` (list[float]): per-segment vertical scale, high→low ppm
          (left→right). Length must equal the number of panels.
        - ``equal_width`` (bool): equal panel widths instead of ppm-proportional.
        - ``width_ratios`` (list[float] | None): explicit relative panel widths
          (one per panel); overrides ``equal_width`` when set.
        - ``label_scale`` (float): multiplier on peak-label font size (default 1).
        """
        return self._susc_fit_spectra_break

    @susc_fit_spectra_break.setter
    def susc_fit_spectra_break(self, value) -> None:
        if value is None or value == "":
            self._susc_fit_spectra_break = None
            return
        if not isinstance(value, dict):
            raise ValueError(
                "susc_fit:spectra_break must be a mapping, e.g. "
                "{after_label: tBu4a, scales: [1, 4], equal_width: true}"
            )
        allowed = {
            "after_label", "after_ppm", "segments", "scales",
            "equal_width", "width_ratios", "label_scale",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "susc_fit:spectra_break contains unknown key(s): "
                + ", ".join(sorted(unknown))
            )

        def _as_list(v):
            if v is None:
                return []
            return list(v) if isinstance(v, (list, tuple)) else [v]

        equal_width = bool(value.get("equal_width", False))

        label_scale = float(value.get("label_scale", 1.0))
        if label_scale <= 0.0:
            raise ValueError(
                "susc_fit:spectra_break:label_scale must be positive"
            )

        def _parse_width_ratios(n_panels: int) -> list[float] | None:
            raw = value.get("width_ratios")
            if raw is None:
                return None
            ratios = [float(x) for x in _as_list(raw)]
            if len(ratios) != n_panels:
                raise ValueError(
                    "susc_fit:spectra_break:width_ratios must have "
                    f"{n_panels} entries (one per panel) but got {len(ratios)}"
                )
            if any(r <= 0.0 for r in ratios):
                raise ValueError(
                    "susc_fit:spectra_break:width_ratios must all be positive"
                )
            return ratios

        # Explicit per-panel limits take precedence over break points.
        if value.get("segments") is not None:
            raw_segs = value["segments"]
            if not isinstance(raw_segs, (list, tuple)) or not raw_segs:
                raise ValueError(
                    "susc_fit:spectra_break:segments must be a non-empty list "
                    "of [hi, lo] ppm pairs"
                )
            segs: list[list[float]] = []
            for pair in raw_segs:
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    raise ValueError(
                        "susc_fit:spectra_break:segments entries must be "
                        "[hi, lo] ppm pairs"
                    )
                a, b = float(pair[0]), float(pair[1])
                segs.append([min(a, b), max(a, b)])  # store as (lo, hi)
            n_seg = len(segs)
            scales = _as_list(value.get("scales")) or [1.0] * n_seg
            scales = [float(x) for x in scales]
            if len(scales) != n_seg:
                raise ValueError(
                    "susc_fit:spectra_break:scales must have "
                    f"{n_seg} entries (one per segment) but got {len(scales)}"
                )
            width_ratios = _parse_width_ratios(n_seg)
            # Order high→low ppm (left→right), keeping scales/widths aligned.
            order = sorted(range(n_seg), key=lambda i: segs[i][1], reverse=True)
            segs = [segs[i] for i in order]
            scales = [scales[i] for i in order]
            if width_ratios is not None:
                width_ratios = [width_ratios[i] for i in order]
            self._susc_fit_spectra_break = {
                "after_labels": [],
                "after_ppms": [],
                "segments": segs,
                "scales": scales,
                "equal_width": equal_width,
                "width_ratios": width_ratios,
                "label_scale": label_scale,
            }
            return

        if "after_label" not in value and "after_ppm" not in value:
            raise ValueError(
                "susc_fit:spectra_break requires 'after_label', 'after_ppm', "
                "or 'segments'"
            )

        after_labels = [str(x) for x in _as_list(value.get("after_label"))]
        after_ppms = [float(x) for x in _as_list(value.get("after_ppm"))]
        n_breaks = len(after_labels) + len(after_ppms)
        if n_breaks == 0:
            raise ValueError(
                "susc_fit:spectra_break must define at least one break"
            )

        scales = _as_list(value.get("scales")) or [1.0] * (n_breaks + 1)
        scales = [float(x) for x in scales]
        if len(scales) != n_breaks + 1:
            raise ValueError(
                "susc_fit:spectra_break:scales must have "
                f"{n_breaks + 1} entries (n_breaks + 1) but got {len(scales)}"
            )

        self._susc_fit_spectra_break = {
            "after_labels": after_labels,
            "after_ppms": after_ppms,
            "segments": None,
            "scales": scales,
            "equal_width": equal_width,
            "width_ratios": _parse_width_ratios(n_breaks + 1),
            "label_scale": label_scale,
        }

    @property
    def nuclei_include(self) -> list | str:
        return self._nuclei_include

    @nuclei_include.setter
    def nuclei_include(self, values: list | str):
        self._nuclei_include = values
        return

    @property
    def nuclei_isotope(self) -> list | str:
        return self._nuclei_include

    @nuclei_isotope.setter
    def nuclei_isotope(self, values: list | str):
        """Accept isotope strings and convert to element symbols.

        ``"1H"`` → ``"H"``, ``"13C"`` → ``"C"``, ``"H"`` → ``"H"``.
        Stored in ``_nuclei_include`` to reuse existing element-filtering.
        The ordered list is also stored in ``_nuclei_isotope_order`` so that
        positional ``include_groups`` lists can be paired with isotopes.
        """
        import re as _re
        if isinstance(values, str):
            values = [values]
        elements = [_re.sub(r"^\d+", "", str(v).strip()) for v in values]
        self._nuclei_isotope_order = elements
        current = self._nuclei_include
        if isinstance(current, list) and current:
            merged = current + elements
        elif isinstance(current, str) and current:
            merged = [current] + elements
        else:
            merged = elements
        seen: set[str] = set()
        self._nuclei_include = [
            x for x in merged if not (x in seen or seen.add(x))
        ]

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
        if value not in ["dft", "csv", "values"]:
            raise ValueError(f"Unknown diamagnetic_reference:method {value}")
        else:
            self._diamagnetic_ref_method = value
        return

    @property
    def diamagnetic_ref_file(self) -> str | dict[str, str]:
        return self._diamagnetic_ref_file

    @diamagnetic_ref_file.setter
    def diamagnetic_ref_file(self, value: str | dict):
        if isinstance(value, dict):
            # Per-isotope file mapping: {isotope: path}
            self._diamagnetic_ref_file = {
                str(iso): os.path.abspath(str(path))
                for iso, path in value.items()
            }
        elif isinstance(value, str):
            self._diamagnetic_ref_file = os.path.abspath(value)
        else:
            raise ValueError("Diamagnetic reference file must be a string or dict")
        return

    @property
    def diamagnetic_ref_values(self) -> dict[str, float] | None:
        return self._diamagnetic_ref_values

    @diamagnetic_ref_values.setter
    def diamagnetic_ref_values(self, value: dict | None):
        if value is None:
            self._diamagnetic_ref_values = None
            return
        if not isinstance(value, dict):
            raise ValueError(
                "diamagnetic_ref:values must be a mapping of {isotope: float}, "
                "e.g. {1H: 31.74, 13C: 188.07}"
            )
        self._diamagnetic_ref_values = {
            str(iso): float(v) for iso, v in value.items()
        }
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
    def susc_vt_zeta_eff(self) -> float | None:
        """Effective spin-orbit coupling constant ζ̄_eff in cm⁻¹, or None."""
        return self._susc_vt_zeta_eff

    @susc_vt_zeta_eff.setter
    def susc_vt_zeta_eff(self, value: float | None):
        if value is None:
            self._susc_vt_zeta_eff = None
            return
        try:
            self._susc_vt_zeta_eff = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("susc_vt:zeta_eff must be a number or None") from exc

    @property
    def susc_vt_zeta_eff_tol(self) -> float:
        """Fractional uncertainty on ζ̄_eff for the solution-line band (default 0.15)."""
        return self._susc_vt_zeta_eff_tol

    @susc_vt_zeta_eff_tol.setter
    def susc_vt_zeta_eff_tol(self, value: float | None):
        if value is None:
            self._susc_vt_zeta_eff_tol = 0.15
            return
        try:
            tol = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("susc_vt:zeta_eff_tol must be a number or None") from exc
        if not (0.0 < tol < 1.0):
            raise ValueError("susc_vt:zeta_eff_tol must be between 0 and 1")
        self._susc_vt_zeta_eff_tol = tol

    @property
    def susc_vt_evans_g_iso(self) -> float | None:
        """Isotropic g-value from an Evans-method measurement, or None."""
        return self._susc_vt_evans_g_iso

    @susc_vt_evans_g_iso.setter
    def susc_vt_evans_g_iso(self, value: float | None):
        if value is None:
            self._susc_vt_evans_g_iso = None
            return
        try:
            self._susc_vt_evans_g_iso = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "susc_vt:evans_g_iso must be a number or None"
            ) from exc

    @property
    def susc_vt_evans_g_iso_err(self) -> float:
        """1-σ uncertainty on the Evans g_iso (default 0)."""
        return self._susc_vt_evans_g_iso_err

    @susc_vt_evans_g_iso_err.setter
    def susc_vt_evans_g_iso_err(self, value: float | None):
        if value is None:
            self._susc_vt_evans_g_iso_err = 0.0
            return
        try:
            self._susc_vt_evans_g_iso_err = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "susc_vt:evans_g_iso_err must be a number or None"
            ) from exc

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

    @property
    def fit_relaxation_distance_power(self) -> float:
        """Exponent k for distance-based r^-6 fit weighting (w = r**k)."""
        return self._fit_relaxation_distance_power

    @fit_relaxation_distance_power.setter
    def fit_relaxation_distance_power(self, value):
        if value is None or value == "":
            self._fit_relaxation_distance_power = 0.0
            return
        self._fit_relaxation_distance_power = float(value)

    @classmethod
    def from_file(cls, file_name) -> "FitSuscConfig":
        """Creates a `FitSuscConfig` from a YAML input file.

        Args:
            file_name: Path to the YAML file to read.

        Returns:
            A populated `FitSuscConfig` instance.
        """

        config = super().from_file(file_name)

        # If an ab initio file is provided without a format, warn but continue —
        # comparison plots and ζ extraction require both; the pipeline skips
        # the ab initio block when format is absent.
        if getattr(config, "susc_vt_ab_initio_file", ""):
            if not getattr(config, "susc_vt_ab_initio_format", ""):
                logger.warning(
                    "susc_vt:ab_initio_file is set but susc_vt:ab_initio_format "
                    "is missing — ab initio comparison and ζ extraction will be "
                    "skipped. Add e.g. 'ab_initio_format: orca_nev'."
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

        # ab_initio_file is now useful without TIP (comparison plots, ζ extraction)
        # so this is no longer an error.

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
            "isotope",
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
        "nuclei": ["isotope", "include", "include_groups", "exclude_groups"],
        "project": ["name"],
        "chem_labels": ["file"],
        "diamagnetic": [
            "method",
            "file",
        ],
        "diamagnetic_ref": ["method", "file"],
        "susceptibility": [
            "file", "format", "temperatures", "method", "sh",
            "reduced_chi", "bleaney",
        ],
        "relaxation": [
            "model",
            "temperature",
            "magnetic_field_tesla",
            "T1e",
            "T2e",
            "tR",
            "tau_r_method",
            "tau_r_solvent",
            "tau_r_eta",
            "tau_r_shell",
            "tau_r_sigma",
            "min_linewidth_hz",
        ],
    }

    def __init__(self, **kwargs):
        self._susceptibility_file = None
        self._susceptibility_format = None
        self._susceptibility_temperatures = []
        self._susceptibility_method = None
        self._susceptibility_sh = {}
        self._susceptibility_reduced_chi = {}
        self._susceptibility_bleaney = {}
        self._relaxation_model = ""
        self._hyperfine_paramagnetic_centre = None
        self._relaxation_temperature = None
        self._relaxation_magnetic_field_tesla = None
        self._relaxation_T1e = None
        self._relaxation_T2e = None
        self._relaxation_tR = None
        self._relaxation_tau_r_method = None
        self._relaxation_tau_r_solvent = None
        self._relaxation_tau_r_eta = None
        self._relaxation_tau_r_shell = None
        self._relaxation_tau_r_sigma = None
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
    def susceptibility_method(self) -> str | None:
        return self._susceptibility_method

    @susceptibility_method.setter
    def susceptibility_method(self, value: str | None):
        if value is None or value == "":
            self._susceptibility_method = None
            return None
        method = value.strip().lower()
        allowed = {"spin_only", "sh", "reduced_chi", "bleaney"}
        if method not in allowed:
            raise ValueError(
                f"Unknown susceptibility:method '{value}'. "
                f"Allowed: {', '.join(sorted(allowed))}."
            )
        self._susceptibility_method = method
        return None

    @property
    def susceptibility_bleaney(self) -> dict:
        return self._susceptibility_bleaney

    @susceptibility_bleaney.setter
    def susceptibility_bleaney(self, value):
        if value is None:
            self._susceptibility_bleaney = {}
            return None
        if not isinstance(value, dict):
            raise ValueError("susceptibility:bleaney must be a mapping")
        required = {"B20", "B22", "alpha", "beta", "gamma"}
        missing = required - set(value.keys())
        if missing:
            raise ValueError(
                f"susceptibility:bleaney is missing required keys: "
                f"{', '.join(sorted(missing))}"
            )
        self._susceptibility_bleaney = value
        return None

    @property
    def susceptibility_sh(self) -> dict:
        return self._susceptibility_sh

    @susceptibility_sh.setter
    def susceptibility_sh(self, value):
        if value is None:
            self._susceptibility_sh = {}
            return None
        if not isinstance(value, dict):
            raise ValueError("susceptibility:sh must be a mapping")
        required = {"gx", "gy", "gz", "D", "E_over_D", "alpha", "beta", "gamma"}
        missing = required - set(value.keys())
        if missing:
            raise ValueError(
                f"susceptibility:sh is missing required keys: "
                f"{', '.join(sorted(missing))}"
            )
        self._susceptibility_sh = value
        return None

    @property
    def susceptibility_reduced_chi(self) -> dict:
        return self._susceptibility_reduced_chi

    @susceptibility_reduced_chi.setter
    def susceptibility_reduced_chi(self, value):
        if value is None:
            self._susceptibility_reduced_chi = {}
            return None
        if not isinstance(value, dict):
            raise ValueError("susceptibility:reduced_chi must be a mapping")
        required = {"chi_iso_T", "chi_ax_T", "rh_over_ax",
                    "alpha", "beta", "gamma"}
        missing = required - set(value.keys())
        if missing:
            raise ValueError(
                f"susceptibility:reduced_chi is missing required keys: "
                f"{', '.join(sorted(missing))}"
            )
        self._susceptibility_reduced_chi = value
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
        self, value: list[float] | tuple[float, float, float] | str | None
    ):
        if value is None or value == "":
            self._hyperfine_paramagnetic_centre = None
            return None
        if isinstance(value, str):
            value = yaml.safe_load(value)
        if isinstance(value, str):
            self._hyperfine_paramagnetic_centre = value
            return None
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                self._hyperfine_paramagnetic_centre = [float(val) for val in value]
            except Exception as exc:
                raise ValueError(
                    f"Cannot convert hyperfine:paramagnetic_centre={value} to "
                    "list of 3 floats"
                ) from exc
            return None
        raise ValueError(
            "hyperfine:paramagnetic_centre must be an atom label (e.g. Ni1) "
            "or a list of 3 floats [x, y, z]"
        )

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
    def relaxation_tau_r_method(self) -> str | None:
        """Hydrodynamic model for estimating τ_R ('ellipsoid' or 'beadshell').

        When set, τ_R is computed from the molecular geometry and solvent
        viscosity instead of being read from ``tR``.
        """
        return self._relaxation_tau_r_method

    @relaxation_tau_r_method.setter
    def relaxation_tau_r_method(self, value):
        if value is None or value == "":
            self._relaxation_tau_r_method = None
            return
        if value not in ("ellipsoid", "beadshell"):
            raise ValueError(
                "relaxation:tau_r_method must be 'ellipsoid' or 'beadshell'"
            )
        self._relaxation_tau_r_method = value

    @property
    def relaxation_tau_r_solvent(self) -> str | None:
        """Solvent name for viscosity lookup when estimating τ_R."""
        return self._relaxation_tau_r_solvent

    @relaxation_tau_r_solvent.setter
    def relaxation_tau_r_solvent(self, value):
        if value is None or value == "":
            self._relaxation_tau_r_solvent = None
            return
        self._relaxation_tau_r_solvent = str(value)

    @property
    def relaxation_tau_r_eta(self) -> float | None:
        """Explicit solvent viscosity (Pa·s), overrides ``tau_r_solvent``."""
        return self._relaxation_tau_r_eta

    @relaxation_tau_r_eta.setter
    def relaxation_tau_r_eta(self, value):
        if value is None or value == "":
            self._relaxation_tau_r_eta = None
            return
        v = float(value)
        if v <= 0:
            raise ValueError("relaxation:tau_r_eta must be positive")
        self._relaxation_tau_r_eta = v

    @property
    def relaxation_tau_r_shell(self) -> float | None:
        """Solvent shell thickness (Å) added to vdW radii when estimating τ_R."""
        return self._relaxation_tau_r_shell

    @relaxation_tau_r_shell.setter
    def relaxation_tau_r_shell(self, value):
        if value is None or value == "":
            self._relaxation_tau_r_shell = None
            return
        v = float(value)
        if v < 0:
            raise ValueError("relaxation:tau_r_shell must be non-negative")
        self._relaxation_tau_r_shell = v

    @property
    def relaxation_tau_r_sigma(self) -> float | None:
        """Minibead radius (Å) for the bead-shell τ_R model."""
        return self._relaxation_tau_r_sigma

    @relaxation_tau_r_sigma.setter
    def relaxation_tau_r_sigma(self, value):
        if value is None or value == "":
            self._relaxation_tau_r_sigma = None
            return
        v = float(value)
        if v <= 0:
            raise ValueError("relaxation:tau_r_sigma must be positive")
        self._relaxation_tau_r_sigma = v

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
            "isotope",
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
        "nuclei": ["isotope", "include", "include_groups", "exclude_groups"],
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
        self, value: list[float] | tuple[float, float, float] | str | None
    ):
        if value is None or value == "":
            self._hyperfine_paramagnetic_centre = None
            return None
        if isinstance(value, str):
            value = yaml.safe_load(value)
        if isinstance(value, str):
            self._hyperfine_paramagnetic_centre = value
            return None
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                self._hyperfine_paramagnetic_centre = [float(val) for val in value]
            except Exception as exc:
                raise ValueError(
                    f"Cannot convert hyperfine:paramagnetic_centre={value} "
                    "to list of 3 floats"
                ) from exc
            return None
        raise ValueError(
            "hyperfine:paramagnetic_centre must be an atom label (e.g. Ni1) "
            "or a list of 3 floats [x, y, z]"
        )

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
            "isotope",
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
        "nuclei": ["isotope", "include", "include_groups", "exclude_groups"],
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
