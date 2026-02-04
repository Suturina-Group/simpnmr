# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define susceptibility fitting models and utilities.

Provides SusceptibilityModel base classes, concrete model parameterizations,
and helpers for least-squares fitting and uncertainty estimation.
"""

import copy
import logging
from abc import ABC, abstractmethod

import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray
from scipy.optimize import least_squares, lsq_linear
from scipy.optimize._optimize import OptimizeResult

from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule, Nucleus
from simpnmr.core.domain.tensor import Susceptibility

logger = logging.getLogger(__name__)


class SusceptibilityModel(ABC):
    """Base class for susceptibility fitting models.

    Concrete subclasses define a parameterization of the magnetic susceptibility
    tensor and a corresponding chemical-shift model.
    """

    def __init__(
        self, fit_vars: dict[str, float | str], fix_vars: dict[str, float | str]
    ):
        """Initializes a susceptibility model.

        Args:
            fit_vars: Parameters to be fitted. Keys must be in `VARNAMES`.
            fix_vars: Parameters to be held fixed. Keys must be in `VARNAMES`.

        Raises:
            ValueError: If required model variables are missing from `fit_vars` and
                `fix_vars`.
        """

        self.fit_vars = fit_vars
        self.fix_vars = fix_vars

        # Check all VARNAMES are provided in fit+fix
        input_names = [name for name in {**self.fit_vars, **self.fix_vars}.keys()]

        if any([req_name not in input_names for req_name in self.VARNAMES]):
            raise ValueError(f"Missing fit/fix parameters in {self.NAME} Model")

        # Final model parameter values
        self._final_var_values = {var: None for var in self.VARNAMES}
        # Standard deviation of each parameter
        self._fit_stdev = {var: None for var in self.fit_vars.keys()}

        # Fit status and temperature
        self._fit_status = False
        self._temperature = None

        # r2 and adjusted r2
        self._r2 = None
        self._adj_r2 = None

        # Residual
        self._mae = None
        # RMSE
        self._rmse = None

        return

    @property
    def fit_status(self) -> bool:
        """Whether the last fit was successful."""
        return self._fit_status

    @fit_status.setter
    def fit_status(self, value: bool):
        if isinstance(value, bool):
            self._fit_status = value
        else:
            raise TypeError
        return

    @property
    def temperature(self) -> float:
        """Temperature of the fit (K)."""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if isinstance(value, (np.floating, float)):
            self._temperature = value
        else:
            raise TypeError
        return

    @property
    def final_var_values(self) -> float:
        """Final values of all model parameters (fitted + fixed)."""
        return self._final_var_values

    @final_var_values.setter
    def final_var_values(self, value: dict):
        if isinstance(value, dict):
            self._final_var_values = value
        else:
            raise TypeError
        return

    @property
    def fit_stdev(self) -> float:
        """Standard deviation of fitted parameters from the fitting routine."""
        return self._fit_stdev

    @fit_stdev.setter
    def fit_stdev(self, value: dict):
        if isinstance(value, dict):
            self._fit_stdev = value
        else:
            raise TypeError
        return

    @property
    def fix_vars(self) -> dict[str, float]:
        """Fixed model parameters.

        Returns:
            Mapping from parameter name (in `VARNAMES`) to fixed value.
        """
        return self._fix_vars

    @fix_vars.setter
    def fix_vars(self, value: dict):
        if isinstance(value, dict):
            unknown = [key for key in value.keys() if key not in self.VARNAMES]
            if any(unknown):
                raise KeyError(f"Unknown variable names {unknown} provided to fix")
            self._fix_vars = value
        else:
            raise TypeError("fix must be dictionary")
        return

    @property
    def fit_vars(self) -> dict[str, float]:
        """Fitted model parameters.

        Returns:
            Mapping from parameter name (in `VARNAMES`) to initial guess.
        """
        return self._fit_vars

    @fit_vars.setter
    def fit_vars(self, value: dict):
        if isinstance(value, dict):
            unknown = [key for key in value.keys() if key not in self.VARNAMES]
            if any(unknown):
                raise KeyError(f"Unknown variable names {unknown} provided to fix")
            self._fit_vars = value
        else:
            raise TypeError("Fit must be dictionary")

        # Reset final model parameter values
        self._final_var_values = {var: None for var in self.VARNAMES}
        # Reset standard deviation of each parameter
        self._fit_stdev = {var: None for var in self.fit_vars.keys()}
        return

    @property
    def r2(self) -> float:
        """Coefficient of determination (R²) of the fit."""
        return self._r2

    @r2.setter
    def r2(self, value):
        if isinstance(value, (np.floating, float)):
            self._r2 = value
        else:
            raise TypeError
        return

    @property
    def adj_r2(self) -> float:
        """Adjusted coefficient of determination (adjusted R²)."""
        return self._adj_r2

    @adj_r2.setter
    def adj_r2(self, value):
        if isinstance(value, (np.floating, float)):
            self._adj_r2 = value
        else:
            raise TypeError
        return

    @property
    def mae(self) -> float:
        """Mean absolute error (MAE) of the fit."""
        return self._mae

    @mae.setter
    def mae(self, value):
        if isinstance(value, (np.floating, float)):
            self._mae = value
        else:
            raise TypeError
        return

    @property
    def rmse(self) -> float:
        """Root mean square error (RMSE) of the fit."""
        return self._rmse

    @rmse.setter
    def rmse(self, value):
        if isinstance(value, (np.floating, float)) or np.isnan(value):
            self._rmse = value
        else:
            raise TypeError
        return

    @property
    @abstractmethod
    def NAME() -> str:
        """Human-readable name of the model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def VARNAMES() -> list[str]:
        """Names of parameters that can be fitted or fixed."""
        raise NotImplementedError

    @property
    @abstractmethod
    def VARNAMES_MM() -> dict[str, str]:
        """Math-mode (LaTeX) labels for model parameters.

        Returns:
            Mapping from parameter names in `VARNAMES` to LaTeX strings.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def UNITS_MM() -> dict[str, str]:
        """Math-mode (LaTeX) units for model parameters.

        Returns:
            Mapping from parameter names in `VARNAMES` to LaTeX unit strings.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def BOUNDS() -> dict[str, list[float, float]]:
        """Bounds for each model parameter.

        Returns:
            Mapping from parameter name to ``[lower, upper]`` bounds.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def model(parameters: dict[str, float], nuclei: list[Nucleus]) -> dict[str, float]:
        """Evaluates the model prediction for paramagnetic shifts.

        Args:
            parameters: Model parameters. Keys are `VARNAMES`.
            nuclei: Nuclei for which shifts will be computed.

        Returns:
            Mapping from nucleus atom labels to predicted paramagnetic shifts.
        """
        raise NotImplementedError

    def tosusceptibility(self) -> Susceptibility:
        """Converts the fitted model into a `Susceptibility` instance.

        Returns:
            A `Susceptibility` object at `self.temperature`.
        """
        tensor = self.totensor(self.final_var_values)
        susc = Susceptibility(tensor, self.temperature)
        return susc

    @staticmethod
    @abstractmethod
    def totensor(params: dict[str, float]) -> NDArray:
        """Converts model parameters to a susceptibility tensor.

        Args:
            params: Model parameters. Keys are `VARNAMES`.

        Returns:
            Susceptibility tensor as a ``(3, 3)`` NumPy array.
        """
        raise NotImplementedError

    def _post_fit(self) -> None:
        """Hook for model-specific post-processing after a successful fit.

        Called at the end of `fit_to` after `final_var_values` and `fit_stdev` are set.
        Subclasses may override to compute derived quantities.

        Returns:
            None.
        """
        return

    def residuals(
        self,
        parameters: dict[str, float],
        nuclei: list[Nucleus],
        al_to_para_shift: dict[str, float],
        average_labels: list[list[str]] = [],
    ) -> list[float]:
        """Computes residuals between experimental and predicted shifts.

        Args:
            parameters: Trial parameters used to compute model shifts.
            nuclei: Nuclei for which shifts are computed.
            al_to_para_shift: Mapping from atom label to experimental
            paramagnetic shift.
            average_labels: Optional groups of atom labels whose predicted shifts are
                averaged prior to residual computation.

        Returns:
            A list of residuals (experimental - predicted), optionally reweighted for
            averaged groups.
        """

        trial_shifts = self.model(parameters, nuclei)

        # Initialize weights for all atom labels to 1.0
        weights = {lab: 1.0 for lab in trial_shifts.keys()}
        if average_labels:
            # For each group, compute the average shift and assign a weight factor
            # such that the overall contribution of the group is independent of its size
            for group in average_labels:
                group_average = np.mean([trial_shifts[lab] for lab in group])
                group_size = len(group)
                for lab in group:
                    trial_shifts[lab] = group_average
                    # residuals will be divided by this
                    weights[lab] = np.sqrt(group_size)

        # Compute residuals using uniform weighting for single signals
        # and scaled weights for groups
        residuals = [
            (exp_shift - trial_shifts[atom_label]) / weights.get(atom_label, 1.0)
            for atom_label, exp_shift in al_to_para_shift.items()
        ]

        return residuals

    def residual_from_float_list(
        self,
        new_vals: list[float],
        fit_vars: dict[str, float],
        fix_vars: dict[str, float],
        nuclei: list[Nucleus],
        al_to_para_shift: dict[str, float],
        average_labels: list[list[str]] = [],
    ) -> list[float]:
        """Adapter for optimizers that pass parameters as a flat float list.

        Converts `new_vals` into a parameter dictionary (using `fit_vars` key order),
        merges it with `fix_vars`, then calls `residuals`.

        Args:
            new_vals: New values provided by the optimizer (order matches `fit_vars`).
            fit_vars: Fit-variable template mapping names to initial guesses.
            fix_vars: Fixed parameters that remain constant during fitting.
            nuclei: Nuclei for which shifts are computed.
            al_to_para_shift: Mapping from atom label to experimental
            paramagnetic shift.
            average_labels: Optional groups of atom labels whose predicted shifts are
                averaged prior to residual computation.

        Returns:
            A list of residuals.
        """

        # Swap fit values for new values from fit routine
        new_fit_vars = {name: guess for guess, name in zip(new_vals, fit_vars.keys())}

        # And make combined dict of fit and fixed
        # variable names (keys) and values
        all_vars = {**fix_vars, **new_fit_vars}

        residuals = self.residuals(
            all_vars, nuclei, al_to_para_shift, average_labels=average_labels
        )

        return residuals

    def fit_to(
        self,
        molecule: Molecule,
        experiment: Experiment,
        verbose: bool = True,
        average_labels: list[list[str]] = [],
    ) -> None:
        """Fits the model to experimental susceptibility data.

        Args:
            molecule: Molecule providing nuclei and geometric information.
            experiment: Experimental data object.
            verbose: If ``False``, suppresses terminal output.
            average_labels: Optional groups of atom labels whose predicted shifts are
                averaged prior to residual computation.

        Returns:
            None.
        """

        # Starting values
        guess = [val for val in self.fit_vars.values()]

        # Get bounds for variables to be fitted
        bounds = np.array([self.BOUNDS[name] for name in self.fit_vars.keys()]).T

        # Chemical label to paramagnetic shift
        al_to_para_shift = {
            nuc.label: experiment[nuc.chem_label].shift - nuc.shift.dia
            for nuc in molecule.nuclei
        }

        curr_fit = least_squares(
            fun=self.residual_from_float_list,
            args=(
                self.fit_vars,
                self.fix_vars,
                molecule.nuclei,
                al_to_para_shift,
                average_labels,
            ),
            x0=guess,
            bounds=bounds,
            jac="3-point",
        )

        self.temperature = experiment.temperature

        # Fitted parameters
        curr_fit_dict = {
            name: value for name, value in zip(self.fit_vars.keys(), curr_fit.x)
        }

        if curr_fit.status == 0:
            if verbose:
                logger.warning(
                    "Fit at %s K failed - Too many iterations", self.temperature
                )
            self.final_var_values = copy.deepcopy(curr_fit_dict)
            self.fit_stdev = {label: np.nan for label in self.fit_vars.keys()}
            self.fit_status = False
            self.mae = np.NaN
            self.rmse = np.NaN
            self.r2 = np.NaN
            self.adj_r2 = np.NaN
        else:
            # Calculate standard deviation error on the parameters
            stdev, _ = svd_stdev(curr_fit)

            # Standard deviation error on the parameters
            self.fit_stdev = {
                label: val for label, val in zip(self.fit_vars.keys(), stdev)
            }
            self.fit_status = True

            # Set fitted values
            self.final_var_values = copy.deepcopy(curr_fit_dict)

            # and fixed values
            for key, val in self.fix_vars.items():
                self.final_var_values[key] = val

            # Model-specific post-processing (e.g., derived parameter uncertainties)
            self._post_fit()

            # R2
            self.mae = np.sum(np.abs(curr_fit.fun)) / len(curr_fit.fun)
            ss_res = np.sum(curr_fit.fun**2)
            self.rmse = np.sqrt(ss_res / len(curr_fit.fun))
            ecs = [al_to_para_shift[nuc.label] for nuc in molecule.nuclei]
            ss_tot = np.sum((ecs - np.mean(ecs)) ** 2)
            self.r2 = 1 - (ss_res / ss_tot)
            self.adj_r2 = 1 - (1 - self.r2) * (len(ecs) - 1) / (
                len(ecs) - len(self.fit_vars) - 1
            )

        return


class LinearSusceptibilityModel(SusceptibilityModel):
    def fit_to(
        self,
        molecule: Molecule,
        experiment: Experiment,
        verbose: bool = True,
        average_labels: list[list[str]] = [],
    ) -> None:
        """Fits the linear model to experimental susceptibility data.

        Uses a linear least-squares formulation ``A x = b``.

        Args:
            molecule: Molecule providing nuclei and geometric information.
            experiment: Experimental data object.
            verbose: If ``False``, suppresses terminal output.
            average_labels: Optional groups of atom labels whose shifts are averaged
                prior to residual computation.

        Returns:
            None.
        """

        # Get bounds for variables to be fitted
        bounds = np.array([self.BOUNDS[name] for name in self.fit_vars.keys()]).T

        curr_fit = lsq_linear(
            A=self.design_matrix(molecule.nuclei, self.fix_vars),
            b=self.target_vector(molecule.nuclei, experiment, self.fix_vars),
            bounds=bounds,
        )

        self.temperature = experiment.temperature

        fit_var_names = [name for name in self.VARNAMES if name in self.fit_vars.keys()]

        # Fitted parameters
        curr_fit_dict = {name: value for name, value in zip(fit_var_names, curr_fit.x)}

        if curr_fit.status == 0:
            if verbose:
                logger.warning(
                    "Fit at %s K failed - Too many iterations", self.temperature
                )
            self.final_var_values = copy.deepcopy(curr_fit_dict)
            self.fit_stdev = {label: np.nan for label in self.fit_vars.keys()}
            self.fit_status = False
            self.rmse = np.NaN
            self.r2 = np.NaN
            self.adj_r2 = np.NaN
        else:
            # Calculate Jacobian, here equal to the design matrix
            curr_fit.jac = self.design_matrix(molecule.nuclei, self.fix_vars)

            # Calculate standard deviation error on the parameters
            stdev, _ = svd_stdev(curr_fit)

            # Standard deviation error on the parameters
            self.fit_stdev = {
                label: val for label, val in zip(self.fit_vars.keys(), stdev)
            }
            self.fit_status = True

            # Set fitted values
            self.final_var_values = copy.deepcopy(curr_fit_dict)
            # and fixed values
            for key, val in self.fix_vars.items():
                self.final_var_values[key] = val

            # R2
            ss_res = np.sum(curr_fit.fun**2)
            self.rmse = np.sqrt(ss_res / len(curr_fit.fun))
            ecs = [experiment[nuc.chem_label] for nuc in molecule.nuclei]
            ss_tot = np.sum((ecs - np.mean(ecs)) ** 2)
            self.r2 = 1 - (ss_res / ss_tot)
            self.adj_r2 = 1 - (1 - self.r2) * (len(ecs) - 1) / (
                len(ecs) - len(self.fit_vars) - 1
            )

        return

    @staticmethod
    @abstractmethod
    def design_matrix(nuclei: list[Nucleus], fix_vars: dict[str, float]):
        """Builds the design matrix for the linear model.

        Args:
            nuclei: Nuclei for which the model is evaluated.
            fix_vars: Fixed model parameters.

        Returns:
            The design matrix ``A`` in the linear system ``A x = b``.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def target_vector(
        nuclei: list[Nucleus],
        experiment: Experiment,
        fix_vars: dict[str, float],
    ):
        """Builds the target vector for the linear model.

        Args:
            nuclei: Nuclei for which the model is evaluated.
            experiment: Experimental data object.
            fix_vars: Fixed model parameters.

        Returns:
            The target vector ``b`` in the linear system ``A x = b``.
        """
        raise NotImplementedError


class SplitFitter(SusceptibilityModel):
    NAME = "Split Isotropic and Anisotropic Components of Susceptibility"

    VARNAMES = ["iso", "dxx", "dyy", "dxy", "dxz", "dyz"]

    VARNAMES_MM = {
        "iso": r"$\chi_\mathregular{iso}$",
        "dxx": r"$\Delta\chi_{xx}$",
        "dyy": r"$\Delta\chi_{yy}$",
        "dxy": r"$\Delta\chi_{xy}$",
        "dxz": r"$\Delta\chi_{xz}$",
        "dyz": r"$\Delta\chi_{yz}$",
    }

    UNITS_MM = {
        "iso": r"Å$^3$",
        "dxx": r"Å$^3$",
        "dyy": r"Å$^3$",
        "dxy": r"Å$^3$",
        "dxz": r"Å$^3$",
        "dyz": r"Å$^3$",
    }

    BOUNDS = {
        "iso": [0.0, np.inf],
        "dxx": [-np.inf, np.inf],
        "dyy": [-np.inf, np.inf],
        "dxy": [-np.inf, np.inf],
        "dxz": [-np.inf, np.inf],
        "dyz": [-np.inf, np.inf],
    }

    @staticmethod
    def model(parameters: dict[str, float], nuclei: list[Nucleus]) -> dict[str, float]:
        """Computes predicted paramagnetic shifts for the split-tensor model.

        The model uses an isotropic term and a traceless anisotropic tensor written
        in Cartesian components.

        Args:
            parameters: Model parameters. Keys are `VARNAMES`.
            nuclei: Nuclei for which shifts will be computed.

        Returns:
            Mapping from nucleus labels to predicted paramagnetic shifts.
        """

        delta_params = copy.deepcopy(parameters)
        tnsr = SplitFitter.totensor(delta_params)

        shifts = {
            nuc.label: 1.0 / 3.0 * np.trace(tnsr @ nuc.A.tensor) for nuc in nuclei
        }

        return shifts

    @staticmethod
    def totensor(params: dict[str, float]) -> NDArray:
        """Converts split-model parameters to a susceptibility tensor.

        Args:
            params: Model parameters. Keys are `VARNAMES`.

        Returns:
            Susceptibility tensor as a ``(3, 3)`` NumPy array.
        """

        tensor = np.array(
            [
                [params["dxx"], params["dxy"], params["dxz"]],
                [params["dxy"], params["dyy"], params["dyz"]],
                [
                    params["dxz"],
                    params["dyz"],
                    -params["dxx"] - params["dyy"],
                ],
            ]
        )
        tensor += np.eye(3) * params["iso"]

        return tensor


class IsoAxRhoFitter(SusceptibilityModel):
    NAME = "Isotropic, Axial, and Rhombic over Axial Components of Susceptibility"

    VARNAMES = [
        "iso",
        "ax",
        "rho_over_ax",
    ]

    VARNAMES_MM = {
        "iso": r"$\chi_\mathregular{iso}$",
        "ax": r"$\chi_\mathregular{ax}$",
        "rho_over_ax": r"$\chi_\mathregular{rho} / \chi_\mathregular{ax}$",
    }

    UNITS_MM = {
        "iso": r"Å$^3$",
        "ax": r"Å$^3$",
        "rho_over_ax": "",  # need to solve the problem with units
    }

    BOUNDS = {
        "iso": [0.0, np.inf],
        "ax": [-np.inf, np.inf],
        "rho_over_ax": [0.0, 1 / 3],
    }

    @staticmethod
    def model(parameters: dict[str, float], nuclei: list[Nucleus]) -> dict[str, float]:
        """Computes predicted paramagnetic shifts for the iso/ax/rho model.

        The anisotropic part is parameterized by axiality and a rhombicity ratio
        ``rho_over_ax``.

        Args:
            parameters: Model parameters. Keys are `VARNAMES`.
            nuclei: Nuclei for which shifts will be computed.

        Returns:
            Mapping from nucleus labels to predicted paramagnetic shifts.
        """

        delta_params = copy.deepcopy(parameters)
        tnsr = IsoAxRhoFitter.totensor(delta_params)

        shifts = {
            nuc.label: 1.0 / 3.0 * np.trace(tnsr @ nuc.A.tensor) for nuc in nuclei
        }

        return shifts

    @staticmethod
    def totensor(params: dict[str, float]) -> NDArray:
        """Converts iso/ax/rho parameters to a susceptibility tensor.

        Args:
            params: Model parameters. Keys are `VARNAMES`.

        Returns:
            Susceptibility tensor as a ``(3, 3)`` NumPy array.
        """

        tensor = np.array(
            [
                [-params["ax"] / 3 + params["rho_over_ax"] * params["ax"], 0.0, 0.0],
                [0.0, -params["ax"] / 3 - params["rho_over_ax"] * params["ax"], 0.0],
                [0.0, 0.0, 2 / 3 * params["ax"]],
            ]
        )
        tensor += np.eye(3) * params["iso"]

        return tensor

    def _post_fit(self) -> None:
        """Adds derived uncertainty for chi_rho.

        The fit uses `rho_over_ax`, but reporting prefers `rho = ax * rho_over_ax`.

        Notes:
            - If `rho_over_ax` is fixed, treat its uncertainty as zero and propagate
              only the `ax` uncertainty.
            - If `ax` is fixed (no `ax` stdev available), `rho` uncertainty cannot be
              propagated reliably and is omitted.
        """
        ax = self.final_var_values.get("ax")
        rho_over_ax = self.final_var_values.get("rho_over_ax")
        ax_st_dev = self.fit_stdev.get("ax")
        rho_over_ax_st_dev = self.fit_stdev.get("rho_over_ax")

        # Require values for rho computation
        if ax is None or rho_over_ax is None:
            self.fit_stdev.pop("rho", None)
            return

        # If ax is fixed, we cannot propagate an uncertainty for rho.
        if ax_st_dev is None:
            self.fit_stdev.pop("rho", None)
            return

        # If rho_over_ax is fixed, assume sigma_rho_over_ax = 0.
        if rho_over_ax_st_dev is None:
            if "rho_over_ax" in self.fix_vars:
                self.fit_stdev["rho"] = float(np.abs(rho_over_ax) * ax_st_dev)
                return
            self.fit_stdev.pop("rho", None)
            return

        # General case: first-order propagation under independence.
        self.fit_stdev["rho"] = float(
            np.hypot(rho_over_ax * ax_st_dev, ax * rho_over_ax_st_dev)
        )
        return


class EigenFitter(SusceptibilityModel):
    NAME = "Eigenvalue model of Susceptibility"

    VARNAMES = ["x", "y", "z"]

    VARNAMES_MM = {
        "x": r"$\chi_\mathregular{x}$",
        "y": r"$\chi_\mathregular{y}$",
        "z": r"$\chi_\mathregular{z}$",
    }

    UNITS_MM = {"x": r"Å$^3$", "y": r"Å$^3$", "z": r"Å$^3$"}

    BOUNDS = {"x": [-np.inf, np.inf], "y": [-np.inf, np.inf], "z": [-np.inf, np.inf]}

    @staticmethod
    def model(parameters: dict[str, float], nuclei: list[Nucleus]) -> dict[str, float]:
        """Computes predicted paramagnetic shifts for the eigenvalue model.

        Args:
            parameters: Model parameters (principal values). Keys are `VARNAMES`.
            nuclei: Nuclei for which shifts will be computed.

        Returns:
            Mapping from nucleus labels to predicted paramagnetic shifts.
        """
        chix = parameters["x"]
        chiy = parameters["y"]
        chiz = parameters["z"]

        tnsr = np.array(
            [
                [chix, 0, 0],
                [0, chiy, 0],
                [0, 0, chiz],
            ]
        )

        shifts = {
            nuc.label: 1.0 / 3.0 * np.trace(tnsr @ nuc.A.tensor) for nuc in nuclei
        }

        return shifts

    @staticmethod
    def totensor(params: dict[str, float]) -> NDArray:
        """Converts eigenvalue parameters to a diagonal susceptibility tensor.

        Args:
            params: Model parameters. Keys are `VARNAMES`.

        Returns:
            Susceptibility tensor as a ``(3, 3)`` NumPy array.
        """

        chix = params["x"]
        chiy = params["y"]
        chiz = params["z"]

        tensor = np.array(
            [
                [chix, 0, 0],
                [0, chiy, 0],
                [0, 0, chiz],
            ]
        )

        return tensor


class IsoEigenFitter(SusceptibilityModel):
    NAME = "Eigenvalue model of Susceptibility"

    VARNAMES = ["dxx", "dyy", "iso"]

    VARNAMES_MM = {
        "dxx": r"$\Delta\chi_\mathregular{xx}$",
        "dyy": r"$\Delta\chi_\mathregular{yy}$",
        "iso": r"$\chi_\mathregular{iso}$",
    }

    UNITS_MM = {"dxx": r"Å$^3$", "dyy": r"Å$^3$", "iso": r"Å$^3$"}

    BOUNDS = {"dxx": [-np.inf, np.inf], "dyy": [-np.inf, np.inf], "iso": [0, np.inf]}

    @staticmethod
    def model(parameters: dict[str, float], nuclei: list[Nucleus]) -> dict[str, float]:
        """Computes predicted paramagnetic shifts for the iso + deviatoric eigen model.

        Args:
            parameters: Model parameters. Keys are `VARNAMES`.
            nuclei: Nuclei for which shifts will be computed.

        Returns:
            Mapping from nucleus labels to predicted paramagnetic shifts.
        """
        dxx = parameters["dxx"]
        dyy = parameters["dyy"]
        iso = parameters["iso"]
        shifts = {
            nuc.label: nuc.A.iso * iso
            + nuc.shift.dia
            + 1.0 / 3.0 * dxx * (nuc.A.tensor[0, 0] - nuc.A.tensor[2, 2])
            + 1.0 / 3.0 * dyy * (nuc.A.tensor[1, 1] - nuc.A.tensor[2, 2])
            for nuc in nuclei
        }

        return shifts

    @staticmethod
    def totensor(params: dict[str, float]) -> NDArray:
        """Converts iso + deviatoric eigen parameters to a susceptibility tensor.

        Args:
            params: Model parameters. Keys are `VARNAMES`.

        Returns:
            Susceptibility tensor as a ``(3, 3)`` NumPy array.
        """

        dxx = params["dxx"]
        dyy = params["dyy"]
        iso = params["iso"]

        tensor = np.array(
            [
                [dxx + iso, 0, 0],
                [0, dyy + iso, 0],
                [0, 0, -(dxx + dyy) + iso],
            ]
        )

        return tensor


class FullSuscFitter(SusceptibilityModel):
    NAME = "Full Susceptibility"

    VARNAMES = ["xx", "xy", "xz", "yy", "yz", "zz"]

    VARNAMES_MM = {
        "xx": r"$\chi_{xx}$",
        "yy": r"$\chi_{yy}$",
        "zz": r"$\chi_{zz}$",
        "xy": r"$\chi_{xy}$",
        "xz": r"$\chi_{xz}$",
        "yz": r"$\chi_{yz}$",
    }

    UNITS_MM = {
        "xx": r"Å$^3$",
        "yy": r"Å$^3$",
        "zz": r"Å$^3$",
        "xy": r"Å$^3$",
        "xz": r"Å$^3$",
        "yz": r"Å$^3$",
    }

    BOUNDS = {
        "xx": [-np.inf, np.inf],
        "yy": [-np.inf, np.inf],
        "zz": [-np.inf, np.inf],
        "xy": [-np.inf, np.inf],
        "xz": [-np.inf, np.inf],
        "yz": [-np.inf, np.inf],
    }

    @staticmethod
    def model(parameters: dict[str, float], nuclei: list[Nucleus]) -> dict[str, float]:
        """Computes predicted paramagnetic shifts for the full susceptibility model.

        Args:
            parameters: Model parameters. Keys are `VARNAMES`.
            nuclei: Nuclei for which shifts will be computed.

        Returns:
            Mapping from nucleus labels to predicted paramagnetic shifts.
        """

        tnsr = FullSuscFitter.totensor(parameters)

        shifts = {
            nuc.label: 1.0 / 3.0 * np.trace(tnsr @ nuc.A.tensor) for nuc in nuclei
        }

        return shifts

    @staticmethod
    def design_matrix(nuclei: list[Nucleus], fix_vars: dict[str, float]):
        """Builds the design matrix for the full susceptibility linear model.

        Args:
            nuclei: Nuclei for which the model is evaluated.
            fix_vars: Fixed model parameters.

        Returns:
            The design matrix ``A`` in the linear system ``A x = b``.
        """
        to_pop = {"xx": 0, "xy": 1, "xz": 2, "yy": 3, "yz": 4, "zz": 5}

        design = []

        for nuc in nuclei:
            _vec = [
                nuc.A.tensor[0, 0],
                nuc.A.tensor[1, 0] + nuc.A.tensor[0, 1],
                nuc.A.tensor[0, 2] + nuc.A.tensor[2, 0],
                nuc.A.tensor[1, 1],
                nuc.A.tensor[2, 1] + nuc.A.tensor[1, 2],
                nuc.A.tensor[2, 2],
            ]

            for var in fix_vars.keys():
                _vec.pop(to_pop[var])
            design.append(_vec)

        design = 1.0 / 3.0 * np.asarray(design)

        return design

    @staticmethod
    def target_vector(
        nuclei: list[Nucleus],
        experiment: Experiment,
        fix_vars: dict[str, float],
    ):
        """Builds the target vector for the full susceptibility linear model.

        Args:
            nuclei: Nuclei for which the model is evaluated.
            experiment: Experimental data object.
            fix_vars: Fixed model parameters.

        Returns:
            The target vector ``b`` in the linear system ``A x = b``.
        """

        target = []

        for nuc in nuclei:
            _tgt = experiment[nuc.chem_label]
            to_subtract = {
                "xx": nuc.A.tensor[0, 0],
                "xy": nuc.A.tensor[1, 0] + nuc.A.tensor[0, 1],
                "xz": nuc.A.tensor[0, 2] + nuc.A.tensor[2, 0],
                "yy": nuc.A.tensor[1, 1],
                "yz": nuc.A.tensor[2, 1] + nuc.A.tensor[1, 2],
                "zz": nuc.A.tensor[2, 2],
            }
            for key, val in fix_vars.items():
                _tgt -= 1.0 / 3.0 * to_subtract[key] * val

            _tgt -= nuc.shift.dia
            target.append(_tgt)

        target = np.asarray(target)

        return target


def svd_stdev(curr_fit: OptimizeResult) -> tuple[list[float], list[bool]]:
    """Estimates standard deviations of fit parameters from the Jacobian.

    Uses an SVD of the Jacobian to identify near-singular directions. Singular values
    below a numerical threshold are discarded.

    Args:
        curr_fit: Result object returned by `scipy.optimize.least_squares` or a
            compatible object exposing ``jac``, ``fun``, and ``x``.

    Returns:
        A tuple ``(stdev, has_stdev)`` where:

        - `stdev` is the per-parameter standard deviation estimate.
        - `has_stdev` is a boolean list indicating whether each standard deviation
          is numerically meaningful.
    """

    # SVD of jacobian
    _, s, VT = la.svd(curr_fit.jac, full_matrices=False)
    # Zero threshold as multiple of machine precision
    threshold = np.finfo(float).eps * max(curr_fit.jac.shape) * s[0]
    # Find singular values = 0.
    nonzero_sing = s > threshold
    # Truncate to remove these values
    s = s[nonzero_sing]
    VT = VT[: s.size]
    # Calculate covariance of each parameter using truncated arrays
    pcov = VT.T / s**2 @ VT
    # Scale by reduced chi**2 to remove influence of input sigma (if present)
    # and just obtain standard deviation of fit
    chi2dof = np.sum(curr_fit.fun**2)
    chi2dof /= curr_fit.fun.size - curr_fit.x.size
    pcov *= chi2dof
    stdev = np.sqrt(np.diag(pcov))

    no_stdev = stdev > threshold

    if sum(nonzero_sing) == len(nonzero_sing):
        no_stdev = [True] * len(nonzero_sing)

    return stdev, no_stdev
