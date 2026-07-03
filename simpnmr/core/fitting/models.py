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
from simpnmr.core.util.uncertainty import delta_method_sigma

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
            A `Susceptibility` object at `self.temperature` with canonical
            ``chi.iso`` assigned from the fitted model.
        """
        tensor = self.totensor(self.final_var_values)
        susc = Susceptibility(tensor, self.temperature)

        fitted_iso = self.final_var_values.get("iso")
        if fitted_iso is None:
            fitted_iso = float(np.trace(tensor) / 3.0)

        susc.iso = float(fitted_iso)
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

    def _compute_euler_stdev(self) -> None:
        """Delta-method propagation of fit uncertainties to ZYZ Euler angles.

        Uses central finite differences through ``totensor → Susceptibility``
        for each fitted parameter that has a valid standard deviation.  The
        resulting uncertainties (in degrees) are stored in ``self.fit_stdev``
        under the keys ``"alpha"``, ``"beta"``, and ``"gamma"``.

        When β is within 1° of its singular values (0° or 180°), α and γ
        become degenerate and their uncertainties are set to ``nan``.
        """
        params0 = {k: float(v) for k, v in self.final_var_values.items()}

        # Collect fitted parameters that have a usable stdev
        fit_names = []
        sigmas = []
        for name in self.VARNAMES:
            if name not in self.fit_vars:
                continue
            sig = self.fit_stdev.get(name)
            if sig is None or not np.isfinite(sig) or sig <= 0:
                continue
            fit_names.append(name)
            sigmas.append(float(sig))

        if not fit_names:
            return

        # Base Euler angles
        susc0 = Susceptibility(self.totensor(params0), self.temperature)
        alpha0, beta0, gamma0 = susc0.alpha, susc0.beta, susc0.gamma

        # Jacobian columns: d(alpha, beta, gamma) / d(param_i)
        d_alpha = []
        d_beta = []
        d_gamma = []

        for name, sig in zip(fit_names, sigmas):
            p_plus = dict(params0)
            p_minus = dict(params0)
            p_plus[name] = params0[name] + sig
            p_minus[name] = params0[name] - sig

            try:
                s_plus = Susceptibility(self.totensor(p_plus), self.temperature)
                s_minus = Susceptibility(self.totensor(p_minus), self.temperature)
            except Exception:
                d_alpha.append(0.0)
                d_beta.append(0.0)
                d_gamma.append(0.0)
                continue

            def _wrap(diff: float) -> float:
                """Wrap angle difference into [-180, 180)."""
                return (diff + 180.0) % 360.0 - 180.0

            two_sig = 2.0 * sig
            d_alpha.append(_wrap(s_plus.alpha - s_minus.alpha) / two_sig)
            d_beta.append((s_plus.beta - s_minus.beta) / two_sig)
            d_gamma.append(_wrap(s_plus.gamma - s_minus.gamma) / two_sig)

        sig_arr = np.asarray(sigmas, dtype=float)
        self.fit_stdev["beta"] = float(
            np.sqrt(np.sum((np.asarray(d_beta) * sig_arr) ** 2))
        )

        # α and γ are degenerate when β ≈ 0° or 180° (gimbal lock)
        if abs(beta0) < 1.0 or abs(beta0 - 180.0) < 1.0:
            self.fit_stdev["alpha"] = float("nan")
            self.fit_stdev["gamma"] = float("nan")
        else:
            self.fit_stdev["alpha"] = float(
                np.sqrt(np.sum((np.asarray(d_alpha) * sig_arr) ** 2))
            )
            self.fit_stdev["gamma"] = float(
                np.sqrt(np.sum((np.asarray(d_gamma) * sig_arr) ** 2))
            )

    def _compute_canonical_stdev(self) -> None:
        """Delta-method stdevs for canonical physical quantities.

        Populates ``fit_stdev`` with standard deviations for ``"iso"``,
        ``"ax"`` (axiality), ``"rh"`` (rhombicity), ``"rh_over_ax"``
        (Δχ_rh / Δχ_ax ratio), and the three ZYZ Euler angles
        ``"alpha"``, ``"beta"``, ``"gamma"`` via central finite differences
        through ``totensor → Susceptibility``.

        Existing valid entries are preserved (e.g. those set by the
        optimiser or a model-specific ``_post_fit``).  The ratio stdev is
        derived analytically from the ``"rh"`` and ``"ax"`` stdevs.

        Angular finite differences for α and γ use minimum-arc arithmetic
        to avoid wrap-around artefacts at 0°/360°.
        """

        def _valid(key: str) -> bool:
            v = self.fit_stdev.get(key)
            return v is not None and np.isfinite(float(v)) and float(v) > 0

        def _ang_diff(a: float, b: float) -> float:
            """Signed angular difference a − b in (−180°, 180°]."""
            return (a - b + 180.0) % 360.0 - 180.0

        need_iso = not _valid("iso")
        need_ax = not _valid("ax")
        need_rh = not _valid("rh")
        need_alpha = not _valid("alpha")
        need_beta = not _valid("beta")
        need_gamma = not _valid("gamma")

        if need_iso or need_ax or need_rh or need_alpha or need_beta or need_gamma:
            params0 = {k: float(v) for k, v in self.final_var_values.items()}
            d_iso, d_ax, d_rh = [], [], []
            d_alpha, d_beta, d_gamma, sigs = [], [], [], []

            for name in self.VARNAMES:
                if name not in self.fit_vars:
                    continue
                sig = self.fit_stdev.get(name)
                if sig is None or not np.isfinite(float(sig)) or float(sig) <= 0:
                    continue
                sig = float(sig)
                p_plus = {**params0, name: params0[name] + sig}
                p_minus = {**params0, name: params0[name] - sig}
                try:
                    sp = Susceptibility(self.totensor(p_plus), self.temperature)
                    sm = Susceptibility(self.totensor(p_minus), self.temperature)
                except Exception:
                    continue
                d_iso.append((float(sp.iso) - float(sm.iso)) / (2.0 * sig))
                d_ax.append(
                    (float(sp.axiality) - float(sm.axiality)) / (2.0 * sig)
                )
                d_rh.append(
                    (float(sp.rhombicity) - float(sm.rhombicity)) / (2.0 * sig)
                )
                d_alpha.append(
                    _ang_diff(float(sp.alpha), float(sm.alpha)) / (2.0 * sig)
                )
                d_beta.append(
                    (float(sp.beta) - float(sm.beta)) / (2.0 * sig)
                )
                d_gamma.append(
                    _ang_diff(float(sp.gamma), float(sm.gamma)) / (2.0 * sig)
                )
                sigs.append(sig)

            if sigs:
                sa = np.asarray(sigs, dtype=float)
                if need_iso and "iso" not in self.fix_vars:
                    self.fit_stdev["iso"] = float(
                        np.sqrt(np.dot(np.asarray(d_iso) ** 2, sa ** 2))
                    )
                if need_ax and "ax" not in self.fix_vars:
                    self.fit_stdev["ax"] = float(
                        np.sqrt(np.dot(np.asarray(d_ax) ** 2, sa ** 2))
                    )
                if need_rh and "rh" not in self.fix_vars:
                    self.fit_stdev["rh"] = float(
                        np.sqrt(np.dot(np.asarray(d_rh) ** 2, sa ** 2))
                    )
                if need_alpha:
                    self.fit_stdev["alpha"] = float(
                        np.sqrt(np.dot(np.asarray(d_alpha) ** 2, sa ** 2))
                    )
                if need_beta:
                    self.fit_stdev["beta"] = float(
                        np.sqrt(np.dot(np.asarray(d_beta) ** 2, sa ** 2))
                    )
                if need_gamma:
                    self.fit_stdev["gamma"] = float(
                        np.sqrt(np.dot(np.asarray(d_gamma) ** 2, sa ** 2))
                    )

        # Ratio Δχ_rh / Δχ_ax — analytical propagation from rh and ax stdevs
        if not _valid("rh_over_ax"):
            sig_rh = self.fit_stdev.get("rh")
            sig_ax = self.fit_stdev.get("ax")
            params0 = {k: float(v) for k, v in self.final_var_values.items()}
            susc0 = Susceptibility(self.totensor(params0), self.temperature)
            ax0 = float(susc0.axiality)
            rh0 = float(susc0.rhombicity)
            if (
                sig_rh is not None
                and np.isfinite(float(sig_rh))
                and sig_ax is not None
                and np.isfinite(float(sig_ax))
                and abs(ax0) > 1e-12
            ):
                self.fit_stdev["rh_over_ax"] = float(
                    np.sqrt(
                        (float(sig_rh) / ax0) ** 2
                        + (rh0 * float(sig_ax) / ax0 ** 2) ** 2
                    )
                )

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
                present = [lab for lab in group if lab in trial_shifts]
                if not present:
                    continue
                group_average = np.mean([trial_shifts[lab] for lab in present])
                group_size = len(present)
                for lab in present:
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

        # Chemical label to paramagnetic shift — skip nuclei absent from experiment
        _missing = sorted({
            nuc.chem_label for nuc in molecule.nuclei
            if nuc.chem_label not in experiment
        })
        if _missing:
            logger.warning(
                "Nuclei skipped in fit (no experimental signal): %s",
                ", ".join(_missing),
            )
        _fit_nuclei = [
            nuc for nuc in molecule.nuclei if nuc.chem_label in experiment
        ]
        al_to_para_shift = {
            nuc.label: experiment[nuc.chem_label].shift - nuc.shift.dia
            for nuc in _fit_nuclei
        }

        curr_fit = least_squares(
            fun=self.residual_from_float_list,
            args=(
                self.fit_vars,
                self.fix_vars,
                _fit_nuclei,
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
            self._compute_canonical_stdev()
            self._compute_euler_stdev()

            # R2
            self.mae = np.sum(np.abs(curr_fit.fun)) / len(curr_fit.fun)
            ss_res = np.sum(curr_fit.fun**2)
            self.rmse = np.sqrt(ss_res / len(curr_fit.fun))
            ecs = [al_to_para_shift[nuc.label] for nuc in _fit_nuclei]
            ss_tot = np.sum((ecs - np.mean(ecs)) ** 2)
            self.r2 = 1 - (ss_res / ss_tot)
            self.adj_r2 = 1 - (1 - self.r2) * (len(ecs) - 1) / (
                len(ecs) - len(self.fit_vars) - 1
            )

        return


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
            nuc.label: 1.0 / 3.0 * np.trace(tnsr @ nuc.A.tensor_full) for nuc in nuclei
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

    def _post_fit(self) -> None:
        """Adds derived uncertainties for susceptibility invariants.

        Computes 1σ uncertainties for `Susceptibility.axiality` and
        `Susceptibility.rhombicity` via a finite-difference Jacobian with respect to
        the split parameters (iso, dxx, dyy, dxy, dxz, dyz) and propagates using the
        package delta-method helper.

        Notes:
            - Uses the independence assumption (diagonal covariance) consistent with
              `delta_method_sigma`.
            - Fixed parameters are treated as having zero uncertainty.
        """

        # Build input sigma vector in VARNAMES order; fixed params => sigma = 0.
        sig_in = []
        for name in self.VARNAMES:
            if name in self.fit_vars:
                sig = self.fit_stdev.get(name)
                sig_in.append(float(sig) if sig is not None else np.nan)
            else:
                sig_in.append(0.0)
        sig_in = np.asarray(sig_in, dtype=float)

        # Nothing to propagate if all inputs are fixed.
        if np.all(sig_in == 0.0):
            return

        base = {k: float(v) for k, v in self.final_var_values.items()}

        def _invariants(params: dict[str, float]) -> NDArray:
            tensor = SplitFitter.totensor(params)
            susc = Susceptibility(tensor, self.temperature)
            return np.asarray(
                [float(susc.axiality), float(susc.rhombicity)], dtype=float
            )

        jac = np.zeros((2, len(self.VARNAMES)), dtype=float)

        # Central finite differences for d(axiality, rhombicity)/d(params).
        for i, name in enumerate(self.VARNAMES):
            x0 = base[name]
            step = 1e-6 * max(1.0, abs(x0))

            p_plus = base.copy()
            p_minus = base.copy()
            p_plus[name] = x0 + step
            p_minus[name] = x0 - step

            y_plus = _invariants(p_plus)
            y_minus = _invariants(p_minus)

            jac[:, i] = (y_plus - y_minus) / (2.0 * step)

        sig_out = delta_method_sigma(jac, sig_in)

        # Store derived uncertainties alongside fitted ones.
        self.fit_stdev["ax"] = float(sig_out[0])
        self.fit_stdev["rh"] = float(sig_out[1])
        return

    def fit_to(
        self,
        molecule: Molecule,
        experiment: Experiment,
        verbose: bool = True,
        average_labels: list[list[str]] = [],
    ) -> None:
        """Fit using a direct linear solver (lsq_linear).

        The model is linear in all six parameters, so the Jacobian is constant
        and the solution is exact in one step.  ``lsq_linear`` is used rather
        than ``lstsq`` so the ``iso ≥ 0`` bound is enforced.
        """
        _missing = sorted({
            nuc.chem_label for nuc in molecule.nuclei
            if nuc.chem_label not in experiment
        })
        if _missing:
            logger.warning(
                "Nuclei skipped in fit (no experimental signal): %s",
                ", ".join(_missing),
            )
        _fit_nuclei = [
            nuc for nuc in molecule.nuclei if nuc.chem_label in experiment
        ]

        self.temperature = experiment.temperature

        # Full design matrix (n × 6) and target vector
        A_full = self._design_matrix(_fit_nuclei)
        b_full = np.array([
            experiment[nuc.chem_label].shift - nuc.shift.dia
            for nuc in _fit_nuclei
        ], dtype=float)

        # Apply group averaging: replace rows for each group with the mean
        # row, scaled by 1/sqrt(n) so the group contributes as one signal.
        label_to_idx = {nuc.label: i for i, nuc in enumerate(_fit_nuclei)}
        for group in average_labels:
            idxs = [label_to_idx[lab] for lab in group if lab in label_to_idx]
            if len(idxs) < 2:
                continue
            n = len(idxs)
            mean_row = A_full[idxs].mean(axis=0)
            for i in idxs:
                A_full[i] = mean_row / np.sqrt(n)
                b_full[i] = b_full[i] / np.sqrt(n)

        # Subtract fixed-variable contributions from the target
        b = b_full.copy()
        for k, v in self.fix_vars.items():
            col = self.VARNAMES.index(k)
            b -= A_full[:, col] * v

        # Keep only columns for fitted variables
        fit_col_idxs = [self.VARNAMES.index(k) for k in self.fit_vars]
        A = A_full[:, fit_col_idxs]

        bounds = np.array(
            [self.BOUNDS[name] for name in self.fit_vars.keys()]
        ).T

        curr_fit = lsq_linear(A, b, bounds=bounds)

        fit_var_names = list(self.fit_vars.keys())
        curr_fit_dict = {
            name: val for name, val in zip(fit_var_names, curr_fit.x)
        }

        if curr_fit.status == 0:
            if verbose:
                logger.warning(
                    "Fit at %s K failed - Too many iterations", self.temperature
                )
            self.final_var_values = copy.deepcopy(curr_fit_dict)
            self.fit_stdev = {label: np.nan for label in fit_var_names}
            self.fit_status = False
            self.mae = np.NaN
            self.rmse = np.NaN
            self.r2 = np.NaN
            self.adj_r2 = np.NaN
        else:
            curr_fit.jac = A
            stdev, _ = svd_stdev(curr_fit)

            self.fit_stdev = {
                label: val for label, val in zip(fit_var_names, stdev)
            }
            self.fit_status = True
            self.final_var_values = copy.deepcopy(curr_fit_dict)
            for key, val in self.fix_vars.items():
                self.final_var_values[key] = val

            self._post_fit()
            self._compute_canonical_stdev()
            self._compute_euler_stdev()

            self.mae = np.sum(np.abs(curr_fit.fun)) / len(curr_fit.fun)
            ss_res = np.sum(curr_fit.fun ** 2)
            self.rmse = np.sqrt(ss_res / len(curr_fit.fun))
            al_to_para = {
                nuc.label: experiment[nuc.chem_label].shift - nuc.shift.dia
                for nuc in _fit_nuclei
            }
            ecs = [al_to_para[nuc.label] for nuc in _fit_nuclei]
            ss_tot = np.sum((np.asarray(ecs) - np.mean(ecs)) ** 2)
            self.r2 = 1 - (ss_res / ss_tot)
            self.adj_r2 = 1 - (1 - self.r2) * (len(ecs) - 1) / (
                len(ecs) - len(self.fit_vars) - 1
            )

    @staticmethod
    def _design_matrix(nuclei: list[Nucleus]) -> np.ndarray:
        """Build design matrix; columns: [iso, dxx, dyy, dxy, dxz, dyz]."""
        rows = []
        for nuc in nuclei:
            A = nuc.A.tensor_full
            rows.append([
                (A[0, 0] + A[1, 1] + A[2, 2]) / 3.0,   # iso: Tr(A)/3
                (A[0, 0] - A[2, 2]) / 3.0,              # dxx
                (A[1, 1] - A[2, 2]) / 3.0,              # dyy
                (A[0, 1] + A[1, 0]) / 3.0,              # dxy
                (A[0, 2] + A[2, 0]) / 3.0,              # dxz
                (A[1, 2] + A[2, 1]) / 3.0,              # dyz
            ])
        return np.array(rows, dtype=float)


class IsoAxRhFitter(SusceptibilityModel):
    NAME = "Isotropic, Axial, and Rhombic over Axial Components of Susceptibility"

    VARNAMES = ["iso", "ax", "rh_over_ax", "alpha", "beta", "gamma"]

    VARNAMES_MM = {
        "iso": r"$\chi_\mathregular{iso}$",
        "ax": r"$\Delta\chi_\mathregular{ax}$",
        "rh_over_ax": r"$\Delta\chi_\mathregular{rh} / \Delta\chi_\mathregular{ax}$",
        "alpha": r"$\alpha$",
        "beta": r"$\beta$",
        "gamma": r"$\gamma$",
    }

    UNITS_MM = {
        "iso": r"Å$^3$",
        "ax": r"Å$^3$",
        "rh_over_ax": "",
        "alpha": r"°",
        "beta": r"°",
        "gamma": r"°",
    }

    BOUNDS = {
        "iso": [0.0, np.inf],
        "ax": [-np.inf, np.inf],
        "rh_over_ax": [0.0, 1 / 3],
        "alpha": [0.0, 360.0],
        "beta": [0.0, 180.0],
        "gamma": [0.0, 360.0],
    }

    @staticmethod
    def _zyz_rotation(alpha_deg: float, beta_deg: float, gamma_deg: float) -> NDArray:
        """ZYZ rotation matrix R = Rz(α) · Ry(β) · Rz(γ).

        Consistent with ``Susceptibility.calc_euler``: the rotation maps the
        lab frame to the principal-axis frame, so
        ``χ_lab = R · χ_paf · Rᵀ``.
        """
        a = np.deg2rad(alpha_deg)
        b = np.deg2rad(beta_deg)
        g = np.deg2rad(gamma_deg)
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cg, sg = np.cos(g), np.sin(g)
        Rza = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        Ryb = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
        Rzg = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
        return Rza @ Ryb @ Rzg

    @staticmethod
    def model(parameters: dict[str, float], nuclei: list[Nucleus]) -> dict[str, float]:
        """Predicted paramagnetic shifts for the iso/ax/rh/Euler model."""
        tnsr = IsoAxRhFitter.totensor(parameters)
        return {
            nuc.label: 1.0 / 3.0 * np.trace(tnsr @ nuc.A.tensor_full)
            for nuc in nuclei
        }

    @staticmethod
    def totensor(params: dict[str, float]) -> NDArray:
        """Build χ_lab = R(α,β,γ) · χ_paf · R(α,β,γ)ᵀ.

        χ_paf is the diagonal principal-axis-frame tensor parameterised by
        ``iso``, ``ax``, and ``rh_over_ax``.  The ZYZ rotation R then maps
        it to the lab frame.
        """
        chi_paf = np.array(
            [
                [-params["ax"] / 3 + params["rh_over_ax"] * params["ax"], 0.0, 0.0],
                [0.0, -params["ax"] / 3 - params["rh_over_ax"] * params["ax"], 0.0],
                [0.0, 0.0, 2.0 / 3.0 * params["ax"]],
            ]
        ) + np.eye(3) * params["iso"]
        R = IsoAxRhFitter._zyz_rotation(
            params["alpha"], params["beta"], params["gamma"]
        )
        return R @ chi_paf @ R.T

    def _post_fit(self) -> None:
        """Propagate rh = ax * rh_over_ax uncertainty."""
        ax = self.final_var_values.get("ax")
        rh_over_ax = self.final_var_values.get("rh_over_ax")
        ax_st_dev = self.fit_stdev.get("ax")
        rh_over_ax_st_dev = self.fit_stdev.get("rh_over_ax")

        if ax is None or rh_over_ax is None:
            self.fit_stdev.pop("rh", None)
            return
        if ax_st_dev is None:
            self.fit_stdev.pop("rh", None)
            return
        if rh_over_ax_st_dev is None:
            if "rh_over_ax" in self.fix_vars:
                self.fit_stdev["rh"] = float(np.abs(rh_over_ax) * ax_st_dev)
            else:
                self.fit_stdev.pop("rh", None)
            return
        self.fit_stdev["rh"] = float(
            np.hypot(rh_over_ax * ax_st_dev, ax * rh_over_ax_st_dev)
        )

    def _compute_euler_stdev(self) -> None:
        """No-op: α, β, γ standard deviations come directly from the fit."""
        return


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
