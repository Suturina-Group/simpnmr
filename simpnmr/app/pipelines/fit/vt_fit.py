# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fit temperature-dependent (VT) susceptibility components to experimental data.

Fits iso/ax/rh susceptibility components using the VT model to extract
temperature dependence, optionally fixing the temperature-independent
paramagnetism (TIP) from ab initio susceptibilities.
"""

import copy
import logging
import os
from typing import Any, Sequence

import numpy as np

# Application layer
from simpnmr.app.loaders.sh_load import load_g_tensor_ab_initio
from simpnmr.app.loaders.susc_load import load_susceptibilities
from simpnmr.app.policies.susc import resolve_susceptibility_source

# Core / domain
from simpnmr.core.fitting import vt
from simpnmr.core.phys.susc import get_g_corr_iso_susc

# IO layer
from simpnmr.io.csv.fit import save_slope_intercept
from simpnmr.io.qc import gateway as rdrs

# Visualisation
from simpnmr.viz.plots.susc import plot_exp_vs_ab_initio, plot_isoaxrh
from simpnmr.viz.style.theme import apply_profile

logger = logging.getLogger(__name__)


def fit_vt(
    config: Any,
    molecules: Sequence[Any],
    spin: float,
    susc_models: Sequence[Any] | None,
    plot_profile: str,
    show_plots: bool,
) -> None:
    """Fit VT susceptibility-component models to experimental pNMR data.

    This pipeline fits the temperature dependence of the isotropic, axial, and
    rhombic susceptibility components extracted from the input molecule set. It
    supports both the high-temperature-limit model and the second-order VT
    model. When requested, the pipeline also loads an ab initio susceptibility
    series, derives a fixed temperature-independent paramagnetism (TIP)
    correction from the ab initio reference point, and overlays the ab initio
    and analytic chiT curves in the output plots.

    The fitted results are written to disk, and the corresponding chiT plots are
    generated using the resolved visualisation profile.

    Args:
        config: Application configuration object containing VT fitting options,
            susceptibility-model input settings, ab initio file settings, and
            output paths.
        molecules: Molecule domain objects providing fitted susceptibility
            tensors and electronic-state context.
        spin: Spin quantum number ``S`` used in VT model evaluation and analytic
            chi(T) construction.
        susc_models: Optional susceptibility fit-model objects used to extract
            per-component experimental standard deviations and fixed-variable
            metadata.
        plot_profile: Plot-style profile name resolved through the visualisation
            theme system.
        show_plots: Whether to display generated plots interactively.

    Returns:
        None.

    Raises:
        ValueError: If ``fix_tip_from_ab_initio`` is requested with a non-ORCA
            ab initio source.
        ValueError: If no ab initio temperatures fall within the experimental
            fit window when TIP is fixed from ab initio data.
    """
    # Define the components to fit
    fit_component = ["iso", "ax", "rh"]

    # Default to high-temperature limit unless the user explicitly requests vt_2nd_order
    method = config.susc_vt_method or "ht_limit"

    # Read VT variables (may be None if not provided)
    susc_vt_variables = config.susc_vt_variables

    if method == "vt_2nd_order":
        assert susc_vt_variables is not None

    # Load the optional susceptibility-model input used for TIP extraction
    tip_type = config.susc_vt_tip_type

    # Load temperatures from the fitted susceptibility tensors
    temps_fit = np.array([mol.susc.temperature for mol in molecules])

    # Load the fitted Iso/Ax/Rho components
    chi_vals = {
        "iso": np.array([mol.susc.iso for mol in molecules]),
        "ax": np.array([abs(mol.susc.axiality) for mol in molecules]),
        "rh": np.array([abs(mol.susc.rhombicity) for mol in molecules]),
    }

    # Optional ab initio series and analytic chi(T) curves
    # (used only for vt_2nd_order + fixed TIP)
    ab_series: dict[str, np.ndarray] | None = None
    analytic_chi_vt: dict[str, np.ndarray] | None = None

    if tip_type == "fix_tip_from_ab_initio" and method == "vt_2nd_order":
        backend, section = resolve_susceptibility_source(
            config.susc_vt_ab_initio_file,
            config.susc_vt_ab_initio_format,
        )
        if backend != "orca" or section is None:
            raise ValueError("Only Orca is currently supported")

        load_g_tensor_ab_initio(
            molecule=molecules[0],
            susceptibility_file=config.susc_vt_ab_initio_file,
            susceptibility_format=config.susc_vt_ab_initio_format,
        )
        g_tensor = molecules[0].sh.g_tensor_ab_initio
        g_components = {
            "g_iso": molecules[0].sh.g_tensor_ab_initio_iso,
            "g_ax": molecules[0].sh.g_tensor_ab_initio_ax,
            "g_rho": molecules[0].sh.g_tensor_ab_initio_rho,
        }
        if g_tensor is None or any(value is None for value in g_components.values()):
            raise ValueError("Ab initio g-tensor components could not be loaded.")

        suscs_ab_initio = load_susceptibilities(
            config.susc_vt_ab_initio_file,
            config.susc_vt_ab_initio_format,
            electronic=molecules[0].electronic,
            g_tensor=g_tensor,
        )
        eff_H = rdrs.read_eff_hamiltonian_tensor(
            config.susc_vt_ab_initio_file,
            section=section,
        )

        # Find the ab initio susceptibility temperature closest to the fit temperatures
        ab_temps = np.array([s.temperature for s in suscs_ab_initio], dtype=float)
        idx = int(np.argmin(np.abs(ab_temps - np.max(temps_fit))))

        # Use only the reference susceptibility for the VT/TIP pipeline
        susc_ab_initio = copy.deepcopy(suscs_ab_initio[idx])

        # Rotate the effective Hamiltonian tensor into the chi eigenframe
        eff_H_rot = susc_ab_initio.eigvecs.T @ eff_H @ susc_ab_initio.eigvecs

        # Construct a diagonal g-tensor in the chi eigenframe
        g_rot_diag = np.diag(
            np.diag(susc_ab_initio.eigvecs.T @ g_tensor @ susc_ab_initio.eigvecs)
        )

        # Precompute g^2 invariants in the chi eigenframe for analytic chi(T) evaluation
        g_components_sq = vt.compute_g_sq_components(g_rot_diag)

        # Compute the axial and rhombic parts of the effective Hamiltonian tensor (J)
        D_J, E_J = vt.calculate_E_D_components(eff_H_rot)

        # Map VT component identifiers to Susceptibility attribute names
        comp_to_attr = {"iso": "iso_g_corr", "ax": "axiality", "rh": "rhombicity"}

        # Build the full ab initio chiT series
        ab_series_full = _build_ab_initio_chit_series(
            suscs_ab_initio,
            g_corr_iso=True,
            spin=spin,
            orbit=molecules[0].electronic.orbit_L,
            total_momentum_J=molecules[0].electronic.total_J,
            g_tensor=g_tensor,
        )

        exp_t = np.asarray(temps_fit, dtype=float)
        ab_inv_full = np.asarray(ab_series_full["inv_t"], dtype=float)
        ab_t_full = 1.0 / ab_inv_full

        exp_t_min = float(np.min(exp_t))
        exp_t_max = float(np.max(exp_t))
        ab_mask = (ab_t_full >= exp_t_min) & (ab_t_full <= exp_t_max)

        # Native ab initio temperature grid within the fit window.
        t_grid = np.asarray(ab_t_full[ab_mask], dtype=float)
        if t_grid.size == 0:
            raise ValueError(
                "No ab initio temperatures fall within the experimental fit window. "
                "Cannot fix TIP from ab initio data."
            )

        # Compute analytic chi(T) curves on the in-window ab initio temperature grid
        analytic_chi_vt = {}
        for comp in fit_component:
            analytic_chi_vt[comp] = np.asarray(
                vt.compute_analytic_component(
                    comp,
                    t_grid,
                    g_components_sq,
                    g_components,
                    D_J,
                    E_J,
                    spin,
                    total_J=molecules[0].electronic.total_J,
                ),
                dtype=float,
            )

        # Fix TIP using the analytic chi(T) value associated with the chosen
        # ab initio reference susceptibility temperature
        t_ref = float(susc_ab_initio.temperature)
        idx_ref = int(np.argmin(np.abs(t_grid - t_ref)))

        for comp in fit_component:
            analytic_val_ref = float(analytic_chi_vt[comp][idx_ref])
            ab_initio_value = getattr(susc_ab_initio, comp_to_attr[comp])
            if ab_initio_value is None:
                raise ValueError(
                    f"Missing ab initio susceptibility component: {comp_to_attr[comp]}"
                )
            tip_ref = vt.compute_tip_correction(
                ab_initio_value,
                analytic_val_ref,
                spin,
                total_J=molecules[0].electronic.total_J,
            )
            susc_vt_variables[comp]["tip"] = ["fix", float(tip_ref)]

        # Prepare ab initio series for plotting (clipped to the same temperature window)
        ab_series = {"inv_t": ab_inv_full[ab_mask]}
        for comp in fit_component:
            ab_series[comp] = np.asarray(ab_series_full[comp], dtype=float)[ab_mask]

        # Normalise ab initio chiT series by the Curie prefactor for consistency
        curie_prefactor = vt.compute_curie_prefactor(
            spin, molecules[0].electronic.total_J
        )
        for comp in fit_component:
            ab_series[comp] = ab_series[comp] / curie_prefactor

    # Initialize fitted chi errors to zero (if not available)
    chi_errors = {comp: np.zeros(len(temps_fit)) for comp in fit_component}

    # Fixed vars => sigma = 0.0. Missing stdev => sigma = 0.0.
    if susc_models:
        fix = getattr(susc_models[0], "fix_vars", {}) or {}

        if "iso" not in fix:
            chi_errors["iso"] = np.asarray(
                [float(m.fit_stdev.get("iso") or 0.0) for m in susc_models], dtype=float
            )

        if "ax" not in fix:
            chi_errors["ax"] = np.asarray(
                [float(m.fit_stdev.get("ax") or 0.0) for m in susc_models], dtype=float
            )

        if "rh" not in fix:
            chi_errors["rh"] = np.asarray(
                [float(m.fit_stdev.get("rh") or 0.0) for m in susc_models], dtype=float
            )

    # Create dictionaries to store fitted chiT values, errors, and fit parameters
    chiT_reduced = {}
    chiT_err_reduced = {}
    chiT_fit_params = {}

    # Fit the VT model parameters for each susceptibility component
    for comp in fit_component:
        if method == "vt_2nd_order":
            vals, errs, params = vt.fit_chit_linear_model(
                spin=spin,
                fit_temps=temps_fit,
                chi_vals=chi_vals[comp],
                chi_errors=chi_errors[comp],
                susc_vt_variables=susc_vt_variables[comp],
                total_J=molecules[0].electronic.total_J,
            )

        if temps_fit.size == 1 or method == "ht_limit":
            vals, errs, params = vt.compute_chit_high_t_limit(
                spin=spin,
                fit_temps=temps_fit,
                chi_vals=chi_vals[comp],
                chi_errors=chi_errors[comp],
                total_J=molecules[0].electronic.total_J,
            )

        # Store results
        chiT_reduced[comp] = vals
        chiT_err_reduced[comp] = errs
        chiT_fit_params[comp] = params

    # Precompute inverse temperature for plotting
    inv_temps_fit = 1.0 / temps_fit

    # Build the resolved plotting contract for this run.
    spec = apply_profile(plot_profile)

    if tip_type == "fix_tip_from_ab_initio" and method == "vt_2nd_order":
        with spec.context():
            plot_exp_vs_ab_initio(
                params=chiT_fit_params,
                g_sq=g_components_sq,
                inv_t=inv_temps_fit,
                ab_series=ab_series,
                analytic_chi_vt=analytic_chi_vt,
                spec=spec,
                show=show_plots,
                save=True,
                save_name=os.path.join(config.project_name, "exp_vs_ab_initio_susc"),
                verbose=True,
            )

    # Plot chiT temperature dependence
    with spec.context():
        plot_isoaxrh(
            vals=chiT_reduced,
            errs=chiT_err_reduced,
            params=chiT_fit_params,
            inv_t=inv_temps_fit,
            spec=spec,
            show=show_plots,
            save=True,
            save_name=os.path.join(
                config.project_name, "susceptibility_components_chiT"
            ),
            verbose=True,
        )

    # Write iso/ax/rh fit parameters to CSV
    out_file = os.path.join(config.project_name, "isoaxrh_fit.csv")
    fits_list = [
        chiT_fit_params.get("iso"),
        chiT_fit_params.get("ax"),
        chiT_fit_params.get("rh"),
    ]
    save_slope_intercept(fits_list, spin=spin, file_name=out_file)

    return


def _build_ab_initio_chit_series(
    suscs_ab_initio: list,
    *,
    g_corr_iso: bool = False,
    spin: float | None = None,
    orbit: float | None = None,
    total_momentum_J: float | None = None,
    g_tensor: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Project susceptibility objects onto a chiT series for VT fitting.

    This helper sorts the ab initio susceptibility objects by temperature,
    ensures irreducible tensor components are available, and returns chiT
    component arrays on the ab initio temperature grid. The isotropic channel
    can be taken either from the stored susceptibility ``iso`` values or
    recomputed through the g-tensor-corrected isotropic susceptibility model.

    Args:
        suscs_ab_initio: Susceptibility domain objects to project.
        g_corr_iso: Whether to recompute the isotropic channel with the
            g-tensor-corrected isotropic susceptibility model.
        spin: Spin quantum number ``S`` required when ``g_corr_iso=True``.
        orbit: Orbital angular momentum quantum number ``L`` used when
            ``g_corr_iso=True``.
        total_momentum_J: Total angular momentum quantum number ``J`` used when
            ``g_corr_iso=True``.
        g_tensor: g-tensor as a ``(3, 3)`` array required when
            ``g_corr_iso=True``.

    Returns:
        Dictionary with sorted temperature grid data:
            ``temps``: Temperatures in K.
            ``inv_t``: Inverse temperatures in K^-1.
            ``iso``: chiT isotropic values in ``Å^3 K``.
            ``ax``: chiT axial values in ``Å^3 K``.
            ``rh``: chiT rhombic values in ``Å^3 K``.

    Raises:
        ValueError: If ``suscs_ab_initio`` is empty.
        ValueError: If ``g_corr_iso=True`` but ``spin`` or ``g_tensor`` is not
            provided.
    """
    if not suscs_ab_initio:
        raise ValueError("suscs_ab_initio must be a non-empty list.")

    if g_corr_iso:
        if spin is None or g_tensor is None:
            raise ValueError(
                "g_corr_iso=True requires spin and g_tensor to be provided."
            )

    # Ensure irreducible components / eigenframes are available
    for s in suscs_ab_initio:
        s.calc_irred()

    temps = np.array([float(s.temperature) for s in suscs_ab_initio], dtype=float)

    order = np.argsort(temps)
    temps = temps[order]
    suscs_sorted = [suscs_ab_initio[i] for i in order]

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_t = 1.0 / temps

    if g_corr_iso:
        iso_vals = []
        for s in suscs_sorted:
            iso_vals.append(
                float(
                    get_g_corr_iso_susc(
                        spin=float(spin),
                        orbit=0.0 if orbit is None else float(orbit),
                        g_tensor=np.asarray(g_tensor, dtype=float),
                        chi_tensors=s.tensor,
                        total_momentum_J=total_momentum_J,
                    )
                )
            )
        iso_base = np.asarray(iso_vals, dtype=float)
    else:
        iso_base = np.array([float(s.iso) for s in suscs_sorted], dtype=float)

    iso = iso_base * temps
    ax = np.array([float(s.axiality) for s in suscs_sorted], dtype=float) * temps
    rh = np.array([float(s.rhombicity) for s in suscs_sorted], dtype=float) * temps

    return {
        "temps": temps,
        "inv_t": inv_t,
        "iso": iso,
        "ax": ax,
        "rh": rh,
    }
