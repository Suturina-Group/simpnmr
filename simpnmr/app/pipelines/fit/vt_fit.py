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
from simpnmr.core.const.physics import C, H
from simpnmr.core.fitting import vt
from simpnmr.core.phys.susc import get_g_corr_iso_susc

# IO layer
from simpnmr.io.csv.fit import save_slope_intercept
from simpnmr.io.qc import gateway as rdrs

# Visualisation
from simpnmr.viz.plots.sh_analysis import plot_g_iso_solution_lines
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
    save_chi_t: bool = True,
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

    # ── Optional ab initio reference (ORCA NEVPT2 / CASSCF) ──────────────────
    # Loaded whenever an ab_initio_file is provided, regardless of TIP type.
    # Used for: comparison plots, ζ_eff extraction, and (optionally) TIP fixing.
    ab_series: dict[str, np.ndarray] | None = None
    analytic_chi_vt: dict[str, np.ndarray] | None = None
    g_sq: dict[str, float] | None = None
    zeta_eff_from_orca: float | None = None
    orca_point: tuple[float, float, float] | None = None  # (g_iso, g_ax, D_cm)

    ab_initio_file = config.susc_vt_ab_initio_file or ""
    ab_initio_format = config.susc_vt_ab_initio_format or ""

    if ab_initio_file and ab_initio_format:
        if "orca" not in ab_initio_format:
            raise ValueError("Only Orca is currently supported for ab initio data")
        section = ab_initio_format.split("orca_", 1)[1]

        g_tensor = rdrs.read_g_tensor_ab_initio(ab_initio_file, section=section)
        suscs_ab_initio = load_susceptibilities(
            ab_initio_file,
            ab_initio_format,
            electronic=molecules[0].electronic,
            g_tensor=g_tensor,
        )
        eff_H = rdrs.read_eff_hamiltonian_tensor(ab_initio_file, section=section)

        # Build the full ab initio chiT series
        ab_series_full = _build_ab_initio_chit_series(
            suscs_ab_initio,
            g_corr_iso=True,
            spin=spin,
            orbit=molecules[0].electronic.orbit_L,
            total_momentum_J=molecules[0].electronic.total_J,
            g_tensor=g_tensor,
        )

        ab_inv_full = np.asarray(ab_series_full["inv_t"], dtype=float)
        ab_t_full = 1.0 / ab_inv_full
        exp_t_min = float(np.min(temps_fit))
        exp_t_max = float(np.max(temps_fit))
        ab_mask = (ab_t_full >= exp_t_min) & (ab_t_full <= exp_t_max)
        t_grid = np.asarray(ab_t_full[ab_mask], dtype=float)

        if t_grid.size == 0:
            if tip_type == "fix_tip_from_ab_initio":
                raise ValueError(
                    "No ab initio temperatures fall within the experimental "
                    "fit window. Cannot fix TIP from ab initio data."
                )
            logger.warning(
                "No ab initio temperatures in the experimental window; "
                "ab initio comparison plot will be skipped."
            )

        else:
            # Clip and normalise ab initio chiT series
            curie_prefactor = vt.compute_chi_prefactor(
                spin, molecules[0].electronic.total_J
            )
            ab_series = {"inv_t": ab_inv_full[ab_mask]}
            for comp in fit_component:
                ab_series[comp] = (
                    np.asarray(ab_series_full[comp], dtype=float)[ab_mask]
                    / curie_prefactor
                )

            # Derive ζ_eff from ORCA D and g_ax via the LFT relation:
            #   ζ_eff = −4S · D / Δg
            # D comes from the effective Hamiltonian (unrotated principal value);
            # Δg from the principal values of the ab initio g-tensor.
            if eff_H is not None and g_tensor is not None:
                D_J_raw, _ = vt.calculate_E_D_components(eff_H)
                D_cm_orca = D_J_raw / (H * C * 100.0)
                g_sym = 0.5 * (g_tensor + g_tensor.T)
                g_eig = np.sort(np.linalg.eigvalsh(g_sym))
                g_iso_orca = float(np.mean(g_eig))
                g_ax_orca = float(g_eig[-1] - g_eig[0])
                orca_point = (g_iso_orca, g_ax_orca, D_cm_orca)
                if abs(g_ax_orca) > 1e-6:
                    zeta_eff_from_orca = float(
                        -4.0 * spin * D_cm_orca / g_ax_orca
                    )
                    logger.info(
                        "ζ_eff derived from ORCA (D=%.2f cm⁻¹, Δg=%.4f): "
                        "%.1f cm⁻¹",
                        D_cm_orca, g_ax_orca, zeta_eff_from_orca,
                    )

            # TIP fixing — only when explicitly requested with vt_2nd_order
            if tip_type == "fix_tip_from_ab_initio" and method == "vt_2nd_order":
                ab_temps = np.array(
                    [s.temperature for s in suscs_ab_initio], dtype=float
                )
                idx = int(np.argmin(np.abs(ab_temps - np.max(temps_fit))))
                susc_ab_initio = copy.deepcopy(suscs_ab_initio[idx])

                # Rotate tensors into the chi eigenframe
                eff_H_rot = (
                    susc_ab_initio.eigvecs.T @ eff_H @ susc_ab_initio.eigvecs
                )
                g_rot_diag = np.diag(
                    np.diag(
                        susc_ab_initio.eigvecs.T
                        @ g_tensor
                        @ susc_ab_initio.eigvecs
                    )
                )

                g_components_sq = vt.compute_g_sq_components(g_rot_diag)
                _gx = float(g_rot_diag[0, 0])
                _gy = float(g_rot_diag[1, 1])
                _gz = float(g_rot_diag[2, 2])
                g_components = {
                    "g_iso": (_gx + _gy + _gz) / 3.0,
                    "g_ax": _gz - (_gx + _gy + _gz) / 3.0,
                    "g_rho": (_gx - _gy) / 2.0,
                }
                D_J, E_J = vt.calculate_E_D_components(eff_H_rot)

                # Analytic VT curves on the clipped ab initio temperature grid
                analytic_chi_vt = {}
                for comp in fit_component:
                    analytic_chi_vt[comp] = np.asarray(
                        vt.compute_analytic_component(
                            comp, t_grid, g_components_sq, g_components,
                            D_J, E_J, spin,
                            total_J=molecules[0].electronic.total_J,
                        ),
                        dtype=float,
                    )

                # Fix TIP from the analytic value at the reference temperature
                t_ref = float(susc_ab_initio.temperature)
                idx_ref = int(np.argmin(np.abs(t_grid - t_ref)))
                comp_to_attr = {
                    "iso": "iso_g_corr", "ax": "axiality", "rh": "rhombicity"
                }
                for comp in fit_component:
                    analytic_val_ref = float(analytic_chi_vt[comp][idx_ref])
                    tip_ref = vt.compute_tip_correction(
                        getattr(susc_ab_initio, comp_to_attr[comp]),
                        analytic_val_ref,
                        spin,
                        total_J=molecules[0].electronic.total_J,
                    )
                    susc_vt_variables[comp]["tip"] = ["fix", float(tip_ref)]

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

    if save_chi_t and ab_series is not None:
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
                save_name=os.path.join(
                    config.project_name, "exp_vs_ab_initio_susc"
                ),
                verbose=True,
            )

    # Plot chiT temperature dependence
    if save_chi_t:
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
                    config.project_name,
                    "susceptibility_components_chiT",
                ),
                verbose=True,
            )

    # g_iso solution-line figure (axial-only: rh ≈ 0)
    rh_intercept = chiT_fit_params.get("rh", {}).get("intercept", 0.0)
    if save_chi_t and abs(rh_intercept) < 1e-4:
        with spec.context():
            plot_g_iso_solution_lines(
                ax_intercept=chiT_fit_params["ax"]["intercept"],
                ax_slope=chiT_fit_params["ax"]["slope"],
                iso_intercept=chiT_fit_params["iso"]["intercept"],
                chiT_red_iso=chiT_reduced["iso"],
                inv_t=inv_temps_fit,
                spin=spin,
                total_J=molecules[0].electronic.total_J,
                ax_intercept_err=float(
                    chiT_fit_params["ax"].get("intercept_err") or 0.0
                ),
                ax_slope_err=float(
                    chiT_fit_params["ax"].get("slope_err") or 0.0
                ),
                iso_intercept_err=float(
                    chiT_fit_params["iso"].get("intercept_err") or 0.0
                ),
                chiT_err_iso=chiT_err_reduced["iso"],
                zeta_eff=(
                    zeta_eff_from_orca
                    if zeta_eff_from_orca is not None
                    else config.susc_vt_zeta_eff
                ),
                zeta_eff_tol=config.susc_vt_zeta_eff_tol,
                orca_point=orca_point,
                spec=spec,
                show=show_plots,
                save=True,
                save_name=os.path.join(
                    config.project_name, "g_iso_solution_lines"
                ),
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
