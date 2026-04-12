# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define SimpNMR command-line entry points.

Parses CLI arguments and dispatches subcommands to application pipelines.
"""

import argparse
import logging
import os

import yaml

from simpnmr import __version__
from simpnmr.app.params.options import RuntimeSettings
from simpnmr.app.params.settings import apply_runtime_settings
from simpnmr.cfg import config as cfg
from simpnmr.cli.set_logging import setup_logging
from simpnmr.io.qc.errors import QCError

logger = logging.getLogger(__name__)


def predict_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for the predict pipeline."""

    from simpnmr.app.params.options import PredictRunOptions
    from simpnmr.app.pipelines.predict import run_predict

    config = cfg.PredictConfig.from_file(uargs.input_file)
    options = PredictRunOptions.from_namespace(uargs)

    return run_predict(config, options)


def fit_susc_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for the fit_susc pipeline."""

    from simpnmr.app.params.options import FitSuscRunOptions
    from simpnmr.app.pipelines.fit.susc_fit import run_fit_susc

    config = cfg.FitSuscConfig.from_file(uargs.input_file)
    options = FitSuscRunOptions.from_namespace(uargs)

    return run_fit_susc(config, options)


def fit_corr_time_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for the fit_corr_time pipeline."""

    from simpnmr.app.params.options import FitCorrTimeRunOptions
    from simpnmr.app.pipelines.fit.corr_time_fit import run_fit_corr_time

    config = cfg.FitCorrTimeConfig.from_file(uargs.input_file)
    options = FitCorrTimeRunOptions.from_namespace(uargs)

    return run_fit_corr_time(config, options)


def plot_hfc_iso_ax_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for plot_hfc_iso_ax pipeline."""

    from simpnmr.app.params.options import PlotHFCIsoAxRunOptions
    from simpnmr.app.pipelines.plot.hfc_invar_plot import run_plot_hfc_iso_ax

    config = cfg.PlotHFCConfig.from_file(uargs.input_file)
    options = PlotHFCIsoAxRunOptions.from_namespace(uargs)

    return run_plot_hfc_iso_ax(config, options)


def plot_shift_tdep_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for plot_shift_tdep pipeline."""

    from simpnmr.app.params.options import PlotShiftTdepRunOptions
    from simpnmr.app.pipelines.plot.shift_t_plot import run_plot_shift_tdep

    options = PlotShiftTdepRunOptions.from_namespace(uargs)

    return run_plot_shift_tdep(uargs.experiment_files, options)


def calc_pcs_iso_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for calc_pcs_iso pipeline."""

    from simpnmr.app.params.options import CalcPcsIsoRunOptions
    from simpnmr.app.pipelines.pcs_iso import run_calc_pcs_iso

    options = CalcPcsIsoRunOptions.from_namespace(uargs)

    return run_calc_pcs_iso(
        susc_file=uargs.susc_file,
        temperatures=uargs.temperatures,
        structure_file=uargs.structure_file,
        central_atom=uargs.central_atom,
        options=options,
    )


def calc_pdip_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for calc_pdip pipeline."""

    from simpnmr.app.params.options import CalcPdipRunOptions
    from simpnmr.app.pipelines.pdip import run_calc_pdip

    options = CalcPdipRunOptions.from_namespace(uargs)

    return run_calc_pdip(
        structure_file=uargs.structure_file,
        centres=uargs.centres,
        elements=uargs.elements,
        chem_labels=uargs.chem_labels,
        plot_components=uargs.plot_components,
        options=options,
    )


def extract_hfc_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for extract_hfc pipeline."""

    from simpnmr.app.params.options import ExtractHFCRunOptions
    from simpnmr.app.pipelines.extract.hfc_extract import run_extract_hfc

    options = ExtractHFCRunOptions.from_namespace(uargs)

    return run_extract_hfc(uargs.calculation_data, options)


def plot_hfc_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for plot_hfc pipeline."""

    from simpnmr.app.params.options import PlotHFCRunOptions
    from simpnmr.app.pipelines.plot.hfc_plot import run_plot_hfc

    options = PlotHFCRunOptions.from_namespace(uargs)

    return run_plot_hfc(
        calculation_data=uargs.calculation_data,
        components=uargs.components,
        chem_labels=uargs.chem_labels,
        elements=uargs.elements,
        options=options,
    )


def extract_dia_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for extract_dia pipeline."""

    from simpnmr.app.params.options import ExtractDiaRunOptions
    from simpnmr.app.pipelines.extract.dia_extract import run_extract_dia

    options = ExtractDiaRunOptions.from_namespace(uargs)

    return run_extract_dia(
        output_file=uargs.output_file,
        ref_output_file=uargs.ref_output_file or None,
        options=options,
    )


def get_sh_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for get_sh workflow."""

    from simpnmr.app.params.options import GetSHRunOptions
    from simpnmr.app.pipelines.get_sh import run_get_sh

    options = GetSHRunOptions.from_namespace(uargs)

    return run_get_sh(options)


def calc_tau_c_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Compute rotational correlation time from molecular coordinates."""

    from simpnmr.core.phys.tau_c import compute_tau_r, SOLVENTS

    results = []
    for temp in uargs.temperatures:
        result = compute_tau_r(
            xyz_file=uargs.xyz_file,
            temperature=temp,
            solvent=uargs.solvent,
            eta=uargs.eta,
            method=uargs.method,
            shell=uargs.shell,
            sigma=uargs.sigma,
        )
        results.append((temp, result))

    # Print summary table
    header = f"{'T (K)':>8}  {'η (mPa·s)':>10}  {'τ_R (ps)':>10}  {'D_iso (rad²/s)':>16}  {'Anisotropy':>10}"
    print(header)
    print("-" * len(header))
    for temp, r in results:
        if r is None:
            print(f"{temp:8.1f}  {'—':>10}  {'—':>10}  {'—':>16}  {'—':>10}")
            continue
        eta_mpa = r["eta"] * 1e3
        tau_ps = r["tau_iso"] * 1e12
        d_iso = r["D_iso"]
        aniso = r.get("anisotropy", float("nan"))
        print(
            f"{temp:8.1f}  {eta_mpa:10.4f}  {tau_ps:10.1f}  {d_iso:16.4e}  {aniso:10.3f}"
        )

    if uargs.solvent and not uargs.eta:
        avail = ", ".join(sorted(SOLVENTS.keys()))
        logger.debug("Available solvents: %s", avail)

    return 0


def read_args(arg_list=None):
    """
    Parse CLI arguments and dispatch to the selected subcommand handler.

    Args:
        arg_list (list[str] | None): Optional argument list to parse (used for testing).
            If None, arguments are read from `sys.argv`.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """

    description = """
    A package for fitting susceptibility tensors from paramagnetic NMR data
    """

    epilog = (
        "To display options for a specific program, use\n\n      simpnmr SUBPROGRAM -h"
    )

    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse inputs and initialise the pipeline, but exit before execution",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Show only errors",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"simpnmr {__version__}",
    )

    parser.add_argument(
        "--plot-profile",
        choices=["paper", "poster"],
        metavar="<str>",
        type=str,
        default=os.environ.get("SIMPNMR_PLOT_PROFILE", "paper"),
        help=(
            "Global plotting profile used by all plot-generating workflows.\n"
            " - 'paper': compact typography for manuscripts\n"
            " - 'poster': larger typography for slides/posters\n"
            "You can also set SIMPNMR_PLOT_PROFILE=paper|poster.\n"
            "Default: paper"
        ),
    )
    parser.add_argument(
        "--hide",
        action="store_true",
        help=("Do not display plots interactively (default: show; always saves)."),
    )

    parser.add_argument(
        "--accessibility",
        choices=["default", "colorblind"],
        metavar="<str>",
        type=str,
        default=os.environ.get("SIMPNMR_ACCESSIBILITY", "default"),
        help=(
            "Accessibility mode for plots.\n"
            " - 'default': colour-only encoding (default)\n"
            " - 'colorblind': colour + pattern encoding for shift components\n"
            "You can also set SIMPNMR_ACCESSIBILITY=default|colorblind."
        ),
    )

    parser._positionals.title = "Subprograms"

    subparsers = parser.add_subparsers(dest="prog")

    extract_dia = subparsers.add_parser(
        "extract_dia",
        description="Extract diamagnetic shifts from quantum chemistry output",
    )
    extract_dia.set_defaults(func=extract_dia_cli)

    extract_dia.add_argument(
        "output_file",
        type=str,
        help=("Quantum Chemistry output file containing chemical shift information"),
    )

    extract_dia.add_argument(
        "--ref_output_file",
        metavar="ref_output_file",
        default="",
        type=str,
        help=(
            "Quantum Chemistry output file containing reference "
            "chemical shift information"
        ),
    )

    get_sh = subparsers.add_parser(
        "get_sh",
        description="Derive g-tensor and optional ZFS parameters from chiT regression",
    )
    get_sh.set_defaults(func=get_sh_cli)

    get_sh.add_argument(
        "--spin",
        type=float,
        required=True,
        help="Total spin quantum number (e.g. 2.0)",
    )

    get_sh.add_argument(
        "chiT_regression_csv",
        type=str,
        help="CSV file produced by chiT regression (slope/intercept and uncertainties)",
    )

    fit_susc = subparsers.add_parser(
        "fit_susc",
        description=(
            "Fit susceptibility tensor using DFT hyperfines and experimental peaks"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    fit_susc.set_defaults(func=fit_susc_cli)

    fit_susc.add_argument(
        "input_file",
        type=str,
        help=("Input file for fit_susc -- see documentation for format"),
    )

    fit_susc.add_argument(
        "--pcs_isosurface",
        action="store_true",
        help=("Saves PCS isosurface for each temperature to separate cube files"),
    )

    fit_susc.add_argument(
        "--susc_units",
        "-su",
        choices=["cm3 mol-1", "A3"],
        metavar="<str>",
        type=str,
        default="A3",
        help=('Controls susceptibility units of plots \n(wrap with "")\nDefault: A3'),
    )

    fit_susc.add_argument(
        "--shift_plots",
        choices=["on", "off"],
        metavar="<str>",
        type=str,
        default="on",
        help=(
            "Enable/disable the fitted shifts plot.\n"
            " - 'on': generate the plot (always saved; use --show to display)\n"
            " - 'off': skip\n"
            "Default: on"
        ),
    )

    fit_susc.add_argument(
        "--spread_plots",
        choices=["on", "off"],
        metavar="<str>",
        type=str,
        default="on",
        help=(
            "Enable/disable the shift spread plot.\n"
            " - 'on': generate the plot (always saved; use --show to display)\n"
            " - 'off': skip\n"
            "Default: on"
        ),
    )

    fit_susc.add_argument(
        "--contrib_plots",
        choices=["on", "off"],
        metavar="<str>",
        type=str,
        default="on",
        help=(
            "Enable/disable the shift component plot.\n"
            " - 'on': generate the plot (always saved; use --show to display)\n"
            " - 'off': skip\n"
            "Default: on"
        ),
    )

    fit_susc.add_argument(
        "--isoaxrh_plots",
        choices=["on", "off"],
        metavar="<str>",
        type=str,
        default="on",
        help=(
            "Enable/disable susceptibility component plots (iso/ax/rh vs T).\n"
            " - 'on': generate the plot (always saved; use --show to display)\n"
            " - 'off': skip\n"
            "Default: on"
        ),
    )

    extract_hfc = subparsers.add_parser(
        "extract_hfc",
        description=("Extract A tensor information from quantum chemistry output"),
    )
    extract_hfc.set_defaults(func=extract_hfc_cli)

    extract_hfc.add_argument(
        "calculation_data",
        type=str,
        help=("Gaussian log file, or Orca output or property file"),
    )

    plot_hfc = subparsers.add_parser(
        "plot_hfc",
        description=("Plot A tensor information from quantum chemistry output"),
    )
    plot_hfc.set_defaults(func=plot_hfc_cli)

    plot_hfc.add_argument(
        "calculation_data",
        type=str,
        help=("Gaussian log file, or Orca output or property file"),
    )

    plot_hfc.add_argument(
        "components",
        choices=[
            "iso",
            "xx",
            "xy",
            "xz",
            "yx",
            "yy",
            "yz",
            "zx",
            "zy",
            "zz",
            "dxx",
            "dxy",
            "dxz",
            "dyx",
            "dyy",
            "dyz",
            "dzx",
            "dzy",
            "dzz",
            "ax",
            "rh",
        ],
        nargs="+",
        help=("Component() to plot"),
    )

    plot_hfc.add_argument(
        "--chem_labels", type=str, help=("chemical label file (.csv)")
    )

    plot_hfc.add_argument(
        "--elements",
        type=str,
        nargs="*",
        default="all",
        help=("Elements to include in plot"),
    )

    plot_hfc.add_argument("--save", action="store_true", help="Save plot(s) to file")

    plot_hfc.add_argument(
        "--hide_plots",
        action="store_true",
        help="Do not display plots on screen",
    )

    plot_hfc_iso = subparsers.add_parser(
        "plot_hfc_iso_ax",
        description=("Plot HFC tensor information from quantum chemistry output"),
    )

    plot_hfc_iso.set_defaults(func=plot_hfc_iso_ax_cli)

    plot_hfc_iso.add_argument("input_file", type=str, help=("simpnmr Input file"))

    plot_hfc_iso.add_argument("--save", action="store_true", help=("Save plot to file"))

    plot_hfc_iso.add_argument(
        "--hide_plots",
        action="store_true",
        help="Do not display plots on screen",
    )

    calc_pdip = subparsers.add_parser(
        "calc_pdip",
        description=(
            "Calculate dipolar Hyperfine tensor using point dipole approximation"
        ),
    )
    calc_pdip.set_defaults(func=calc_pdip_cli)

    calc_pdip.add_argument(
        "structure_file",
        type=str,
        help=("File containing molecular structure: .xyz, .log, ORCA .out"),
    )

    calc_pdip.add_argument(
        "centres",
        type=str,
        nargs="+",
        help=("Atomic label (with index number) of paramagnetic centre(s)"),
    )

    calc_pdip.add_argument(
        "--chem_labels", type=str, help=("chemical label file (.csv)")
    )

    calc_pdip.add_argument(
        "--elements",
        type=str,
        nargs="*",
        default="all",
        help=("Elements to include in plot"),
    )

    calc_pdip.add_argument(
        "--plot_components",
        default=[],
        choices=[
            "xx",
            "xy",
            "xz",
            "yx",
            "yy",
            "yz",
            "zx",
            "zy",
            "zz",
            "x",
            "y",
            "z",
            "ax",
            "rh",
        ],
        nargs="+",
        help=("Component(s) to plot"),
    )

    calc_pcs_iso = subparsers.add_parser(
        "calc_pcs_iso",
        description=("Calculates PCS isosurface and saves to .cube file"),
    )
    calc_pcs_iso.set_defaults(func=calc_pcs_iso_cli)

    calc_pcs_iso.add_argument("susc_file", help="File containing susceptibility data")

    calc_pcs_iso.add_argument(
        "temperatures",
        nargs="+",
        help="Temperatures for which to produce pcs isosurface plot",
        type=float,
    )

    calc_pcs_iso.add_argument(
        "structure_file",
        help=(
            "File containing molecular structure"
            "Either Gaussian .log, ORCA .out, or plain old .xyz"
        ),
    )

    calc_pcs_iso.add_argument(
        "central_atom",
        help="Atom on which isosurface is centered. Must include indexing",
    )

    predict = subparsers.add_parser(
        "predict", description="Calculate shifts using Hyperfine and Susceptibility"
    )
    predict.set_defaults(func=predict_cli)

    predict.add_argument(
        "input_file",
        type=str,
        help=("Input file for predict - see documentation for format"),
    )

    predict.add_argument(
        "--susc_units",
        "-su",
        choices=["cm3 mol-1", "A3"],
        metavar="<str>",
        type=str,
        default="A3",
        help=(
            "Controls susceptibility units of plots and output files \n"
            '(wrap with "")\n'
            "Default: A3"
        ),
    )

    plot_shift_tdep = subparsers.add_parser(
        "plot_shift_tdep",
        description="Calculate shifts using Hyperfine and Susceptibility",
    )
    plot_shift_tdep.set_defaults(func=plot_shift_tdep_cli)

    plot_shift_tdep.add_argument(
        "experiment_files", type=str, nargs="+", help=("simpnmr experiment.csv files")
    )

    fit_corr_time = subparsers.add_parser(
        "fit_corr_time", description="Fit correlation times using experimental R1 data"
    )
    fit_corr_time.set_defaults(func=fit_corr_time_cli)

    fit_corr_time.add_argument(
        "input_file",
        type=str,
        help="Input file for fit_corr_time -- see documentation for format",
    )

    calc_tau_c = subparsers.add_parser(
        "calc_tau_c",
        description=(
            "Calculate rotational correlation time τ_R from molecular coordinates "
            "using the Perrin ellipsoid (default) or bead-shell hydrodynamic model"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    calc_tau_c.set_defaults(func=calc_tau_c_cli)

    calc_tau_c.add_argument(
        "xyz_file",
        type=str,
        help="Molecular coordinate file (.xyz or .pdb)",
    )

    calc_tau_c.add_argument(
        "temperatures",
        type=float,
        nargs="+",
        help="Temperature(s) in K",
    )

    calc_tau_c.add_argument(
        "--solvent",
        type=str,
        default=None,
        metavar="<str>",
        help=(
            "Solvent name for viscosity lookup (e.g. D2O, CDCl3, DMSO-d6).\n"
            "Viscosity is corrected to each temperature via Arrhenius scaling.\n"
            "Use --eta to supply a custom viscosity directly."
        ),
    )

    calc_tau_c.add_argument(
        "--eta",
        type=float,
        default=None,
        metavar="<float>",
        help="Solvent viscosity in Pa·s (overrides --solvent)",
    )

    calc_tau_c.add_argument(
        "--method",
        choices=["ellipsoid", "beadshell"],
        default="ellipsoid",
        metavar="<str>",
        help=(
            "Hydrodynamic model to use:\n"
            " - 'ellipsoid': Perrin triaxial ellipsoid (default, fast)\n"
            " - 'beadshell': HYDRONMR-style bead-shell (slower, more accurate)\n"
        ),
    )

    calc_tau_c.add_argument(
        "--shell",
        type=float,
        default=0.0,
        metavar="<float>",
        help="Solvent shell thickness added to vdW radii (Å, default: 0.0)",
    )

    calc_tau_c.add_argument(
        "--sigma",
        type=float,
        default=0.6,
        metavar="<float>",
        help="Minibead radius for bead-shell model (Å, default: 0.6)",
    )

    # Read sub-parser and parse arguments
    parser.set_defaults(func=lambda args, runtime: parser.print_help())
    args = parser.parse_args(arg_list)

    return args


def interface(argv=None):
    args = read_args(argv)

    setup_logging(verbose=args.verbose, quiet=args.quiet, base_dir=os.getcwd())
    logger.info("Working directory: %s", os.getcwd())

    runtime = apply_runtime_settings(
        plot_profile=args.plot_profile,
        show_plots=not args.hide,
        # accessibility=args.accessibility, TODO
    )

    args.runtime = runtime

    try:
        exit_code = args.func(args, runtime)
        logger.info("Job finished successfully")
        raise SystemExit(exit_code or 0)

    except QCError as err:
        logger.error("%s", err)
        raise SystemExit(1) from None

    except yaml.YAMLError as err:
        logger.error("%s", err)
        raise SystemExit(1) from None

    except ValueError as err:
        logger.error("%s", err)
        raise SystemExit(1) from None

    except FileNotFoundError as err:
        logger.error("File not found: %s", err.filename)
        raise SystemExit(1) from None
