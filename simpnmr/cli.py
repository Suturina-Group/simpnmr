# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""SimpNMR command-line interface.

This module defines the CLI entry points used to run SimpNMR workflows such as
pNMR prediction, susceptibility fitting, hyperfine plotting, and data extraction.
"""

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from simpnmr.config import config as cfg
from simpnmr.core import main
from simpnmr.core.pipelines.options import RuntimeSettings
from simpnmr.core.pipelines.settings import apply_runtime_settings
from simpnmr.io.qc import qc_readers as rdrs
from simpnmr.tools.coords_tools import xyz_format as xyzf
from simpnmr.viz import visualise as vis

logger = logging.getLogger(__name__)


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[90m",  # gray
        logging.INFO: "\033[36m",  # cyan
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def format(self, record):
        msg = super().format(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{msg}{self.RESET}"


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.ERROR

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = ColorFormatter("%(levelname)-7s |  %(message)s")
        handler.setFormatter(formatter)
        root.addHandler(handler)


def extract_dia_func(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """
    Extract diamagnetic isotropic shifts and save them to a CSV file.

    If a reference output file is provided, the shifts are referenced by atom type
    (non-indexed labels).

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """
    logger.info("Extracting shifts from %s", uargs.output_file)

    data = rdrs.QCCS.guess_from_file(uargs.output_file)

    if len(uargs.ref_output_file):
        ref_data = rdrs.QCCS.guess_from_file(uargs.ref_output_file)
        logger.info("Extracting reference shifts from %s", uargs.ref_output_file)

        ref_labels = list(ref_data.cs_iso.keys())
        ref_labels_nn = xyzf.remove_label_indices(ref_labels)

        avg_ref_iso = dict.fromkeys(ref_labels_nn, 0)

        for lab, lab_nn in zip(ref_labels, ref_labels_nn):
            avg_ref_iso[lab_nn] += ref_data.cs_iso[lab]

        for lab_nn in np.unique(ref_labels_nn):
            avg_ref_iso[lab_nn] /= ref_labels_nn.count(lab_nn)

        # Subtract from reference shifts based on atom type
        labels = list(data.cs_iso.keys())
        labels_nn = xyzf.remove_label_indices(labels)

        for lab, lab_nn in zip(labels, labels_nn):
            if lab_nn in avg_ref_iso.keys():
                data.cs_iso[lab] = avg_ref_iso[lab_nn] - data.cs_iso[lab]

    # Save diamagnetic shifts to file
    iso_shifts = list(data.cs_iso.values())
    labels = list(data.cs_iso.keys())
    labels = xyzf.remove_label_indices(labels)
    labels = xyzf.add_label_indices(labels)

    out = [f"{label}, {value:.6f}" for label, value in zip(labels, iso_shifts)]

    np.savetxt(
        "extracted_dia.csv",
        out,
        delimiter=runtime.csv_delimiter,
        header="atom_label, shift",
        fmt="%s",
        comments="",
    )

    logger.info("Extracted shifts saved to extracted_dia.csv")

    return 0


def plot_a_func(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """
    Plot hyperfine data from a single quantum-chemistry output file.

    Depending on the flags provided, this generates spread and/or component plots and
    optionally saves the figures to disk.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """
    # Load quantum chemical hyperfine data
    calc_data = rdrs.QCA.guess_from_file(uargs.calculation_data)

    # Create molecule object from quantum chemical hyperfine data
    # to convert units
    molecule = main.Molecule.from_QCA(
        calc_data, converter="MHz_to_Ang-3", elements=uargs.elements
    )

    if uargs.chem_labels is not None:
        molecule.add_chem_labels_from_file(uargs.chem_labels)

    file_head = os.path.splitext(uargs.calculation_data)[0]

    if not (uargs.hide_plots and not uargs.save):
        if uargs.chem_labels is not None:
            vis.plot_hyperfine_spread(
                molecule.nuclei,
                components=uargs.components,
                save=uargs.save,
                show=False,
                save_name=f"hyperfine_spread_{file_head}{runtime.plot_format}",
                window_title=(
                    f"Spread of hyperfine data from {uargs.calculation_data}"
                ),
                verbose=True,
            )

        vis.plot_hyperfine(
            molecule.nuclei,
            components=uargs.components,
            save=uargs.save,
            show=False,
            save_name=f"hyperfine_{file_head}{runtime.plot_format}",
            window_title=f"Hyperfine data from {uargs.calculation_data}",
            verbose=True,
        )

        if not uargs.hide_plots:
            plt.show()

    return 0


def plot_a_iso_ax_func(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """
    Plot isotropic-vs-axial hyperfine ratios for multiple input files.

    This handler loads the base configuration from a YAML file, computes
    A_iso/(A_xx + A_yy) per nucleus, and generates a comparison plot.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """

    config = cfg.PlotAConfig.from_file(uargs.input_file)

    # Make output directory and file
    os.makedirs(config.project_name, exist_ok=True)

    symbols = ["x", "o"]
    fig, ax = plt.subplots(1, 1)

    hf_files = config.hyperfine_file
    if isinstance(hf_files, str):
        hf_files = [hf_files]

    for i, hf_file in enumerate(hf_files):
        symb = symbols[i % len(symbols)]
        # Either load hyperfines from DFT output file
        if config.hyperfine_method == "dft":
            qc_hyperfine_data = rdrs.QCA.guess_from_file(hf_file)
            # Write raw calculation data to output file
            qc_hyperfine_data.save_to_csv(
                os.path.join(config.project_name, "dft_hyperfines.csv"),
                verbose=True,
                delimiter=runtime.csv_delimiter,
                comment=f"# Data taken from file {hf_file}",
            )

            # Create molecule object from quantum chemical hyperfine data
            # Retain only the atoms that are given in the labels file
            base_molecule = main.Molecule.from_QCA(
                qc_hyperfine_data,
                converter="MHz_to_Ang-3",
                elements=config.nuclei_include,
            )

        # generate using point dipole approximation
        elif config.hyperfine_method == "pdip":
            if os.path.splitext(hf_file)[1] == ".xyz":
                labels, coords = xyzf.load_xyz(hf_file)
            elif os.path.splitext(hf_file)[1] in [".log", ".out"]:
                QCS = rdrs.QCStructure.guess_from_file(hf_file)
                labels = QCS.labels
                coords = QCS.coords
            else:
                raise ValueError(
                    "Specified hyperfine file format "
                    f"{os.path.splitext(hf_file)[1]} unsupported"
                )

            # Create molecule
            base_molecule = main.Molecule.from_labels_coords(
                labels, coords, elements=config.nuclei_include
            )

            # Calculate point dipole hyperfine
            base_molecule.calc_pdip(config.hyperfine_pdip_centres)

        if len(config.hyperfine_average):
            for av in config.hyperfine_average:
                base_molecule.average_hyperfine(av)

        if len(config.chem_labels_file):
            base_molecule.add_chem_labels_from_file(config.chem_labels_file)

        file_head = os.path.splitext(hf_file)[0]

        iso_div_ax = {
            nuc.chem_math_label: nuc.A.iso / (nuc.A.dip[0, 0] + nuc.A.dip[1, 1])
            for nuc in base_molecule.nuclei
        }

        if symb == "x":
            order = np.argsort(list(iso_div_ax.values()))

        if not (uargs.hide_plots and not uargs.save):
            vis.plot_hyperfine_iso_vs_ax(
                iso_div_ax,
                order,
                fig=fig,
                ax=ax,
                symbol=symb,
                save=uargs.save,
                show=False,
                save_name=f"hyperfine_iso_ax_{file_head}{runtime.plot_format}",
                verbose=True,
                window_title=f"Hyperfine data from {hf_file}",
            )

    xlims = ax.get_xlim()

    ax.hlines(0, *xlims, colors="k")

    ax.set_xlim(xlims)
    plt.show()
    return 0


def extract_a_func(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """
    Extract hyperfine data from a quantum-chemistry output file and write a CSV.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """
    # Load quantum chemical hyperfine data
    calc_data = rdrs.QCA.guess_from_file(uargs.calculation_data)

    # Create molecule object from quantum chemical hyperfine data
    # to convert units
    base = main.Molecule.from_QCA(calc_data, converter="MHz_to_Ang-3")

    base.to_csv(
        "hyperfine_{}.csv".format(uargs.calculation_data),
        verbose=True,
        delimiter=runtime.csv_delimiter,
    )

    return 0


def calc_pdip_func(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """
    Compute point-dipole hyperfine dipolar tensors for a structure and optionally plot.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """

    # Parse user specified centres
    centres = [centre.lower().capitalize() for centre in uargs.centres]

    if os.path.splitext(uargs.structure_file)[1] == ".xyz":
        labels, coords = xyzf.load_xyz(uargs.structure_file)
    elif os.path.splitext(uargs.structure_file)[1] in [".log", ".out"]:
        QCS = rdrs.QCStructure.guess_from_file(uargs.structure_file)
        labels = QCS.labels
        coords = QCS.coords
    else:
        raise ValueError(
            "Specified hyperfine file format "
            f"{os.path.splitext(uargs.structure_file)[1]} unsupported"
        )

    # Create molecule
    molecule = main.Molecule.from_labels_coords(labels, coords, elements=uargs.elements)

    # Calculate point dipole A_dip tensor
    molecule.calc_pdip(centres)

    if uargs.chem_labels is not None:
        molecule.add_chem_labels_from_file(uargs.chem_labels)

    # Save hyperfine data to file
    out = np.array(
        [
            "{}, {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(
                nuc.label,
                nuc.chem_label,
                *nuc.A.dip[0, :],
                *nuc.A.dip[1, 1:],
                nuc.A.dip[2, 2],
            )
            for nuc in molecule.nuclei
        ]
    )

    # Save to file
    file_head = os.path.splitext(uargs.structure_file)[0]
    file_name = f"point_dipole_A_dip_{file_head}.csv"

    header = (
        "Label, Adip_xx (ppm Å^-3), "
        "Adip_xy (ppm Å^-3), "
        "Adip_xz (ppm Å^-3), "
        "Adip_yy (ppm Å^-3), "
        "Adip_yz (ppm Å^-3), "
        "Adip_zz (ppm Å^-3)"
    )

    np.savetxt(file_name, out, delimiter=runtime.csv_delimiter, header=header, fmt="%s")
    logger.info("Point dipole dipolar tensors saved to %s", file_name)

    if len(uargs.plot_components):
        vis.plot_hyperfine(
            molecule.nuclei,
            uargs.plot_components,
            save=True,
            show=False,
            save_name=f"point_dipole_A_dip_{file_head}{runtime.plot_format}",
            verbose=True,
            window_title="Point-Dipole Hyperfines",
        )

        if uargs.chem_labels is not None:
            vis.plot_hyperfine_spread(
                molecule.nuclei,
                uargs.plot_components,
                save=True,
                show=False,
                save_name=f"spread_point_dipole_A_dip_{file_head}{runtime.plot_format}",
                verbose=True,
                window_title="Point-Dipole Hyperfines Spread",
            )

        plt.show()

    return 0


def calc_pcs_iso_func(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """
    Generate PCS isosurfaces for selected temperatures using an isotropic chi tensor.

    The script loads a structure, reads susceptibility data (ORCA or CSV formats),
    and writes a cube file for each requested temperature.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """

    if os.path.splitext(uargs.structure_file)[1] == ".xyz":
        labels, coords = xyzf.load_xyz(uargs.structure_file)
    elif os.path.splitext(uargs.structure_file)[1] in [".log", ".out"]:
        QCS = rdrs.QCStructure.guess_from_file(uargs.structure_file)
        labels = QCS.labels
        coords = QCS.coords
    else:
        raise ValueError(
            "Specified structure file format "
            f"{os.path.splitext(uargs.structure_file)[1]} unsupported"
        )

    if uargs.central_atom not in labels:
        raise ValueError(
            "Specified central atom not present in structure file \n"
            "Perhaps try with indexing e.g. Ni1"
        )

    # Load susceptibility information
    if "orca" in uargs.susc_format:
        suscs = main.Susceptibility.from_orca(
            uargs.susc_file, section=uargs.susc_format.split("orca_")[1]
        )
    elif "csv" in uargs.susc_format:
        suscs = main.Susceptibility.from_csv(uargs.susc_file)
    elif "molcas" in uargs.susc_format:
        raise ValueError("Molcas files are not currently supported")

    for susc in suscs:
        if susc.temperature in uargs.temperatures:
            # Calculate irreducible representations of susceptibility tensor
            susc.calc_irred()

            # Generate and save PCS isosurface
            susc.save_pcs_isosurface(
                labels,
                coords,
                uargs.central_atom,
                comment=(
                    f"PCS Isosurface from {uargs.susc_file} at {susc.temperature:.2f} K"
                ),
                file_name=f"pcs_isosurface_{susc.temperature:.2f}_K.cube",
            )

    return 0


def fit_susc_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for the fit_susc pipeline."""

    from simpnmr.core.pipelines.fit_susc import run_fit_susc
    from simpnmr.core.pipelines.options import FitSuscRunOptions

    config = cfg.FitSuscConfig.from_file(uargs.input_file)
    options = FitSuscRunOptions.from_namespace(uargs)

    return run_fit_susc(config, options)


def predict_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for the predict pipeline."""

    from simpnmr.core.pipelines.options import PredictRunOptions
    from simpnmr.core.pipelines.predict import run_predict

    config = cfg.PredictConfig.from_file(uargs.input_file)
    options = PredictRunOptions.from_namespace(uargs)

    return run_predict(config, options)


def fit_corr_time_cli(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """Thin CLI wrapper for the fit_corr_time pipeline."""

    from simpnmr.core.pipelines.fit_corr_time import run_fit_corr_time
    from simpnmr.core.pipelines.options import FitCorrTimeRunOptions

    config = cfg.FitCorrTimeConfig.from_file(uargs.input_file)
    options = FitCorrTimeRunOptions.from_namespace(uargs)

    return run_fit_corr_time(config, options)


def plot_shift_tdep_func(uargs: argparse.Namespace, runtime: RuntimeSettings) -> int:
    """
    Plot temperature dependence of chemical shifts from experimental datasets.

    This CLI handler loads one or more experimental CSV files and generates
    shift-versus-temperature and shift-versus-inverse-temperature plots.

    Args:
        uargs (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """

    experiments = main.Experiment.from_file(uargs.experiment_files)

    vis.plot_shift_tdep(
        experiments,
        "ShiftT_vs_T",
        show=True,
        save=True,
        save_name=f"shift_x_T_vs_T{runtime.plot_format}",
    )

    vis.plot_shift_tdep(
        experiments,
        "Shift_vs_1/T",
        show=True,
        save=True,
        save_name=f"shift_vs_T-1{runtime.plot_format}",
    )

    return 0


def read_args(arg_list=None):
    """
    Parse CLI arguments and dispatch to the selected subcommand handler.

    Args:
        arg_list (list[str] | None): Optional argument list to parse (used for testing).
            If None, arguments are read from `sys.argv`.

    Returns:
        argparse.Namespace: Parsed arguments object
        (after `args.func(args)` is executed).
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

    parser._positionals.title = "Subprograms"

    subparsers = parser.add_subparsers(dest="prog")

    extract_dia = subparsers.add_parser(
        "extract_dia",
        description="Extract diamagnetic shifts from quantum chemistry output",
    )
    extract_dia.set_defaults(func=extract_dia_func)

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
        choices=["on", "show", "save", "off"],
        metavar="<str>",
        type=str,
        default="save",
        help=(
            "Plot Experimental and Calculated Chemical shifts "
            "against each other \n"
            " - 'on' shows and saves the plots\n"
            " - 'show' shows the plots\n"
            " - 'save' saves the plots\n"
            " - 'off' neither shows nor saves\n"
            "Default: save"
        ),
    )

    fit_susc.add_argument(
        "--spread_plots",
        choices=["on", "show", "save", "off"],
        metavar="<str>",
        type=str,
        default="save",
        help=(
            "Plot spread of contributions to calculated shifts \n"
            " - 'on' shows and saves the plots\n"
            " - 'show' shows the plots\n"
            " - 'save' saves the plots\n"
            " - 'off' neither shows nor saves\n"
            "Default: save"
        ),
    )

    fit_susc.add_argument(
        "--contrib_plots",
        choices=["on", "show", "save", "off"],
        metavar="<str>",
        type=str,
        default="save",
        help=(
            "Plot mean of contributions to mean calculated shifts \n"
            " - 'on' shows and saves the plots\n"
            " - 'show' shows the plots\n"
            " - 'save' saves the plots\n"
            " - 'off' neither shows nor saves\n"
            "Default: on"
        ),
    )

    fit_susc.add_argument(
        "--isoaxrho_plots",
        choices=["on", "show", "save", "off"],
        metavar="<str>",
        type=str,
        default="save",
        help=(
            "Plot Isotropic, Axial, and Rhombic susceptibility "
            "as a function of temperature \n"
            " - 'on' shows and saves the plots\n"
            " - 'show' shows the plots\n"
            " - 'save' saves the plots\n"
            " - 'off' neither shows nor saves\n"
            "Default: save"
        ),
    )

    extract_a = subparsers.add_parser(
        "extract_a",
        description=("Extract A tensor information from quantum chemistry output"),
    )
    extract_a.set_defaults(func=extract_a_func)

    extract_a.add_argument(
        "calculation_data",
        type=str,
        help=("Gaussian log file, or Orca output or property file"),
    )

    plot_a = subparsers.add_parser(
        "plot_a",
        description=("Plot A tensor information from quantum chemistry output"),
    )
    plot_a.set_defaults(func=plot_a_func)

    plot_a.add_argument(
        "calculation_data",
        type=str,
        help=("Gaussian log file, or Orca output or property file"),
    )

    plot_a.add_argument(
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
            "rho",
        ],
        nargs="+",
        help=("Component() to plot"),
    )

    plot_a.add_argument("--chem_labels", type=str, help=("chemical label file (.csv)"))

    plot_a.add_argument(
        "--elements",
        type=str,
        nargs="*",
        default="all",
        help=("Elements to include in plot"),
    )

    plot_a.add_argument("--save", action="store_false", help=("Save plot to file"))

    plot_a.add_argument(
        "--hide_plots", action="store_true", help=("Display plot on screen")
    )

    plot_a_iso = subparsers.add_parser(
        "plot_a_iso_ax",
        description=("Plot A tensor information from quantum chemistry output"),
    )

    plot_a_iso.set_defaults(func=plot_a_iso_ax_func)

    plot_a_iso.add_argument("input_file", type=str, help=("simpnmr Input file"))

    plot_a_iso.add_argument("--save", action="store_true", help=("Save plot to file"))

    plot_a_iso.add_argument(
        "--hide_plots", action="store_true", help=("Display plot on screen")
    )

    calc_pdip = subparsers.add_parser(
        "calc_pdip",
        description=(
            "Calculate dipolar Hyperfine tensor using point dipole approximation"
        ),
    )
    calc_pdip.set_defaults(func=calc_pdip_func)

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
            "rho",
        ],
        nargs="+",
        help=("Component(s) to plot"),
    )

    calc_pcs_iso = subparsers.add_parser(
        "calc_pcs_iso",
        description=("Calculates PCS isosurface and saves to .cube file"),
    )
    calc_pcs_iso.set_defaults(func=calc_pcs_iso_func)

    calc_pcs_iso.add_argument("susc_file", help="File containing susceptibility data")

    calc_pcs_iso.add_argument(
        "susc_format",
        help="Susceptibility file format",
        choices=["csv", "orca_nev", "orca_cas", "molcas"],
    )

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
    plot_shift_tdep.set_defaults(func=plot_shift_tdep_func)

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

    # Read sub-parser and parse arguments
    parser.set_defaults(func=lambda args, runtime: parser.print_help())
    args = parser.parse_args(arg_list)

    return args


def interface(argv=None):
    args = read_args(argv)
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    runtime = apply_runtime_settings()
    args.runtime = runtime

    try:
        raise SystemExit(args.func(args, runtime))
    except ValueError as err:
        logger.error("%s", err)
        raise SystemExit(1) from None
    except FileNotFoundError as err:
        logger.error("File not found: %s", err.filename)
        raise SystemExit(1) from None
