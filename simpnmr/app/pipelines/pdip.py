# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Compute point-dipole hyperfine dipolar tensors.

Loads structural data, evaluates point-dipole A_dip tensors, optionally applies
chemical labels, and writes results to CSV with optional plots.
"""

import logging
import os

import numpy as np

from simpnmr.app.loaders.labels_load import load_chem_labels_from_csv
from simpnmr.app.params.options import CalcPdipRunOptions
from simpnmr.core.domain.mol import Molecule
from simpnmr.io.qc import gateway as rdrs
from simpnmr.tools.coords import xyz_fmt as xyzf
from simpnmr.viz.plots.hfc import plot_hyperfine, plot_hyperfine_spread
from simpnmr.viz.style.theme import apply_profile

logger = logging.getLogger(__name__)


def run_calc_pdip(
    structure_file: str,
    centres: list[str],
    elements: list[str] | str,
    chem_labels: str | None,
    plot_components: list[str],
    options: CalcPdipRunOptions,
) -> int:
    """
    Compute point-dipole hyperfine dipolar tensors for a structure and optionally plot.
    """

    # Normalise centres
    centres = [centre.lower().capitalize() for centre in centres]

    # Load structure
    ext = os.path.splitext(structure_file)[1]
    if ext == ".xyz":
        labels, coords = xyzf.load_xyz(structure_file)
    elif ext in {".log", ".out"}:
        qcs = rdrs.QCStructure.guess_from_file(structure_file)
        labels = qcs.labels
        coords = qcs.coords
    else:
        raise ValueError(f"Unsupported structure file format: {ext}")

    # Create molecule
    molecule = Molecule.from_labels_coords(labels, coords, elements=elements)

    # Calculate point dipole A_dip tensor
    molecule.calc_pdip(centres)

    if chem_labels is not None:
        al_to_cl, al_to_cml = load_chem_labels_from_csv(chem_labels)
        molecule.apply_chem_labels(al_to_cl, al_to_cml)

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

    file_head = os.path.splitext(structure_file)[0]
    file_name = f"point_dipole_A_dip_{file_head}.csv"

    header = (
        "Label, Adip_xx (ppm Å^-3), "
        "Adip_xy (ppm Å^-3), "
        "Adip_xz (ppm Å^-3), "
        "Adip_yy (ppm Å^-3), "
        "Adip_yz (ppm Å^-3), "
        "Adip_zz (ppm Å^-3)"
    )

    np.savetxt(
        file_name,
        out,
        delimiter=options.runtime.csv_delimiter,
        header=header,
        fmt="%s",
    )
    logger.info("Point dipole dipolar tensors saved to %s", file_name)

    # Build the resolved plotting contract once per run.
    spec = apply_profile(options.runtime.plot_profile)

    if plot_components:
        with spec.context():
            plot_hyperfine(
                molecule.nuclei,
                plot_components,
                spec=spec,
                save=True,
                show=options.runtime.show_plots,
                save_name=f"point_dipole_A_dip_{file_head}",
                verbose=True,
                window_title="Point-Dipole Hyperfines",
            )

            if chem_labels is not None:
                plot_hyperfine_spread(
                    molecule.nuclei,
                    plot_components,
                    spec=spec,
                    save=True,
                    show=options.runtime.show_plots,
                    save_name=f"spread_point_dipole_A_dip_{file_head}",
                    verbose=True,
                    window_title="Point-Dipole Hyperfines Spread",
                )

    return 0
