# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot hyperfine tensor components from QC output.

Loads molecular data, optionally applies chemical labels, and generates
hyperfine plots with optional saving and display.
"""

import os

from simpnmr.app.loaders.labels_load import load_chem_labels_from_csv
from simpnmr.app.loaders.mol_load import load_molecule_from_qca
from simpnmr.app.params.options import PlotHFCRunOptions
from simpnmr.viz.plots.hfc import plot_hyperfine, plot_hyperfine_spread
from simpnmr.viz.style.theme import apply_profile


def run_plot_hfc(
    calculation_data: str,
    components: list[str],
    chem_labels: str | None,
    elements: list[str] | str,
    options: PlotHFCRunOptions,
) -> int:
    """Plot hyperfine data from a single QC output file."""

    molecule = load_molecule_from_qca(
        calculation_data,
        elements=elements,
        converter="MHz_to_Ang-3",
    )

    if chem_labels is not None:
        al_to_cl, al_to_cml = load_chem_labels_from_csv(chem_labels)
        molecule.apply_chem_labels(al_to_cl, al_to_cml)

    file_head = os.path.splitext(os.path.basename(calculation_data))[0]

    # Build the resolved plotting contract once per run.
    spec = apply_profile(options.runtime.plot_profile)

    with spec.context():
        if chem_labels is not None:
            plot_hyperfine_spread(
                molecule.nuclei,
                components=components,
                spec=spec,
                save=True,
                show=options.runtime.show_plots,
                save_name=f"hyperfine_spread_{file_head}",
                window_title=f"Spread of hyperfine data from {calculation_data}",
                verbose=True,
            )

        plot_hyperfine(
            molecule.nuclei,
            components=components,
            spec=spec,
            save=True,
            show=options.runtime.show_plots,
            save_name=f"hyperfine_{file_head}",
            window_title=f"Hyperfine data from {calculation_data}",
            verbose=True,
        )

    return 0
