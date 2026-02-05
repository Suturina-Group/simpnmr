# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot hyperfine isotropic-to-axial ratios.

Loads hyperfine data, optionally applies chemical labels and averaging, and
generates iso/ax plots for one or more input files.
"""

import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from simpnmr.app.loaders.hfc_load import load_base_molecule_from_hyperfines
from simpnmr.app.loaders.labels_load import load_chem_labels_from_csv
from simpnmr.app.params.options import PlotHFCIsoAxRunOptions
from simpnmr.cfg import config as cfg
from simpnmr.viz.plots.hfc import plot_hyperfine_iso_vs_ax
from simpnmr.viz.style.theme import apply_profile


def run_plot_hfc_iso_ax(
    config: cfg.PlotHFCConfig, options: PlotHFCIsoAxRunOptions
) -> int:
    os.makedirs(config.project_name, exist_ok=True)

    # Build the resolved plotting contract once per run.
    spec = apply_profile(options.runtime.plot_profile)

    symbols = ["x", "o"]
    order: list[int] | None = None

    with spec.context():
        fig, ax = plt.subplots(1, 1)

        hf_files = config.hyperfine_file
        if isinstance(hf_files, str):
            hf_files = [hf_files]

        for i, hf_file in enumerate(hf_files):
            symb = symbols[i % len(symbols)]

            local_cfg = replace(config, hyperfine_file=hf_file)
            base_molecule = load_base_molecule_from_hyperfines(
                config=local_cfg,
                delimiter=options.runtime.csv_delimiter,
            )

            for av in config.hyperfine_average or []:
                base_molecule.average_hyperfine(av)

            if config.chem_labels_file:
                al_to_cl, al_to_cml = load_chem_labels_from_csv(config.chem_labels_file)
                base_molecule.apply_chem_labels(al_to_cl, al_to_cml)

            iso_div_ax = {
                nuc.chem_math_label: nuc.A.iso / (nuc.A.dip[0, 0] + nuc.A.dip[1, 1])
                for nuc in base_molecule.nuclei
            }

            if order is None:
                order = [int(j) for j in np.argsort(list(iso_div_ax.values()))]

            if order is None:
                raise RuntimeError("Internal error: plot order was not initialised.")

            file_head = os.path.splitext(os.path.basename(hf_file))[0]

            plot_hyperfine_iso_vs_ax(
                iso_div_ax,
                order,
                fig=fig,
                ax=ax,
                spec=spec,
                symbol=symb,
                save=True,
                show=options.runtime.show_plots,
                save_name=os.path.join(
                    config.project_name,
                    f"hyperfine_iso_ax_{file_head}",
                ),
                verbose=True,
                window_title=f"Hyperfine data from {hf_file}",
            )

        xlims = ax.get_xlim()
        ax.hlines(0, *xlims, colors="k")
        ax.set_xlim(xlims)

    return 0
