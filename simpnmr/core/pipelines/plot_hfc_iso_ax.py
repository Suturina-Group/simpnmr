import os

import matplotlib.pyplot as plt
import numpy as np

from simpnmr.config import config as cfg
from simpnmr.core import main
from simpnmr.core.pipelines.setup.options import PlotHFCIsoAxRunOptions
from simpnmr.io.qc import qc_readers as rdrs
from simpnmr.tools.coords_tools import xyz_format as xyzf
from simpnmr.viz import visualise as vis


def run_plot_hfc_iso_ax(
    config: cfg.PlotHFCConfig, options: PlotHFCIsoAxRunOptions
) -> int:
    os.makedirs(config.project_name, exist_ok=True)

    symbols = ["x", "o"]
    fig, ax = plt.subplots(1, 1)
    order: list[int] | None = None

    hf_files = config.hyperfine_file
    if isinstance(hf_files, str):
        hf_files = [hf_files]

    for i, hf_file in enumerate(hf_files):
        symb = symbols[i % len(symbols)]

        if config.hyperfine_method == "dft":
            qc_hyperfine_data = rdrs.QCA.guess_from_file(hf_file)
            qc_hyperfine_data.save_to_csv(
                os.path.join(config.project_name, "dft_hyperfines.csv"),
                verbose=True,
                delimiter=options.runtime.csv_delimiter,
                comment=f"# Data taken from file {hf_file}",
            )
            base_molecule = main.Molecule.from_QCA(
                qc_hyperfine_data,
                converter="MHz_to_Ang-3",
                elements=config.nuclei_include,
            )

        elif config.hyperfine_method == "pdip":
            ext = os.path.splitext(hf_file)[1]
            if ext == ".xyz":
                labels, coords = xyzf.load_xyz(hf_file)
            elif ext in {".log", ".out"}:
                qcs = rdrs.QCStructure.guess_from_file(hf_file)
                labels, coords = qcs.labels, qcs.coords
            else:
                raise ValueError(f"Unsupported hyperfine file format: {ext}")

            base_molecule = main.Molecule.from_labels_coords(
                labels, coords, elements=config.nuclei_include
            )
            base_molecule.calc_pdip(config.hyperfine_pdip_centres)
        else:
            raise ValueError(f"Unknown hyperfine_method: {config.hyperfine_method}")

        for av in config.hyperfine_average or []:
            base_molecule.average_hyperfine(av)

        if config.chem_labels_file:
            base_molecule.add_chem_labels_from_file(config.chem_labels_file)

        iso_div_ax = {
            nuc.chem_math_label: nuc.A.iso / (nuc.A.dip[0, 0] + nuc.A.dip[1, 1])
            for nuc in base_molecule.nuclei
        }

        if order is None:
            order = [int(j) for j in np.argsort(list(iso_div_ax.values()))]

        if order is None:
            raise RuntimeError("Internal error: plot order was not initialised.")

        file_head = os.path.splitext(os.path.basename(hf_file))[0]

        vis.plot_hyperfine_iso_vs_ax(
            iso_div_ax,
            order,
            fig=fig,
            ax=ax,
            symbol=symb,
            save=options.save,
            show=False,
            save_name=os.path.join(
                config.project_name,
                f"hyperfine_iso_ax_{file_head}{options.runtime.plot_format}",
            ),
            verbose=True,
            window_title=f"Hyperfine data from {hf_file}",
        )

    xlims = ax.get_xlim()
    ax.hlines(0, *xlims, colors="k")
    ax.set_xlim(xlims)

    if options.show:
        plt.show()

    return 0
