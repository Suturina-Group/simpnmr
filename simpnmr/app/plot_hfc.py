import os

import matplotlib.pyplot as plt

from simpnmr.app.setup.options import PlotHFCRunOptions
from simpnmr.core.domain.molecule import Molecule
from simpnmr.io.qc import qc_readers as rdrs
from simpnmr.viz import visualise as vis


def run_plot_hfc(
    calculation_data: str,
    components: list[str],
    chem_labels: str | None,
    elements: list[str] | str,
    options: PlotHFCRunOptions,
) -> int:
    """Plot hyperfine data from a single QC output file."""

    calc_data = rdrs.QCA.guess_from_file(calculation_data)

    molecule = Molecule.from_QCA(
        calc_data,
        converter="MHz_to_Ang-3",
        elements=elements,
    )

    if chem_labels is not None:
        molecule.add_chem_labels_from_file(chem_labels)

    file_head = os.path.splitext(os.path.basename(calculation_data))[0]

    if not (not options.show and not options.save):
        if chem_labels is not None:
            vis.plot_hyperfine_spread(
                molecule.nuclei,
                components=components,
                save=options.save,
                show=False,
                save_name=f"hyperfine_spread_{file_head}{options.runtime.plot_format}",
                window_title=f"Spread of hyperfine data from {calculation_data}",
                verbose=True,
            )

        vis.plot_hyperfine(
            molecule.nuclei,
            components=components,
            save=options.save,
            show=False,
            save_name=f"hyperfine_{file_head}{options.runtime.plot_format}",
            window_title=f"Hyperfine data from {calculation_data}",
            verbose=True,
        )

        if options.show:
            plt.show()

    return 0
