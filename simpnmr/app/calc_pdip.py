import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from simpnmr.app.setup.options import CalcPdipRunOptions
from simpnmr.core.domain.molecule import Molecule
from simpnmr.io.qc import qc_readers as rdrs
from simpnmr.tools.coords_tools import xyz_format as xyzf
from simpnmr.viz import visualise

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
        molecule.add_chem_labels_from_file(chem_labels)

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

    if plot_components:
        visualise.plot_hyperfine(
            molecule.nuclei,
            plot_components,
            save=options.save,
            show=options.show,
            save_name=f"point_dipole_A_dip_{file_head}{options.runtime.plot_format}",
            verbose=True,
            window_title="Point-Dipole Hyperfines",
        )

        if chem_labels is not None:
            visualise.plot_hyperfine_spread(
                molecule.nuclei,
                plot_components,
                save=options.save,
                show=options.show,
                save_name=(
                    f"spread_point_dipole_A_dip_{file_head}"
                    f"{options.runtime.plot_format}"
                ),
                verbose=True,
                window_title="Point-Dipole Hyperfines Spread",
            )

        if options.show:
            plt.show()

    return 0
