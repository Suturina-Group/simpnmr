# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Extract diamagnetic isotropic shifts from QC output.

Reads QC results, optionally applies atom-type referencing, and writes
diamagnetic shifts to CSV.
"""

import logging

import numpy as np

from simpnmr.app.params.options import ExtractDiaRunOptions
from simpnmr.io.qc import gateway as rdrs
from simpnmr.tools.coords import xyz_fmt as xyzf

logger = logging.getLogger(__name__)


def run_extract_dia(
    output_file: str,
    ref_output_file: str | None,
    options: ExtractDiaRunOptions,
) -> int:
    """Extract diamagnetic isotropic shifts and save them to a CSV file.

    If a reference output file is provided, the shifts are referenced by atom type
    (non-indexed labels).
    """
    logger.info("Extracting shifts from %s", output_file)
    data = rdrs.QCCS.guess_from_file(output_file)

    if ref_output_file:
        ref_data = rdrs.QCCS.guess_from_file(ref_output_file)
        logger.info("Extracting reference shifts from %s", ref_output_file)

        ref_labels = list(ref_data.cs_iso.keys())
        ref_labels_nn = xyzf.remove_label_indices(ref_labels)

        avg_ref_iso: dict[str, float] = dict.fromkeys(ref_labels_nn, 0.0)

        for lab, lab_nn in zip(ref_labels, ref_labels_nn):
            avg_ref_iso[lab_nn] += ref_data.cs_iso[lab]

        for lab_nn in np.unique(ref_labels_nn):
            avg_ref_iso[lab_nn] /= ref_labels_nn.count(lab_nn)

        labels = list(data.cs_iso.keys())
        labels_nn = xyzf.remove_label_indices(labels)

        for lab, lab_nn in zip(labels, labels_nn):
            if lab_nn in avg_ref_iso:
                data.cs_iso[lab] = avg_ref_iso[lab_nn] - data.cs_iso[lab]

    iso_shifts = list(data.cs_iso.values())
    labels = list(data.cs_iso.keys())
    labels = xyzf.remove_label_indices(labels)
    labels = xyzf.add_label_indices(labels)

    out = [f"{label}, {value:.6f}" for label, value in zip(labels, iso_shifts)]

    np.savetxt(
        "extracted_dia.csv",
        out,
        delimiter=options.runtime.csv_delimiter,
        header="atom_label, shift",
        fmt="%s",
        comments="",
    )

    logger.info("Extracted shifts saved to extracted_dia.csv")
    return 0
