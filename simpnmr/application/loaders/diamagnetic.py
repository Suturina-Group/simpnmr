"""Load diamagnetic shifts from CSV or QC output.

Reads external data and returns plain mappings for application-level workflows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from simpnmr.io.csv.utils import read_csv_safe
from simpnmr.io.qc import qc_readers as rdrs
from simpnmr.tools.coords_tools import xyz_format as xyzf


def load_diamagnetic_shifts(
    file_name: str,
    file_type: str = "csv",
    ref_file_name: str = "",
    ref_file_type: str = "csv",
) -> tuple[dict[str, float], str, Optional[dict[str, float]]]:
    """Load diamagnetic shifts and optional reference corrections.

    Returns:
        dia_by_key: Mapping key -> dia shift.
        key_kind: 'atom_label' or 'chem_label' (which key the mapping uses).
        ref_avg_by_label_nn: Optional mapping label_nn -> averaged reference shift.
    """
    # --- main dia file ---
    if file_type == "csv":
        dia = read_csv_safe(file_name)
        if "shift" not in dia.columns:
            raise KeyError("Missing required column 'shift' in diamagnetic shift file")

        if "atom_label" in dia.columns:
            key_kind = "atom_label"
            dia_by_key = {
                str(k): float(v) for k, v in zip(dia["atom_label"], dia["shift"])
            }
        elif "chem_label" in dia.columns:
            key_kind = "chem_label"
            dia_by_key = {
                str(k): float(v) for k, v in zip(dia["chem_label"], dia["shift"])
            }
        else:
            raise KeyError(
                "atom_label or chem_label not present in diamagnetic shift file"
            )

    elif file_type == "dft":
        data = rdrs.QCCS.guess_from_file(file_name)

        # QC labels are typically index-free; align with index-bearing labels
        labels = list(data.cs_iso.keys())
        labels_with_idx = xyzf.add_label_indices(labels)

        key_kind = "atom_label"
        dia_by_key = {
            str(lab_idx): float(val)
            for lab_idx, val in zip(labels_with_idx, data.cs_iso.values())
        }

    else:
        raise ValueError("Unknown file_type")

    # --- optional reference file (averaged by label_nn) ---
    ref_avg_by_label_nn: Optional[dict[str, float]] = None

    if len(ref_file_name):
        if ref_file_type == "csv":
            ref = read_csv_safe(ref_file_name)
            if "atom_label" not in ref.columns or "shift" not in ref.columns:
                raise KeyError(
                    "Reference CSV must include 'atom_label' and 'shift' columns"
                )

            # Average by nucleus (remove indices first)
            ref_labels_nn = xyzf.remove_label_indices(
                [str(x) for x in ref["atom_label"].tolist()]
            )
            ref = ref.copy()
            ref["atom_label_nn"] = ref_labels_nn
            grouped = ref.groupby("atom_label_nn")["shift"].mean()
            ref_avg_by_label_nn = {
                str(lab_nn): float(val) for lab_nn, val in grouped.items()
            }

        elif ref_file_type == "dft":
            ref_data = rdrs.QCCS.guess_from_file(ref_file_name)

            ref_labels = list(ref_data.cs_iso.keys())
            ref_labels_nn = xyzf.remove_label_indices(ref_labels)

            avg_ref_iso = dict.fromkeys(ref_labels_nn, 0.0)
            counts = dict.fromkeys(ref_labels_nn, 0)

            for lab, lab_nn in zip(ref_labels, ref_labels_nn):
                avg_ref_iso[lab_nn] += float(ref_data.cs_iso[lab])
                counts[lab_nn] += 1

            for lab_nn in np.unique(ref_labels_nn):
                avg_ref_iso[lab_nn] /= float(counts[lab_nn])

            ref_avg_by_label_nn = {str(k): float(v) for k, v in avg_ref_iso.items()}

        else:
            raise ValueError("Unknown ref_file_type")

    return dia_by_key, key_kind, ref_avg_by_label_nn
