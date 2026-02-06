# QC data containers (DTO, no logic)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class HyperfineData:
    """Passive DTO for hyperfine (A-tensor) data parsed from QC outputs."""

    file_name: str
    labels: npt.NDArray[np.str_]
    coords: npt.NDArray[np.float64]
    a_iso: dict[str, float]
    a_dip: dict[str, npt.NDArray[np.float64]]
    a_units: str
