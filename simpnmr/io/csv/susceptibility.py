# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import scipy.constants as constants

from simpnmr.io.csv.readers import read_csv_safe


def read_susceptibilities_csv(file_name: str) -> List[Tuple[np.ndarray, float]]:
    data = read_csv_safe(file_name)

    # Forward conversion, A^3 --> Key (same mapping as old domain code)
    convs = {
        "(A^3)": 1.0,
        "(Å^3 mol^-1)": constants.Avogadro,
        "(A^3 mol^-1)": constants.Avogadro,
        "(cm^3)": 1e-24,
        "(cm^3 mol^-1)": 1e-24 * constants.Avogadro / (4 * np.pi),
    }

    renamer = {}
    for name in data.keys():
        for unit, factor in convs.items():
            if unit in name:
                data[name] /= factor
                renamer[name] = name.replace(unit, "(Å^3)")

    if renamer:
        data.rename(renamer, inplace=True, axis=1)

    out: List[Tuple[np.ndarray, float]] = []
    for _, row in data.iterrows():
        tensor = np.array(
            [
                [row["chi_xx (Å^3)"], row["chi_xy (Å^3)"], row["chi_xz (Å^3)"]],
                [row["chi_xy (Å^3)"], row["chi_yy (Å^3)"], row["chi_yz (Å^3)"]],
                [row["chi_xz (Å^3)"], row["chi_yz (Å^3)"], row["chi_zz (Å^3)"]],
            ],
            dtype=float,
        )
        out.append((tensor, float(row["Temperature (K)"])))

    return out
