# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""TODO"""

import datetime
import logging

from simpnmr.__version__ import __version__
from simpnmr.mappers import dataframes as ser

logger = logging.getLogger(__name__)


def save_molecule_to_csv(
    molecule,
    file_name: str = "molecule.csv",
    verbose: bool = True,
    comment: str = "",
    delimiter: str = ",",
) -> None:
    """Save molecule structure, hyperfine data, and shifts to a CSV file.

    Args:
        file_name: Output CSV file name.
        verbose: If True, prints the output file path.
        comment: Optional additional comment line (including comment marker).
        delimiter: CSV delimiter.
    """

    df = ser.build_molecule_df(molecule)

    _comment = (
        f"# This file was generated with SimpNMR v{__version__} at {{}}\n".format(
            datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
        )
    )

    _comment += comment + "\n"

    with open(file_name, "w") as _f:
        _f.write(_comment)

        df.to_csv(_f, sep=delimiter, header=True, float_format="%.5f", index=None)

    if verbose:
        logger.info("Molecule data written to %s", file_name)

    return
