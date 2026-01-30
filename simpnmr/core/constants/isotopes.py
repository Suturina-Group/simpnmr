# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""TODO"""

DEFAULT_ISOTOPES = {
    "H": "1H",
    "C": "13C",
    "P": "31P",
    "N": "15N",
    "Si": "29Si",
    "B": "10B",
    "Li": "6Li",
}

OTHER_ISOTOPES = ["2H"]

SUPPORTED_ISOTOPES = list(DEFAULT_ISOTOPES.values()) + OTHER_ISOTOPES
