# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define physical constants.

Provides commonly used physical constants imported from SciPy.
"""

import scipy.constants as consts

# Physical constants
MU0 = consts.physical_constants["vacuum mag. permeability"][0]  # [N A^-2]
MUB = consts.physical_constants["Bohr magneton"][0]
H = consts.physical_constants["Planck constant"][0]  # [J s]
C = consts.physical_constants["speed of light in vacuum"][0]  # [m s^-1]
KB = consts.physical_constants["Boltzmann constant"][0]  # Boltzmann constant k [J·K⁻¹]
NA = consts.physical_constants["Avogadro constant"][0]  # Avogadro constant [mol^-1]
GE = abs(consts.physical_constants["electron g factor"][0])  # g value of free electron
EGAMMA = consts.physical_constants["electron gyromag. ratio in MHz/T"][0]
