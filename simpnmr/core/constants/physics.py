# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""TODO"""

import scipy.constants as consts

# Physical constants
MU0 = consts.physical_constants["vacuum mag. permeability"][0]  # [N A^-2]
MUB = consts.physical_constants["Bohr magneton"][0]
HBAR = consts.hbar  # [J s radian-1]
H = consts.h  # [J s radian-1]
KB = consts.physical_constants["Boltzmann constant"][0]  # Boltzmann constant k [J·K⁻¹]
GE = abs(consts.physical_constants["electron g factor"][0])  # g value of free electron
EGAMMA = consts.physical_constants["electron gyromag. ratio in MHz/T"][0]
