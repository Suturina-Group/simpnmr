# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define plot mode options and helpers.

Provides utilities to map plot modes to show/save flags used by pipelines.
"""

from __future__ import annotations

from typing import Literal

PlotMode = Literal["on", "off"]

PlotProfile = Literal["paper", "poster"]
