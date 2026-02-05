# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define global plot styling presets.

Provides theme-level configuration helpers for consistent Matplotlib styling.
"""

from __future__ import annotations

from typing import Literal

SizeClass = Literal["small", "standard", "large"]

DEFAULT_SIZE: SizeClass = "standard"
