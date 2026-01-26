# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""Runtime option schemas for pipelines.

This module defines lightweight dataclasses and enums that represent runtime
options typically provided by the CLI (as opposed to options stored in YAML
configuration files).

Important:
- This module must be importable without side effects (no env-var reads, no
  matplotlib rcParams changes, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from simpnmr.core.pipelines.setup.plotting import PlotMode


# Shared runtime context passed from CLI into all pipelines
@dataclass(frozen=True)
class RuntimeSettings:
    """Global runtime settings applied/selected by the CLI."""

    csv_delimiter: str = ","
    plot_format: str = ".png"
    echo_r2: bool = False


@dataclass(frozen=True)
class PredictRunOptions:
    """Runtime options for prediction pipelines."""

    runtime: RuntimeSettings
    susc_units: str = "A3"

    @classmethod
    def from_namespace(cls, ns):
        return cls(runtime=ns.runtime, susc_units=ns.susc_units)


@dataclass(frozen=True)
class FitCorrTimeRunOptions:
    """Runtime options for correlation-time fitting pipelines."""

    runtime: RuntimeSettings
    dry_run: bool = False

    @classmethod
    def from_namespace(cls, ns):
        return cls(runtime=ns.runtime, dry_run=ns.dry_run)


@dataclass(frozen=True)
class FitSuscRunOptions:
    runtime: RuntimeSettings
    dry_run: bool = False
    susc_units: str = "A3"
    shift_plots: PlotMode = "save"
    spread_plots: PlotMode = "save"
    contrib_plots: PlotMode = "save"
    isoaxrho_plots: PlotMode = "save"
    pcs_isosurface: bool = False

    @classmethod
    def from_namespace(cls, ns):
        return cls(
            runtime=ns.runtime,
            dry_run=ns.dry_run,
            susc_units=ns.susc_units,
            shift_plots=ns.shift_plots,
            spread_plots=ns.spread_plots,
            contrib_plots=ns.contrib_plots,
            isoaxrho_plots=ns.isoaxrho_plots,
            pcs_isosurface=ns.pcs_isosurface,
        )
