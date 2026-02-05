# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define runtime option schemas for CLI-driven workflows.

Provides lightweight dataclasses representing runtime parameters passed from
the CLI into application pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

from simpnmr.app.params.plot_cfg import PlotMode, PlotProfile


# Shared runtime context passed from CLI into all pipelines
@dataclass(frozen=True)
class RuntimeSettings:
    """Global runtime settings applied/selected by the CLI."""

    csv_delimiter: str = ","
    echo_r2: bool = False
    plot_profile: PlotProfile = "paper"
    accessibility: str = "default"
    show_plots: bool = False


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
    shift_plots: PlotMode = "on"
    spread_plots: PlotMode = "on"
    contrib_plots: PlotMode = "on"
    isoaxrho_plots: PlotMode = "on"
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


@dataclass(frozen=True)
class PlotShiftTdepRunOptions:
    runtime: RuntimeSettings
    show: bool = True
    save: bool = True

    @classmethod
    def from_namespace(cls, ns):
        return cls(
            runtime=ns.runtime,
            show=True,
            save=True,
        )


@dataclass(frozen=True)
class CalcPcsIsoRunOptions:
    runtime: RuntimeSettings

    @classmethod
    def from_namespace(cls, ns):
        return cls(runtime=ns.runtime)


@dataclass(frozen=True)
class CalcPdipRunOptions:
    """Runtime options for point-dipole PCS calculation."""

    runtime: RuntimeSettings
    plot_mode: PlotMode = "on"
    plot_components: list[str] | None = None

    @classmethod
    def from_namespace(cls, ns):
        return cls(
            runtime=ns.runtime,
            plot_mode=getattr(ns, "plot_mode", "on"),
            plot_components=getattr(ns, "plot_components", None),
        )


@dataclass(frozen=True)
class ExtractHFCRunOptions:
    runtime: RuntimeSettings

    @classmethod
    def from_namespace(cls, ns):
        return cls(runtime=ns.runtime)


@dataclass(frozen=True)
class PlotHFCIsoAxRunOptions:
    runtime: RuntimeSettings
    save: bool
    show: bool

    @classmethod
    def from_namespace(cls, ns) -> "PlotHFCIsoAxRunOptions":
        return cls(
            runtime=ns.runtime,
            save=bool(getattr(ns, "save", False)),
            show=not bool(getattr(ns, "hide_plots", False)),
        )


@dataclass(frozen=True)
class PlotHFCRunOptions:
    runtime: RuntimeSettings
    save: bool
    show: bool

    @classmethod
    def from_namespace(cls, ns) -> "PlotHFCRunOptions":
        return cls(
            runtime=ns.runtime,
            save=bool(getattr(ns, "save", False)),
            show=not bool(getattr(ns, "hide_plots", False)),
        )


@dataclass(frozen=True)
class ExtractDiaRunOptions:
    """Run options for the extract_dia CLI-driven pipeline."""

    runtime: RuntimeSettings

    @classmethod
    def from_namespace(cls, ns) -> "ExtractDiaRunOptions":
        return cls(runtime=ns.runtime)


@dataclass(frozen=True)
class GetSHRunOptions:
    """Run options for the get_sh CLI-driven workflow."""

    runtime: RuntimeSettings
    chiT_regression_csv: str
    spin: float

    @classmethod
    def from_namespace(cls, ns) -> "GetSHRunOptions":
        return cls(
            runtime=ns.runtime,
            chiT_regression_csv=ns.chiT_regression_csv,
            spin=float(ns.spin),
        )
