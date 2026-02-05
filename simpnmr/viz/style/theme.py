# simpnmr/viz/style/theme.py
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import matplotlib as mpl
from matplotlib.axes import Axes

from simpnmr.app.params.plot_cfg import PlotProfile
from simpnmr.viz.style import glyphs, legend, palette, patterns, typography
from simpnmr.viz.style.patterns import AccessibilityMode


@dataclass(frozen=True)
class PlotSpec:
    """Resolved plotting style contract for a run.

    This object is created once (in a pipeline) from a PlotProfile and then
    passed down to plotting functions.
    """

    profile: PlotProfile
    typography: typography.TypographyScale
    glyphs: glyphs.GlyphScale
    legend: legend.LegendStyle
    palette: palette.Palette
    shift_colours: palette.ShiftColours
    shift_patterns: patterns.ShiftPatterns | None = None

    accessibility: AccessibilityMode = "default"

    @contextmanager
    def context(self) -> Generator[None, None, None]:
        """Apply this plotting profile within an isolated Matplotlib rc-context.

        Use this to avoid global rcParams leakage across tests/plots.

        Example:
            spec = apply_profile(runtime.plot_profile,
            accessibility=runtime.accessibility)
            with spec.context():
                fig, ax = plt.subplots()
                ...
        """
        with mpl.rc_context():
            mpl.rcdefaults()
            typography.apply_global_typography(self.profile)
            legend.apply_global_legend_style(self.profile)
            yield

    def skin_axes(self, ax: Axes):
        """Apply per-Axes typography tweaks for the current profile.

        This is intentionally a thin wrapper so plot modules do not import
        `simpnmr.viz.style.typography` directly.

        Args:
            ax: Matplotlib Axes to style.

        Returns:
            TypographyScale used for styling.
        """
        return typography.apply_typography(ax, size=self.profile)


def build_spec(
    profile: PlotProfile,
    accessibility: AccessibilityMode = "default",
) -> PlotSpec:
    """Build a PlotSpec for a profile without applying any Matplotlib side effects."""
    return PlotSpec(
        profile=profile,
        accessibility=accessibility,
        typography=typography.get_scale(profile),
        glyphs=glyphs.get_glyphs(profile),
        legend=legend.get_legend_style(profile),
        palette=palette.get_palette(profile),
        shift_colours=palette.get_shift_colours(profile),
        shift_patterns=(
            patterns.get_shift_patterns(accessibility, profile)
            if accessibility == "colorblind"
            else None
        ),
    )


def apply_profile(
    profile: PlotProfile,
    accessibility: AccessibilityMode = "default",
) -> PlotSpec:
    """Apply global Matplotlib styling for the selected plotting profile.

    This function must be called once per pipeline run, before any figures
    are created.
    """
    spec = build_spec(profile, accessibility=accessibility)

    # Apply global Matplotlib styling for deterministic output.
    mpl.rcdefaults()
    typography.apply_global_typography(profile)
    legend.apply_global_legend_style(profile)

    return spec
