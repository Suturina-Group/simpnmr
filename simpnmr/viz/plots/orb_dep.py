# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot orbital shift contributions against distance from the paramagnetic centre."""

from __future__ import annotations

import logging

import matplotlib.ticker as ticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from simpnmr.core.domain.mol import Molecule
from simpnmr.viz.layout.canvas import create_canvas
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.style.theme import PlotSpec
from simpnmr.viz.utils.fmt import isotope_format

logger = logging.getLogger(__name__)


def _distance_to_paramagnetic_centre(
    coord: np.ndarray,
    paramagnetic_centre: np.ndarray,
) -> float:
    """Return the Euclidean distance from a nucleus to the paramagnetic centre.

    Args:
        coord: Nuclear Cartesian coordinates.
        paramagnetic_centre: Paramagnetic-centre Cartesian coordinates.

    Returns:
        Distance between the two points in the native coordinate units.
    """
    return float(np.linalg.norm(coord - paramagnetic_centre))


def _build_orbital_distance_rows(
    molecule: Molecule,
    *,
    order: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract, average, and sort orbital-shift plotting rows by label.

    Args:
        molecule: Molecule containing nuclei with precomputed orbital shifts.
        order: Distance sort direction. Supported values are ``"ascending"``
            and ``"descending"``.

    Returns:
        Tuple containing labels, mean distances, mean orbital isotropic
        contributions, mean orbital anisotropic contributions, and mean total
        orbital contributions.

    Raises:
        ValueError: If the order is unsupported, the paramagnetic centre is not
            available, or no nuclei with orbital data are found.
    """
    if order not in {"ascending", "descending"}:
        raise ValueError(
            "order must be either 'ascending' or 'descending' for orbital "
            "distance-dependence plotting."
        )

    if molecule.paramagnetic_centre is None:
        raise ValueError(
            "Cannot plot orbital shift distance dependence without a "
            "paramagnetic centre."
        )

    paramagnetic_centre = np.asarray(molecule.paramagnetic_centre, dtype=float)
    grouped_rows: dict[str, list[tuple[float, float, float, float]]] = {}

    for nuc in molecule.nuclei:
        if nuc.coord is None or nuc.shift is None:
            continue

        orb_iso = nuc.shift.orb_iso
        orb_aniso = nuc.shift.orb_aniso
        if orb_iso is None or orb_aniso is None:
            continue

        coord = np.asarray(nuc.coord, dtype=float)
        distance = _distance_to_paramagnetic_centre(coord, paramagnetic_centre)
        grouped_rows.setdefault(nuc.chem_math_label, []).append(
            (distance, float(orb_iso), float(orb_aniso), float(orb_iso + orb_aniso))
        )

    if not grouped_rows:
        raise ValueError(
            "No nuclei with orbital shift data are available for orbital "
            "distance-dependence plotting."
        )

    rows = []
    for label, values in grouped_rows.items():
        distances = np.asarray([value[0] for value in values], dtype=float)
        orb_iso_values = np.asarray([value[1] for value in values], dtype=float)
        orb_aniso_values = np.asarray([value[2] for value in values], dtype=float)
        total_values = np.asarray([value[3] for value in values], dtype=float)
        rows.append(
            (
                label,
                float(np.mean(distances)),
                float(np.mean(orb_iso_values)),
                float(np.mean(orb_aniso_values)),
                float(np.mean(total_values)),
            )
        )

    rows.sort(key=lambda row: row[1], reverse=(order == "descending"))

    labels = [row[0] for row in rows]
    distances = np.asarray([row[1] for row in rows], dtype=float)
    orb_iso = np.asarray([row[2] for row in rows], dtype=float)
    orb_aniso = np.asarray([row[3] for row in rows], dtype=float)
    total = np.asarray([row[4] for row in rows], dtype=float)
    return labels, distances, orb_iso, orb_aniso, total


def plot_orbital_shift_distance_dependence(
    molecule: Molecule,
    spec: PlotSpec,
    save: bool = False,
    show: bool = True,
    save_name: str | None = None,
    verbose: bool = False,
    window_title: str | None = None,
    order: str = "ascending",
) -> tuple[Figure, Axes]:
    """Plot orbital isotropic and anisotropic shift contributions by nucleus.

    The plot orders nuclei by their distance from the paramagnetic centre and
    renders side-by-side bars for ``orb_iso`` and ``orb_aniso``.

    Args:
        molecule: Molecule containing nuclei with precomputed orbital shifts.
        spec: Plot styling and layout specification.
        save: Whether to save the rendered figure.
        show: Whether to show the rendered figure.
        save_name: Output stem/path for figure export.
        verbose: Whether to emit verbose export logging.
        window_title: Optional figure window title.
        order: Distance sort direction. Supported values are ``"ascending"``
            and ``"descending"``.

    Returns:
        Tuple of the created figure and axis.
    """
    labels, distances, orb_iso, orb_aniso, total = _build_orbital_distance_rows(
        molecule,
        order=order,
    )

    fig, ax = create_canvas(
        profile=spec.profile,
        window_title=window_title,
    )

    xvals = np.arange(len(labels), dtype=float)
    xcentres = xvals + 0.5
    glyphs = spec.glyphs
    scale = spec.skin_axes(ax)
    palette = spec.palette
    bar_width = 0.38

    ax.bar(
        xcentres - bar_width / 2,
        orb_iso,
        width=bar_width,
        label=r"$\delta^{\mathrm{ORB}}_{\mathrm{iso}}$",
        color=palette.auxiliary,
        alpha=1.0,
        edgecolor=palette.auxiliary,
        linewidth=glyphs.aux_lw,
    )
    ax.bar(
        xcentres + bar_width / 2,
        orb_aniso,
        width=bar_width,
        label=r"$\delta^{\mathrm{ORB}}_{\mathrm{aniso}}$",
        color=palette.highlight,
        alpha=1.0,
        edgecolor=palette.highlight,
        linewidth=glyphs.aux_lw,
    )
    ax.plot(
        xcentres,
        total,
        label="Total",
        color=palette.primary,
        lw=0,
        marker="x",
        markersize=glyphs.ms,
    )

    ax.hlines(
        0.0,
        -0.5,
        len(labels),
        color=palette.primary,
        lw=glyphs.line_lw,
        zorder=0,
    )

    tick_labels = [
        f"{label}\n{distance:.2f} Å" for label, distance in zip(labels, distances)
    ]
    ax.set_xticks(xcentres)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_xlabel("Mean distance to PM centre (Å)")
    if np.unique([nuc.isotope for nuc in molecule.nuclei]).size == 1:
        ax.set_ylabel(
            r"{} $\delta^{{\mathrm{{ORB}}}}$ (ppm.)".format(
                isotope_format(molecule.nuclei[0].isotope)
            )
        )
    else:
        ax.set_ylabel(r"$\delta^{\mathrm{ORB}}$ (ppm.)")

    yvals = np.concatenate((orb_iso, orb_aniso, total))
    ymin = float(np.min(yvals))
    ymax = float(np.max(yvals))
    yrange = ymax - ymin
    ypad = 0.08 * yrange if yrange > 0.0 else 0.5
    ax.set_ylim([ymin - ypad, ymax + ypad])

    ax.set_xlim([-0.5, xvals[-1] + 1.5])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(axis="x", ls="--", which="minor")
    ax.xaxis.set_tick_params("major", length=0)
    ax.tick_params(axis="x", labelsize=scale.annotation)

    ax.legend(loc="best")

    if verbose:
        logger.info(
            "Prepared orbital distance-dependence plot for %d label groups.",
            len(labels),
        )

    render_figure(
        fig,
        show=show,
        save=save,
        save_name=save_name,
    )
    return fig, ax
