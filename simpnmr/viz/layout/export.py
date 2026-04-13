# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Export and finalize Matplotlib figures.

Provides helpers to save figures as PDF files and to handle show/save/close
logic for visualization workflows.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib.figure

_PREVIEW_SCALE = 2.0


def save_figure_pdf(
    fig: matplotlib.figure.Figure,
    save_name: str | Path,
    *,
    bbox_inches: str | None = None,
    pad_inches: float = 0.02,
    transparent: bool = False,
    facecolor: str = "white",
    metadata: dict[str, str] | None = None,
    close: bool = False,
) -> Path:
    """Export a Matplotlib figure to PDF with standardized settings.

    This enforces a single export policy across all viz modules:
    - Always writes a .pdf file (any provided extension is ignored).
    - Always creates output directories.
    - Uses consistent savefig kwargs suitable for publications.

    Args:
        fig: Matplotlib figure to export.
        save_name: Output path or base name. If an extension is provided
            (e.g., ".png"), it will be replaced with ".pdf".
        bbox_inches: Matplotlib savefig bbox setting. Default: "None".
        pad_inches: Padding (inches) around the tight bounding box.
        transparent: Whether to export with transparent background.
        facecolor: Figure facecolor for export (commonly "white").
        metadata: Optional PDF metadata dict.
        close: If True, closes the figure after saving to avoid memory growth.

    Returns:
        The resolved Path to the written PDF.

    Raises:
        ValueError: If save_name resolves to a directory or empty path.
        OSError: If the output directory cannot be created.
    """

    out = Path(save_name)
    if str(out).strip() == "":
        raise ValueError("save_name must be a non-empty path or filename.")

    # If user passes a directory, refuse: saving requires a filename.
    # (We check both 'endswith separator' style and actual existing dirs.)
    if str(save_name).endswith(("/", "\\")) or out.exists() and out.is_dir():
        raise ValueError(
            f"save_name must be a file path, got a directory-like value: {save_name!r}"
        )

    # Enforce .pdf suffix by appending it (do NOT strip existing suffixes).
    # Contract: if user passes `file.png`, the output will be `file.png.pdf`.
    if not str(out).endswith(".pdf"):
        out = Path(f"{out}.pdf")

    # Ensure output directory exists.
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)

    savefig_kwargs: dict[str, Any] = {
        "format": "pdf",
        "bbox_inches": bbox_inches,
        "pad_inches": pad_inches,
        "transparent": transparent,
        "facecolor": facecolor,
    }
    if metadata is not None:
        savefig_kwargs["metadata"] = metadata

    fig.savefig(out, **savefig_kwargs)

    if close:
        # Avoid memory growth in batch plotting runs.
        import matplotlib.pyplot as plt

        plt.close(fig)

    return out


def _save_figure_pickle(
    fig: matplotlib.figure.Figure,
    save_name: str | Path,
) -> None:
    """Save a Matplotlib figure as a pickle file for later interactive use.

    The pickle can be reopened with::

        import pickle, matplotlib.pyplot as plt
        fig = pickle.load(open("figure.pkl", "rb"))
        plt.show()
    """
    out = Path(f"{save_name}.pkl")
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "wb") as f:
            pickle.dump(fig, f)
    except Exception:
        if out.exists():
            out.unlink()
        return


def render_figure(
    fig: matplotlib.figure.Figure,
    *,
    save: bool,
    show: bool,
    save_name: str | Path | None = None,
) -> None:
    """Finalize a Matplotlib figure according to viz policy.

    This function centralizes figure lifecycle handling:
    - Saving (via ``export_pdf``) if requested
    - Showing the figure interactively if requested
    - Closing the figure automatically in non-interactive (batch) mode

    Args:
        fig: Matplotlib figure to finalize.
        save: Whether to save the figure.
        show: Whether to display the figure interactively.
        save_name: Base name or path used for saving. Required if ``save`` is True.
    """
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided when save=True")
        save_figure_pdf(fig, save_name)
        _save_figure_pickle(fig, save_name)

    if show:
        import matplotlib.pyplot as plt

        if _PREVIEW_SCALE != 1.0:
            fig.set_dpi(fig.get_dpi() * _PREVIEW_SCALE)
        plt.show()
    else:
        # Batch / non-interactive mode: always close to avoid memory growth
        import matplotlib.pyplot as plt

        plt.close(fig)
