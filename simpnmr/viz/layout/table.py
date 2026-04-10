# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Compact in-plot table renderers for visualization layouts."""

from collections.abc import Sequence

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from simpnmr.viz.style.theme import PlotSpec


def _validate_bbox(bbox: Sequence[float]) -> None:
    """Validate a table bounding box in axes-fraction coordinates."""
    if len(bbox) != 4:
        raise ValueError(
            "bbox must contain exactly four floats: [x0, y0, width, height]."
        )
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError("bbox width and height must be positive.")


def _visible_edges(
    row: int,
    col: int,
    *,
    n_cols: int,
    is_header: bool,
    remove_outer_frame: bool,
) -> str:
    """Return the visible cell edges using the legacy table semantics."""
    edges = ""
    if is_header:
        edges += "B"
    if col < n_cols - 1:
        edges += "R"
    if not remove_outer_frame:
        if row == 0:
            edges += "T"
        if col == 0:
            edges += "L"
        if col == n_cols - 1:
            edges += "R"
    return "".join(dict.fromkeys(edges))


def _draw_cell_edges(
    ax: Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    edges: str,
    color: str,
    linewidth: float,
) -> None:
    """Draw selected cell edges in axes-fraction coordinates."""
    segments = {
        "L": ((x, y), (x, y + height)),
        "R": ((x + width, y), (x + width, y + height)),
        "T": ((x, y + height), (x + width, y + height)),
        "B": ((x, y), (x + width, y)),
    }
    for edge in edges:
        start, end = segments[edge]
        ax.add_line(
            Line2D(
                [start[0], end[0]],
                [start[1], end[1]],
                transform=ax.transAxes,
                color=color,
                linewidth=linewidth,
            )
        )


def render_compact_table(
    ax: Axes,
    blocks: Sequence[tuple[str, Sequence[str]]],
    spec: PlotSpec,
    *,
    col_widths: Sequence[float] | None = None,
    bbox: Sequence[float] | None = None,
    row_height: float | None = None,
    cell_align: str = "center",
    remove_outer_frame: bool = True,
) -> list[list[Rectangle]]:
    """Render a compact block-based table inside an existing axes.

    This helper renders one visual table column per semantic block. Each block
    consists of a column header and a variable-length sequence of body lines.
    Unlike Matplotlib's built-in ``Table``, this renderer computes cell
    rectangles and text placement manually in axes-fraction coordinates so that
    font size, row height, and table bounding box remain independently
    controllable.

    Args:
        ax: Matplotlib axes used as the table container.
        blocks: Ordered table blocks as ``(header, lines)`` pairs.
        spec: Plot styling contract that provides palette and typography.
        col_widths: Optional relative widths for the table columns.
        bbox: Optional table bounding box in axes-fraction coordinates.
        cell_align: Horizontal alignment used for table text.
        remove_outer_frame: Whether to remove the outer table frame.

    Returns:
        A column-major nested list of rectangle patches representing the drawn
        cells.

    Raises:
        ValueError: If ``col_widths`` length does not match the number of
            blocks.
    """
    scale = spec.typography
    palette = spec.palette

    n_cols = len(blocks)
    if col_widths is not None and len(col_widths) != n_cols:
        raise ValueError("col_widths length must match the number of blocks.")

    col_widths = list(col_widths) if col_widths is not None else None
    bbox = list(bbox) if bbox is not None else [0.02, 0.06, 0.96, 0.88]
    _validate_bbox(bbox)

    if n_cols == 0:
        return []

    ordered_blocks = sorted(blocks, key=lambda block: len(block[1]), reverse=True)
    widths = col_widths or [1.0 / n_cols] * n_cols

    width_total = sum(widths)
    widths = [width / width_total for width in widths]

    x0, y0, table_width, table_height = bbox

    if len(widths) != n_cols:
        raise ValueError("col_widths length must match the number of blocks.")
    if width_total <= 0:
        raise ValueError("col_widths must sum to a positive value.")

    n_rows = max(len(lines) for _, lines in ordered_blocks) + 1
    if row_height is None:
        row_height = bbox[3] / n_rows

    body_linewidth = 0.8
    header_linewidth = 0.8

    if cell_align == "left":
        text_x_factor = 0.03
        ha = "left"
    elif cell_align == "right":
        text_x_factor = 0.97
        ha = "right"
    else:
        text_x_factor = 0.5
        ha = "center"

    rendered_columns: list[list[Rectangle]] = []
    x_cursor = x0

    for col, ((header, lines), width_fraction) in enumerate(
        zip(ordered_blocks, widths, strict=False)
    ):
        col_width = table_width * width_fraction
        rendered_cells: list[Rectangle] = []

        for row in range(n_rows):
            y = y0 + table_height - (row + 1) * row_height
            is_header = row == 0
            text = (
                header
                if is_header
                else (lines[row - 1] if row - 1 < len(lines) else "")
            )

            rect = Rectangle(
                (x_cursor, y),
                col_width,
                row_height,
                transform=ax.transAxes,
                facecolor=palette.annotation_bg,
                edgecolor="none",
                linewidth=0.0,
            )

            edges = _visible_edges(
                row,
                col,
                n_cols=n_cols,
                is_header=is_header,
                remove_outer_frame=remove_outer_frame,
            )

            ax.add_patch(rect)
            rendered_cells.append(rect)
            _draw_cell_edges(
                ax,
                x=x_cursor,
                y=y,
                width=col_width,
                height=row_height,
                edges=edges,
                color=palette.primary,
                linewidth=header_linewidth if is_header else body_linewidth,
            )

            ax.text(
                x_cursor + col_width * text_x_factor,
                y + row_height * 0.5,
                text,
                transform=ax.transAxes,
                ha=ha,
                va="center",
                fontsize=scale.annotation,
                fontweight="bold" if is_header else "normal",
            )

        rendered_columns.append(rendered_cells)
        x_cursor += col_width

    return rendered_columns
