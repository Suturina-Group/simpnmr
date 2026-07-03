# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Shared obstacle-aware label placement utilities for scatter plots."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

LabelEntry = tuple[str, float, float]

LabelState = tuple[
    float,
    tuple[float, float],
    str,
    str,
    plt.matplotlib.transforms.Bbox,
]

_LABEL_LAYOUT_WEIGHT_LABEL_OVERLAP = 2000.0
_LABEL_LAYOUT_WEIGHT_POINT_OVERLAP = 1500.0
_LABEL_LAYOUT_WEIGHT_LINE_OVERLAP = 1200.0
_LABEL_LAYOUT_WEIGHT_OUT_OF_BOUNDS = 400.0
_LABEL_LAYOUT_WEIGHT_OFFSET_DISTANCE = 0.08


def _create_label_texts(
    ax: plt.Axes,
    entries: Sequence[LabelEntry],
    *,
    fontsize: float,
    colors: dict[str, str] | None = None,
) -> list[plt.Text]:
    """Create annotation artists in an unresolved state for later placement.

    This step deliberately separates artist creation from layout resolution.
    Labels must exist as Matplotlib text artists before the solver can measure
    their rendered screen-space extents via the backend renderer. At this stage
    every label is anchored at its data position but still carries a zero offset;
    the actual collision-aware offset is assigned later by the placement pass.

    Args:
        ax: Target axes that will own the annotation artists.
        entries: Label definitions as ``(label, x, y)`` tuples in data space.
        fontsize: Rendered font size used later for bbox measurement.
        colors: Optional mapping from label string to color.  When provided,
            each annotation is rendered in its label's color.

    Returns:
        Text artists ready to be measured and repositioned by the solver.
    """
    texts = []
    for label, x, y in entries:
        kwargs: dict = {}
        if colors is not None:
            color = colors.get(label)
            if color is not None:
                kwargs["color"] = color
        texts.append(
            ax.annotate(
                label,
                (x, y),
                xytext=(0.0, 0.0),
                textcoords="offset points",
                fontsize=fontsize,
                ha="left",
                va="bottom",
                **kwargs,
            )
        )
    return texts


def _build_safe_bbox(
    ax: plt.Axes,
    *,
    renderer: plt.matplotlib.backend_bases.RendererBase,
    axes_inner_margin_px: float,
) -> plt.matplotlib.transforms.Bbox:
    """Construct the drawable screen-space region available to labels.

    The solver works entirely in display coordinates. Rather than letting labels
    hug the exact axes frame, it shrinks the raw axes bbox by a small inner
    margin and treats the result as the admissible placement region. Candidates
    extending outside this safe box are not forbidden outright, but they are
    penalized by the scorer.

    Args:
        ax: Axes whose visible rectangle defines the layout domain.
        renderer: Active Matplotlib renderer used to measure window extents.
        axes_inner_margin_px: Inset applied on each side in display pixels.

    Returns:
        The inner placement region used by the scorer.
    """
    axes_bbox = ax.get_window_extent(renderer=renderer)
    return plt.matplotlib.transforms.Bbox.from_extents(
        axes_bbox.x0 + axes_inner_margin_px,
        axes_bbox.y0 + axes_inner_margin_px,
        axes_bbox.x1 - axes_inner_margin_px,
        axes_bbox.y1 - axes_inner_margin_px,
    )


def _build_anchor_points(
    ax: plt.Axes,
    entries: Sequence[LabelEntry],
) -> np.ndarray:
    """Transform label anchor coordinates from data space to display space.

    Candidate offsets are expressed in screen-space points, and all collision
    checks are performed against rendered geometry. The data-coordinate anchor
    positions are therefore transformed once up front so point obstacles and
    crowding heuristics can be evaluated in the same coordinate system as label
    bounding boxes.

    Args:
        ax: Axes providing the data-to-display transform.
        entries: Label definitions as ``(label, x, y)`` tuples in data space.

    Returns:
        An ``(N, 2)`` array of anchor coordinates in display space.
    """
    return np.array(
        [ax.transData.transform((x, y)) for _, x, y in entries],
        dtype=float,
    )


def _build_point_obstacles(
    anchor_points: np.ndarray,
    *,
    point_pad_px: float,
) -> list[plt.matplotlib.transforms.Bbox]:
    """Approximate plotted markers by padded screen-space rectangles.

    The solver does not query the true rendered marker outline. Instead, each
    anchor point is expanded into a small axis-aligned bbox that acts as a
    collision obstacle. This approximation is intentionally simple, stable, and
    cheap, while still preventing labels from sitting directly on top of the
    plotted markers.

    Args:
        anchor_points: Label anchors already transformed to display space.
        point_pad_px: Half-size of the marker obstacle box in display pixels.

    Returns:
        One padded bbox per anchor point, used as point obstacles by the scorer.
    """
    return [
        plt.matplotlib.transforms.Bbox.from_extents(
            px - point_pad_px,
            py - point_pad_px,
            px + point_pad_px,
            py + point_pad_px,
        )
        for px, py in anchor_points
    ]


def _build_line_obstacles(
    ax: plt.Axes,
    diag_line: plt.Line2D | None,
    *,
    diag_pad_px: float,
) -> list[plt.matplotlib.transforms.Bbox]:
    """Approximate the optional reference diagonal by sampled obstacle boxes.

    Rather than performing exact geometry tests against the rendered line, the
    algorithm samples points along the diagonal in data space, transforms them
    to display space, and expands each sample into a small bbox. The resulting
    chain of obstacle boxes gives the solver a lightweight representation of the
    diagonal that is sufficient for practical label avoidance.

    Args:
        ax: Axes providing the data-to-display transform.
        diag_line: Optional line artist representing the reference diagonal.
        diag_pad_px: Half-size of each sampled line obstacle in display pixels.

    Returns:
        A list of sampled obstacle boxes. If no usable diagonal is present, an
        empty list is returned.
    """
    line_obstacles: list[plt.matplotlib.transforms.Bbox] = []
    if diag_line is None:
        return line_obstacles

    xdata = np.asarray(diag_line.get_xdata(), dtype=float)
    ydata = np.asarray(diag_line.get_ydata(), dtype=float)
    if xdata.size < 2 or ydata.size < 2:
        return line_obstacles

    samples = np.linspace(0.0, 1.0, 48)
    xs = xdata[0] + samples * (xdata[-1] - xdata[0])
    ys = ydata[0] + samples * (ydata[-1] - ydata[0])
    diag_points = ax.transData.transform(np.column_stack([xs, ys]))
    return [
        plt.matplotlib.transforms.Bbox.from_extents(
            px - diag_pad_px,
            py - diag_pad_px,
            px + diag_pad_px,
            py + diag_pad_px,
        )
        for px, py in diag_points
    ]


def _generate_angles(angle_step_deg: int) -> list[int]:
    """Generate a uniform angular sampling policy for one ring.

    The candidate search space is modeled as several concentric rings around the
    anchor point. Each ring uses the same angular step, so this helper defines
    the discrete directions that will be tested on that ring. The solver's
    behavior therefore stays transparent: for every radius, it simply probes the
    full circle with a fixed angular resolution.

    Args:
        angle_step_deg: Angular increment in degrees.

    Returns:
        A list of angles covering ``[0, 360)`` at the requested step.
    """
    return list(range(0, 360, angle_step_deg))


def _angles_to_offsets(
    radius: float,
    angles_deg: Sequence[int],
) -> list[tuple[float, float]]:
    """Convert one sampled ring into concrete screen-space candidate offsets.

    After a ring radius and its angular sampling have been chosen, this helper
    maps the polar representation into Cartesian ``(dx, dy)`` offsets. These
    offsets are the actual candidate label positions later scored against other
    labels, markers, the diagonal, and the axes bounds.

    Args:
        radius: Ring radius in text-offset points.
        angles_deg: Angular samples in degrees for that ring.

    Returns:
        Candidate offsets on the requested ring in screen-space points.
    """
    return [
        (
            radius * float(np.cos(np.deg2rad(angle_deg))),
            radius * float(np.sin(np.deg2rad(angle_deg))),
        )
        for angle_deg in angles_deg
    ]


def _build_candidate_offsets(
    *,
    offset_inner: float,
    offset_near: float,
    offset_mid: float,
    offset_far: float,
    offset_outer: float,
) -> list[tuple[int, float, float]]:
    """Build the tiered candidate search space used by the placement solver.

    The solver searches on concentric rings around each anchor point. Earlier
    rings represent more desirable placements because they keep the label closer
    to its data point; later rings act as fallback tiers when nearby positions
    are blocked. Within each ring, candidates are sampled uniformly over angle.

    Each returned candidate is encoded as ``(tier_index, dx, dy)`` so the scorer
    can prefer earlier rings even when multiple positions are otherwise similar.

    Args:
        offset_inner: Radius of the closest candidate ring.
        offset_near: Radius of the second candidate ring.
        offset_mid: Radius of the third candidate ring.
        offset_far: Radius of the fourth candidate ring.
        offset_outer: Radius of the final fallback ring.

    Returns:
        The flattened candidate search space consumed by the scorer.
    """
    angle_step_deg = 10
    ring_specs = [
        ("inner", offset_inner),
        ("near", offset_near),
        ("mid", offset_mid),
        ("far", offset_far),
        ("outer", offset_outer),
    ]

    angles_deg = _generate_angles(angle_step_deg)
    return [
        (tier_index, dx, dy)
        for tier_index, (_, radius) in enumerate(ring_specs)
        for dx, dy in _angles_to_offsets(radius, angles_deg)
    ]


def _build_placement_order(anchor_points: np.ndarray) -> list[int]:
    """Choose a greedy placement order based on local crowding.

    The solver is greedy: once a label is placed, subsequent labels must avoid
    it. Because of that, the order matters. This helper approximates crowding by
    each anchor's nearest-neighbour distance in display space and places the
    most crowded labels first. The intent is to reserve scarce space early and
    leave easier, more isolated labels for later.

    Args:
        anchor_points: Anchor coordinates in display space.

    Returns:
        Label indices sorted from most crowded to least crowded.
    """
    if len(anchor_points) == 1:
        return [0]

    diffs = anchor_points[:, None, :] - anchor_points[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1)
    return list(np.argsort(nearest))


def _score_candidate(
    *,
    text: plt.Text,
    dx: float,
    dy: float,
    renderer: plt.matplotlib.backend_bases.RendererBase,
    label_bbox_pad_px: float,
    placed_bboxes: Sequence[plt.matplotlib.transforms.Bbox],
    point_obstacles: Sequence[plt.matplotlib.transforms.Bbox],
    line_obstacles: Sequence[plt.matplotlib.transforms.Bbox],
    safe_bbox: plt.matplotlib.transforms.Bbox,
    tier_index: int,
) -> tuple[LabelState, int, int, int, float]:
    """Score one concrete candidate offset against all active obstacles.

    This is the core local evaluation primitive. For the proposed ``(dx, dy)``
    offset, the helper applies the corresponding text alignment, measures the
    rendered text bbox, pads it, and computes penalties for five effects:

    - overlap with already placed labels,
    - overlap with marker obstacles,
    - overlap with diagonal obstacles,
    - overflow outside the safe axes region,
    - use of a later fallback tier.

    The function returns both the assembled state and the raw overlap counts so
    the caller can implement early-exit heuristics when a clean first-tier
    position is found.

    Args:
        text: Text artist being evaluated.
        dx: Proposed x-offset in text-offset points.
        dy: Proposed y-offset in text-offset points.
        renderer: Active Matplotlib renderer used for bbox measurement.
        label_bbox_pad_px: Additional padding applied around the rendered bbox.
        placed_bboxes: Bboxes of labels already accepted by the greedy solver.
        point_obstacles: Screen-space marker obstacle boxes.
        line_obstacles: Screen-space diagonal obstacle boxes.
        safe_bbox: Admissible screen-space placement region.
        tier_index: Candidate ring index, with smaller values preferred.

    Returns:
        A tuple containing the full candidate state plus raw counts for label,
        point, and line overlaps, and the out-of-bounds amount.
    """
    eps = 1e-9
    ha = "left" if dx > eps else ("right" if dx < -eps else "center")
    va = "bottom" if dy > eps else ("top" if dy < -eps else "center")

    text.set_position((dx, dy))
    text.set_ha(ha)
    text.set_va(va)

    bbox = text.get_window_extent(renderer=renderer)
    bbox = plt.matplotlib.transforms.Bbox.from_extents(
        bbox.x0 - label_bbox_pad_px,
        bbox.y0 - label_bbox_pad_px,
        bbox.x1 + label_bbox_pad_px,
        bbox.y1 + label_bbox_pad_px,
    )

    label_overlap_count = sum(bbox.overlaps(prev) for prev in placed_bboxes)
    point_overlap_count = sum(bbox.overlaps(obs) for obs in point_obstacles)
    line_overlap_count = sum(bbox.overlaps(obs) for obs in line_obstacles)

    out_of_bounds = (
        max(safe_bbox.x0 - bbox.x0, 0.0)
        + max(bbox.x1 - safe_bbox.x1, 0.0)
        + max(safe_bbox.y0 - bbox.y0, 0.0)
        + max(bbox.y1 - safe_bbox.y1, 0.0)
    )

    score = 0.0
    score += _LABEL_LAYOUT_WEIGHT_LABEL_OVERLAP * label_overlap_count
    score += _LABEL_LAYOUT_WEIGHT_POINT_OVERLAP * point_overlap_count
    score += _LABEL_LAYOUT_WEIGHT_LINE_OVERLAP * line_overlap_count
    score += _LABEL_LAYOUT_WEIGHT_OUT_OF_BOUNDS * out_of_bounds
    score += _LABEL_LAYOUT_WEIGHT_OFFSET_DISTANCE * float(tier_index)

    return (
        (score, (dx, dy), ha, va, bbox),
        label_overlap_count,
        point_overlap_count,
        line_overlap_count,
        out_of_bounds,
    )


def _choose_best_state(
    *,
    text: plt.Text,
    candidate_offsets: Sequence[tuple[int, float, float]],
    renderer: plt.matplotlib.backend_bases.RendererBase,
    label_bbox_pad_px: float,
    placed_bboxes: Sequence[plt.matplotlib.transforms.Bbox],
    point_obstacles: Sequence[plt.matplotlib.transforms.Bbox],
    line_obstacles: Sequence[plt.matplotlib.transforms.Bbox],
    safe_bbox: plt.matplotlib.transforms.Bbox,
) -> LabelState:
    """Select the least-penalized candidate state for one label.

    This helper iterates through the full candidate search space for a single
    label, scores every candidate, and keeps the best state encountered so far.
    Because candidates are ordered from more desirable rings to less desirable
    ones, the helper can stop early when it finds a first-tier placement with no
    collisions and no boundary overflow.

    In other words, this function is the deterministic local decision maker used
    by both the initial greedy pass and the local conflict-cluster relayout.

    Args:
        text: Text artist being placed.
        candidate_offsets: Flattened tiered search space as ``(tier, dx, dy)``.
        renderer: Active Matplotlib renderer used for bbox measurement.
        label_bbox_pad_px: Padding around the rendered label bbox.
        placed_bboxes: Bboxes already fixed in the current pass.
        point_obstacles: Screen-space marker obstacle boxes.
        line_obstacles: Screen-space diagonal obstacle boxes.
        safe_bbox: Admissible placement region in display space.

    Returns:
        The best candidate state found for this label.
    """
    best_state: LabelState | None = None

    for tier_index, dx, dy in candidate_offsets:
        (
            state,
            label_overlap_count,
            point_overlap_count,
            line_overlap_count,
            out_of_bounds,
        ) = _score_candidate(
            text=text,
            dx=dx,
            dy=dy,
            renderer=renderer,
            label_bbox_pad_px=label_bbox_pad_px,
            placed_bboxes=placed_bboxes,
            point_obstacles=point_obstacles,
            line_obstacles=line_obstacles,
            safe_bbox=safe_bbox,
            tier_index=tier_index,
        )
        if best_state is None or state[0] < best_state[0]:
            best_state = state

        if (
            label_overlap_count == 0
            and point_overlap_count == 0
            and line_overlap_count == 0
            and out_of_bounds == 0.0
            and tier_index == 0
        ):
            break

    assert best_state is not None
    return best_state


def _apply_state(text: plt.Text, state: LabelState) -> None:
    """Apply a previously chosen screen-space placement state to a label.

    Candidate evaluation mutates the text artist temporarily many times while
    searching. Once a final state has been selected, this helper reapplies the
    winning offset and alignment so the artist reflects the accepted solution.

    Args:
        text: Text artist to update.
        state: Accepted placement state produced by the solver.
    """
    _, (dx, dy), ha, va, _ = state
    text.set_position((dx, dy))
    text.set_ha(ha)
    text.set_va(va)


def _solve_local_cluster(
    *,
    label_texts: Sequence[plt.Text],
    cluster_indices: Sequence[int],
    fixed_bboxes: Sequence[plt.matplotlib.transforms.Bbox],
    forced_first_idx: int,
    candidate_offsets: Sequence[tuple[int, float, float]],
    renderer: plt.matplotlib.backend_bases.RendererBase,
    label_bbox_pad_px: float,
    point_obstacles: Sequence[plt.matplotlib.transforms.Bbox],
    line_obstacles: Sequence[plt.matplotlib.transforms.Bbox],
    safe_bbox: plt.matplotlib.transforms.Bbox,
) -> (
    tuple[
        list[
            tuple[
                float,
                tuple[float, float],
                str,
                str,
                plt.matplotlib.transforms.Bbox,
            ]
        ],
        float,
        list[int],
    ]
    | None
):
    """Re-solve a small conflict cluster with one label forced to go first.

    The main placement pass is greedy, so a label that appears late in the
    ordering may inherit a bad local configuration from earlier decisions.
    This helper performs a limited local repair: it keeps all labels outside
    the cluster fixed, re-solves only the cluster labels, and gives one
    problematic label priority by placing it first in the local order.

    This is not a global optimization step. It is a bounded, local
    re-evaluation designed to repair obvious greedy failures without turning
    the overall algorithm into a large opaque search procedure.

    Args:
        label_texts: All label artists managed by the solver.
        cluster_indices: Indices of labels belonging to the local cluster.
        fixed_bboxes: Already accepted label bboxes outside the cluster.
        forced_first_idx: Cluster label that should be placed first.
        candidate_offsets: Flattened tiered search space as ``(tier, dx, dy)``.
        renderer: Active Matplotlib renderer used for bbox measurement.
        label_bbox_pad_px: Padding around the rendered label bbox.
        point_obstacles: Screen-space marker obstacle boxes.
        line_obstacles: Screen-space diagonal obstacle boxes.
        safe_bbox: Admissible placement region in display space.

    Returns:
        ``None`` if no local solution can be formed; otherwise a triple of
        ``(local_states, total_score, local_order)`` describing the repaired
        local configuration.
    """
    remaining = [idx for idx in cluster_indices if idx != forced_first_idx]
    local_order = [forced_first_idx] + remaining
    local_bboxes: list[plt.matplotlib.transforms.Bbox] = list(fixed_bboxes)
    local_states: list[
        tuple[
            float,
            tuple[float, float],
            str,
            str,
            plt.matplotlib.transforms.Bbox,
        ]
    ] = []
    total_score = 0.0

    for local_idx in local_order:
        local_text = label_texts[local_idx]
        local_best_state = _choose_best_state(
            text=local_text,
            candidate_offsets=candidate_offsets,
            renderer=renderer,
            label_bbox_pad_px=label_bbox_pad_px,
            placed_bboxes=local_bboxes,
            point_obstacles=point_obstacles,
            line_obstacles=line_obstacles,
            safe_bbox=safe_bbox,
        )

        local_states.append(local_best_state)
        local_bboxes.append(local_best_state[4])
        total_score += local_best_state[0]

    return local_states, total_score, local_order


def resolve_label_layout(
    ax: plt.Axes,
    entries: Sequence[LabelEntry],
    *,
    fontsize: float,
    marker_size: float,
    diag_line: plt.Line2D | None = None,
    colors: dict[str, str] | None = None,
) -> list[plt.Text]:
    """Resolve scatter-label placement with a greedy pass plus local repair.

    Algorithm overview
    ------------------
    1. Create all label artists first so Matplotlib can report rendered text
       extents through the active backend renderer.
    2. Convert every relevant geometric object into display coordinates:
       anchor points, marker obstacles, optional diagonal obstacles, and the
       safe axes region.
    3. Build a tiered search space of concentric offset rings around each anchor.
       Earlier rings keep labels closer to their points; later rings are
       fallback positions, and each ring is sampled uniformly over angle.
    4. Order labels from most crowded to least crowded using nearest-neighbour
       spacing in display space.
    5. Run a greedy placement pass: each label independently selects the best
       candidate state while avoiding all labels already fixed before it.
    6. Detect labels that still ended up in a severe conflict state and perform
       a local repair pass. For each problematic label, re-solve only its small
       conflict cluster while forcing the problematic label to be placed first.
       Accept the repair only if the cluster score improves.

    The solver is intentionally heuristic rather than globally optimal. The
    design goal is a deterministic, understandable, and inexpensive placement
    engine whose decisions can be reasoned about in terms of explicit obstacles,
    explicit candidate rings, and explicit scoring weights.

    Args:
        ax: Target axes whose geometry is already final.
        entries: Normalized ``(label, x, y)`` tuples in data coordinates.
        fontsize: Font size used for label rendering and bbox measurement.
        marker_size: Marker size used to approximate point obstacles.
        diag_line: Optional plotted reference diagonal treated as an obstacle.
        colors: Optional mapping from label string to color passed through to
            ``_create_label_texts`` so each annotation matches its marker fill.

    Returns:
        The placed label artists, updated in-place on the axes.
    """
    # Screen-space layout constants.
    axes_inner_margin_px = 4.0
    point_pad_px = max(5.0, float(marker_size) * 0.9)
    diag_pad_px = 3.0
    label_bbox_pad_px = 2.0

    offset_inner = 3.5
    offset_near = 5.0
    offset_mid = 7.0
    offset_far = 9.0
    offset_outer = 11.0

    if not entries:
        return []

    # Create label artists first; positions are resolved after the plot is drawn.
    label_texts = _create_label_texts(ax, entries, fontsize=fontsize, colors=colors)

    # Resolve all obstacle geometry in display coordinates.
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Reserve a small inner safe region so labels do not cling to the axes frame.
    safe_bbox = _build_safe_bbox(
        ax,
        renderer=renderer,
        axes_inner_margin_px=axes_inner_margin_px,
    )

    # Convert label anchors from data space to display space for collision checks.
    anchor_points = _build_anchor_points(ax, entries)

    # Approximate each plotted marker by a padded screen-space bounding box.
    point_obstacles = _build_point_obstacles(anchor_points, point_pad_px=point_pad_px)

    # Approximate the reference diagonal by a sequence of small obstacle boxes.
    line_obstacles = _build_line_obstacles(
        ax,
        diag_line,
        diag_pad_px=diag_pad_px,
    )

    # Candidate label offsets are tried from near to far around each anchor point.
    # Keep an explicit tier index so the scorer always prefers earlier rings
    # over later fallback rings, even when multiple candidate positions are
    # otherwise similar. Each ring is sampled uniformly over angle.
    candidate_offsets = _build_candidate_offsets(
        offset_inner=offset_inner,
        offset_near=offset_near,
        offset_mid=offset_mid,
        offset_far=offset_far,
        offset_outer=offset_outer,
    )

    # Place the most crowded labels first so later labels fit around them.
    order = _build_placement_order(anchor_points)

    placed_bboxes: list[plt.matplotlib.transforms.Bbox] = []
    placed_states: list[
        tuple[
            float,
            tuple[float, float],
            str,
            str,
            plt.matplotlib.transforms.Bbox,
        ]
    ] = []
    placed_indices: list[int] = []

    # Greedily assign the least-colliding screen-space candidate to each label.
    for idx in order:
        text = label_texts[idx]
        best_state = _choose_best_state(
            text=text,
            candidate_offsets=candidate_offsets,
            renderer=renderer,
            label_bbox_pad_px=label_bbox_pad_px,
            placed_bboxes=placed_bboxes,
            point_obstacles=point_obstacles,
            line_obstacles=line_obstacles,
            safe_bbox=safe_bbox,
        )
        _apply_state(text, best_state)
        placed_bboxes.append(best_state[4])
        placed_states.append(best_state)
        placed_indices.append(idx)

    conflicting_positions = [
        pos
        for pos, state in enumerate(placed_states)
        if state[0] >= _LABEL_LAYOUT_WEIGHT_LABEL_OVERLAP
    ]

    for conflict_pos in conflicting_positions:
        target_idx = placed_indices[conflict_pos]
        target_state = placed_states[conflict_pos]
        target_bbox = target_state[4]

        cluster_positions = [
            pos for pos, bbox in enumerate(placed_bboxes) if bbox.overlaps(target_bbox)
        ]
        cluster_positions = sorted(set(cluster_positions + [conflict_pos]))
        if len(cluster_positions) <= 1:
            continue

        cluster_indices = [placed_indices[pos] for pos in cluster_positions]
        fixed_bboxes = [
            placed_bboxes[pos]
            for pos in range(len(placed_bboxes))
            if pos not in cluster_positions
        ]
        baseline_score = sum(placed_states[pos][0] for pos in cluster_positions)

        relayout_result = _solve_local_cluster(
            label_texts=label_texts,
            cluster_indices=cluster_indices,
            fixed_bboxes=fixed_bboxes,
            forced_first_idx=target_idx,
            candidate_offsets=candidate_offsets,
            renderer=renderer,
            label_bbox_pad_px=label_bbox_pad_px,
            point_obstacles=point_obstacles,
            line_obstacles=line_obstacles,
            safe_bbox=safe_bbox,
        )
        if relayout_result is None:
            continue

        relayout_states, relayout_score, relayout_order = relayout_result

        if relayout_score < baseline_score:
            pos_to_state = {
                idx: state
                for idx, state in zip(relayout_order, relayout_states, strict=False)
            }
            for local_idx in cluster_indices:
                state = pos_to_state[local_idx]
                local_text = label_texts[local_idx]
                _, (local_dx, local_dy), local_ha, local_va, local_bbox = state
                local_text.set_position((local_dx, local_dy))
                local_text.set_ha(local_ha)
                local_text.set_va(local_va)

                rewrite_pos = placed_indices.index(local_idx)
                placed_states[rewrite_pos] = state
                placed_bboxes[rewrite_pos] = local_bbox

    # Request a redraw so the final label positions are reflected on screen/export.
    fig.canvas.draw_idle()
    return label_texts
