# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Generate constrained assignment permutations.

Provides helpers to enumerate assignment label permutations subject to grouping
constraints for downstream fitting workflows.
"""

import copy
import logging
from itertools import chain, permutations, product

import numpy as np
from scipy.optimize import linear_sum_assignment

from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.fitting import models

logger = logging.getLogger(__name__)


def _constrained_group_perms(
    signals: list[str],
    labels_pool: list[str],
    signal_allowed: dict[str, set[str]],
) -> list[list[str]]:
    """Generate permutations of labels_pool for signals using backtracking.

    Only branches where the proposed label is in ``signal_allowed[signal]``
    (or all labels when unconstrained) are explored, pruning invalid
    sub-trees immediately rather than generating-then-filtering.

    Args:
        signals: Ordered signal labels in this group (their current assignments).
        labels_pool: The same set of chem_labels to be redistributed.
        signal_allowed: signal_label → set of allowed chem_labels.
            Absent keys mean "no constraint" (all labels allowed).

    Returns:
        List of valid permutations, each a list of chem_labels in ``signals``
        order.
    """
    results: list[list[str]] = []

    def _bt(pos: int, remaining: list[str], current: list[str]) -> None:
        if pos == len(signals):
            results.append(current[:])
            return
        sig = signals[pos]
        allowed = signal_allowed.get(sig)  # None → unconstrained
        for i, label in enumerate(remaining):
            if allowed is not None and label not in allowed:
                continue
            current.append(label)
            _bt(pos + 1, remaining[:i] + remaining[i + 1:], current)
            current.pop()

    _bt(0, list(labels_pool), [])
    return results



def generate_assignment_permutations(
    experiment: Experiment,
    groups: list[list[str]] | None = None,
    signal_allowed: dict[str, set[str]] | None = None,
) -> list[list[str]]:
    """Generate assignment permutations consistent with grouping constraints.

    Args:
        experiment: Reference experiment whose assignment labels are permuted.
        groups: Groups of assignment labels that may be permuted within each
            group. Labels not present in any group are treated as fixed
            (singletons).

    Returns:
        List of permuted assignment label lists, each ordered to match the
        original experiment signal ordering.

    Raises:
        ValueError: If a grouped label is not present in the experiment.
    """

    exp_labels = list(experiment.keys())

    # Normalise input and avoid mutating caller-provided lists.
    if groups is None:
        group_list: list[list[str]] = []
    else:
        group_list = [list(g) for g in groups]

    grouped_labels: list[str]
    if group_list:
        grouped_labels = list(np.concatenate(group_list))
    else:
        grouped_labels = []

    # Validate grouped labels — drop any that have no experimental signal.
    missing = [lab for lab in grouped_labels if lab not in exp_labels]
    if missing:
        logger.warning(
            "Grouped assignment label(s) not present in experiment "
            "(skipped): %s",
            ", ".join(missing),
        )
        group_list = [
            [lab for lab in g if lab not in missing] for g in group_list
        ]
        group_list = [g for g in group_list if g]
        grouped_labels = [
            lab for lab in grouped_labels if lab not in missing
        ]

    # Add fixed assignments as singleton groups.
    fixed = [[lab] for lab in exp_labels if lab not in grouped_labels]
    group_list = group_list + fixed

    # Generate permutations subject to grouping (and optional signal) constraints.
    if signal_allowed:
        per_group = [
            _constrained_group_perms(group, group, signal_allowed)
            for group in group_list
        ]
    else:
        per_group = [list(permutations(group)) for group in group_list]
    perms = [
        list(chain.from_iterable(e))
        for e in product(*per_group, repeat=1)
    ]

    # Map each label in group_list order to its position in the experiment.
    l2i = {label: idx for idx, label in enumerate(exp_labels)}
    flat = list(np.concatenate(group_list)) if group_list else []
    group_to_exp = [l2i[lab] for lab in flat]

    # Reorder each permuted list to match the original experiment ordering.
    order = np.argsort(group_to_exp)
    all_new_assignments = [
        [new_assgn[o] for o in order] for new_assgn in perms
    ]

    return all_new_assignments


def fit_with_hungarian_assignment(
    molecule: Molecule,
    susc_model: models.SusceptibilityModel,
    experiment: Experiment,
    average_labels: list[list[str]],
    n_attempts: int,
    max_iter: int,
    rmse_threshold: float,
    area_weight: float = 0.0,
    width_weight: float = 0.0,
    r1_weight: float = 0.0,
    isotope_partitions: list[tuple[list[int], set[str]]] | None = None,
) -> tuple[float, list[str]]:
    """Fit assignments with alternating Hungarian reassignment and chi refits.

    This routine alternates between fitting the susceptibility model to a
    temporary experiment assignment and recomputing the assignment with the
    Hungarian algorithm from predicted shifts. The first attempt uses the
    assignment already present in the experiment as a warm start. Subsequent
    attempts use random permutations to reduce the risk of converging to a
    poor local optimum. Intermediate attempts are evaluated on temporary
    copies; only the selected best assignment is written back to the caller's
    experiment and model.

    Args:
        molecule: Molecule containing nuclei and tensor data used for fitting.
        susc_model: Susceptibility model instance to fit against the
            experiment.
        experiment: Experiment containing observed shifts and current
            assignments.
        average_labels: Groups of labels whose predicted shifts are averaged
            before solving the assignment problem.
        n_attempts: Maximum number of restart attempts.
        max_iter: Maximum number of alternating fit-assignment iterations per
            attempt.
        rmse_threshold: Early-stop threshold for RMSE (ppm). Stops early
            if a converged attempt achieves RMSE below this value.
            Set to 0.0 to disable early stopping.
        area_weight: Weight for the area-consistency term in the cost matrix.
            Each experimental signal area and each label's group size (number
            of equivalent nuclei) are independently normalised to sum to 1,
            then ``area_weight * (norm_area - norm_group_size)**2`` is added
            to the shift-squared cost. Set to 0.0 (default) to use shifts
            only.
        width_weight: Weight for the linewidth-consistency term. Experimental
            linewidths and per-label mean(1/r^6) values are each normalised to
            sum to 1, then
            ``width_weight * (norm_width - norm_inv_r6)**2`` is added to the
            cost. Requires ``molecule.paramagnetic_centre`` to be set. Set to
            0.0 (default) to skip.
        r1_weight: Weight for the R1-consistency term, analogous to
            ``width_weight`` but using experimental R1 values. Signals with
            no R1 data (``None`` or ``NaN``) are excluded from this term.
            Set to 0.0 (default) to skip.
        isotope_partitions: Optional list of ``(signal_indices, label_set)``
            pairs, one per isotope. When provided, the susceptibility tensor
            is fitted jointly using all signals and all nuclei, but the
            Hungarian cost matrix is built and solved independently for each
            partition — signals from one isotope never compete for labels
            belonging to another. When ``None`` (default), all signals and
            labels form a single partition.

    Returns:
        The RMSE (ppm) after restoring and re-fitting the best assignment,
        together with that assignment.

    Raises:
        RuntimeError: If no attempt converges within the allowed number of
            iterations and restart attempts.
    """
    logger.info(
        "Starting Hungarian assignment optimization for temperature %.4f K",
        experiment.temperature,
    )
    logger.info(
        "Parameters: n_attempts=%d, max_iter=%d, RMSE threshold=%.6f",
        n_attempts,
        max_iter,
        rmse_threshold,
    )

    # All signals are candidates — the solver will pick the best pairing.
    # When n_signals > n_labels, signals not selected by the solver keep
    # their current assignment (frozen for that iteration).
    _n_signals = len(experiment.signals)
    _mol_labels: set[str] = {nuc.chem_label for nuc in molecule.nuclei}
    _n_labels = len(_mol_labels)

    if _n_signals == 0:
        raise RuntimeError(
            "Hungarian assignment: experiment has no signals."
        )
    logger.info(
        "Hungarian assignment: %d experimental signals, %d molecule labels",
        _n_signals, _n_labels,
    )
    if _n_signals > _n_labels:
        logger.warning(
            "Hungarian assignment: %d signal(s) exceed the number of "
            "molecule labels (%d) — excess signals keep their current "
            "assignment.",
            _n_signals - _n_labels, _n_labels,
        )

    # assignment_map: signal_index -> label (only for assigned signals)
    AssignMap = dict  # type alias for clarity

    def _apply_assignment(
        exp: Experiment, assignment_map: AssignMap
    ) -> None:
        """Write labels from assignment_map; unselected signals untouched."""
        for idx, label in assignment_map.items():
            exp.signals[idx].assignment = label

    def _current_assignment_map(exp: Experiment) -> AssignMap:
        """Return {signal_index: label} for all signals."""
        return {i: s.assignment for i, s in enumerate(exp.signals)}

    best_rmse = np.inf
    best_assignment_map: AssignMap | None = None
    attempt = 0

    while best_rmse > rmse_threshold and attempt < n_attempts:
        logger.debug(
            "Attempt %d out of maximum number of attempts %d: "
            "starting Hungarian optimisation",
            attempt + 1,
            n_attempts,
        )

        trial_experiment = copy.deepcopy(experiment)
        trial_model = copy.deepcopy(susc_model)

        # Attempt 0: warm start using the assignment already in the exp file.
        # Subsequent attempts: random permutation of all signal assignments
        # to escape local minima.
        initial_map = _current_assignment_map(experiment)
        if attempt == 0:
            current_map = dict(initial_map)
        else:
            all_labels = list(initial_map.values())
            perm = np.random.permutation(len(all_labels))
            current_map = {
                idx: all_labels[perm[k]]
                for k, idx in enumerate(initial_map)
            }
        _apply_assignment(trial_experiment, current_map)

        converged = False
        final_map: AssignMap = dict(current_map)
        for iteration in range(max_iter):
            # Fit susceptibility model to current assignment
            trial_model.fit_to(
                molecule,
                trial_experiment,
                average_labels=average_labels,
            )
            logger.debug(
                "  Iteration %d/%d: RMSE = %.6f",
                iteration + 1,
                max_iter,
                trial_model.rmse,
            )

            # Predict paramagnetic shifts from fitted model
            pred_para = trial_model.model(
                trial_model.final_var_values,
                molecule.nuclei,
            )

            # Add diamagnetic contribution
            dia = {nuc.label: nuc.shift.dia for nuc in molecule.nuclei}
            pred_total = {
                label: pred_para[label] + dia[label]
                for label in pred_para
            }

            # Map chem_labels to their averaging groups
            cl_to_group: dict[str, list[str]] = {}
            for group in average_labels:
                nuc = next(
                    n for n in molecule.nuclei if n.label == group[0]
                )
                cl_to_group[nuc.chem_label] = group

            # Average predicted shifts within each group
            avg_pred: dict[str, float] = {}
            for cl, group in cl_to_group.items():
                shifts = [pred_total[lbl] for lbl in group]
                avg_pred[cl] = float(np.mean(shifts))

            # Resolve partitions: one per isotope when provided, else one
            # covering all signals and all labels.
            _partitions: list[tuple[list[int], list[str]]]
            if isotope_partitions is not None:
                _partitions = [
                    (sig_idxs, sorted(lbl_set & set(avg_pred)))
                    for sig_idxs, lbl_set in isotope_partitions
                    if lbl_set & set(avg_pred)
                ]
            else:
                _partitions = [
                    (list(range(len(trial_experiment.signals))),
                     sorted(avg_pred))
                ]

            new_map: AssignMap = dict(current_map)
            all_row_idx: list[int] = []

            for part_sig_idxs, part_labels in _partitions:
                if not part_sig_idxs or not part_labels:
                    continue
                part_signals = [
                    trial_experiment.signals[i] for i in part_sig_idxs
                ]
                n_sig = len(part_signals)
                n_lbl = len(part_labels)
                cost = np.zeros((n_sig, n_lbl))

                # Area term
                norm_exp_areas = None
                norm_group_sizes = None
                if area_weight > 0.0:
                    exp_areas = np.array(
                        [sig.area for sig in part_signals], dtype=float
                    )
                    total_exp_area = exp_areas.sum()
                    norm_exp_areas = (
                        exp_areas / total_exp_area
                        if total_exp_area > 0 else exp_areas
                    )
                    group_sizes = np.array(
                        [
                            sum(
                                1 for n in molecule.nuclei
                                if n.chem_label == cl
                            )
                            for cl in part_labels
                        ],
                        dtype=float,
                    )
                    total_group_size = group_sizes.sum()
                    norm_group_sizes = (
                        group_sizes / total_group_size
                        if total_group_size > 0 else group_sizes
                    )

                # mean(1/r^6) per label for width/R1 terms
                norm_inv_r6 = None
                if (
                    (width_weight > 0.0 or r1_weight > 0.0)
                    and molecule.paramagnetic_centre is not None
                ):
                    centre = np.asarray(
                        molecule.paramagnetic_centre, dtype=float
                    )
                    inv_r6_per_label = np.array(
                        [
                            np.mean(
                                [
                                    1.0 / max(
                                        float(
                                            np.linalg.norm(n.coord - centre)
                                        ),
                                        1e-6,
                                    ) ** 6
                                    for n in molecule.nuclei
                                    if n.chem_label == cl
                                ]
                            )
                            for cl in part_labels
                        ],
                        dtype=float,
                    )
                    total_inv_r6 = inv_r6_per_label.sum()
                    norm_inv_r6 = (
                        inv_r6_per_label / total_inv_r6
                        if total_inv_r6 > 0 else inv_r6_per_label
                    )

                norm_exp_widths = None
                if width_weight > 0.0 and norm_inv_r6 is not None:
                    exp_widths = np.array(
                        [sig.width for sig in part_signals], dtype=float
                    )
                    total_width = exp_widths.sum()
                    norm_exp_widths = (
                        exp_widths / total_width
                        if total_width > 0 else exp_widths
                    )

                norm_exp_r1 = None
                if r1_weight > 0.0 and norm_inv_r6 is not None:
                    raw_r1 = np.array(
                        [
                            float(sig.r1)
                            if sig.r1 is not None
                            and not np.isnan(float(sig.r1))
                            else 0.0
                            for sig in part_signals
                        ],
                        dtype=float,
                    )
                    total_r1 = raw_r1.sum()
                    norm_exp_r1 = (
                        raw_r1 / total_r1 if total_r1 > 0 else raw_r1
                    )

                for i, sig in enumerate(part_signals):
                    for j, cl in enumerate(part_labels):
                        cost[i, j] = (sig.shift - avg_pred[cl]) ** 2
                        if area_weight > 0.0:
                            cost[i, j] += area_weight * (
                                norm_exp_areas[i] - norm_group_sizes[j]
                            ) ** 2
                        if (
                            width_weight > 0.0
                            and norm_exp_widths is not None
                        ):
                            cost[i, j] += width_weight * (
                                norm_exp_widths[i] - norm_inv_r6[j]
                            ) ** 2
                        if r1_weight > 0.0 and norm_exp_r1 is not None:
                            cost[i, j] += r1_weight * (
                                norm_exp_r1[i] - norm_inv_r6[j]
                            ) ** 2

                row_idx, col_idx = linear_sum_assignment(cost)
                for local_i, lbl_j in zip(row_idx, col_idx):
                    global_i = part_sig_idxs[local_i]
                    new_map[global_i] = part_labels[lbl_j]
                    all_row_idx.append(global_i)

            final_map = dict(new_map)

            # Convergence: compare newly assigned signals across all partitions
            solver_slice_new = {i: new_map[i] for i in all_row_idx}
            solver_slice_cur = {i: current_map[i] for i in all_row_idx}
            if solver_slice_new == solver_slice_cur:
                converged = True
                logger.debug(
                    "  Converged at iteration %d",
                    iteration + 1,
                )
                break

            current_map = new_map
            _apply_assignment(trial_experiment, current_map)

        # Only record and update best_rmse when the attempt converged.
        # If max_iter was hit without convergence, trial state is
        # unreliable so we discard this attempt entirely.
        if converged:
            attempt_rmse = trial_model.rmse
            if best_assignment_map is None or attempt_rmse < best_rmse:
                best_rmse = attempt_rmse
                best_assignment_map = dict(final_map)
            logger.info(
                "Attempt %d converged, RMSE = %.6f",
                attempt + 1,
                attempt_rmse,
            )
        else:
            logger.info(
                "Attempt %d did not converge within %d iterations, discarding",
                attempt + 1,
                max_iter,
            )

        attempt += 1

    if best_assignment_map is None:
        raise RuntimeError(
            f"Hungarian assignment: no attempt converged within {max_iter} "
            f"iterations across {n_attempts} restarts."
            "Fallback to an alternative method if feasible, "
            "or consider increasing max_iter and/or n_attempts."
        )

    # Restore the best assignment into the caller-owned experiment and
    # perform exactly one final fit on the caller-owned model so that the
    # externally visible state corresponds to the selected best attempt.
    _apply_assignment(experiment, best_assignment_map)
    susc_model.fit_to(
        molecule,
        experiment,
        average_labels=average_labels,
    )

    best_assignment = [
        best_assignment_map[i] for i in sorted(best_assignment_map)
    ]
    return susc_model.rmse, best_assignment


def fit_with_hungarian_assignment_multi(
    records: list[dict],
    n_attempts: int,
    max_iter: int,
    rmse_threshold: float,
    area_weight: float = 0.0,
    width_weight: float = 0.0,
    r1_weight: float = 0.0,
    isotope_partitions: list[tuple[list[int], set[str]]] | None = None,
) -> tuple[float, list[str]]:
    """Hungarian assignment optimised jointly across multiple experiments.

    Finds one assignment (shared across all experiments) that minimises the
    total cost summed over all (T, B) conditions. Each experiment's
    susceptibility model is fitted independently per iteration so that each
    condition retains its own tensor parameters, but the assignment is
    constrained to be identical across all experiments.

    Args:
        records: List of dicts, each containing:
            ``"molecule"`` (:class:`~simpnmr.core.domain.mol.Molecule`),
            ``"susc_model"`` (SusceptibilityModel),
            ``"experiment"`` (:class:`~simpnmr.core.domain.exp.Experiment`),
            ``"average_labels"`` (list of label groups).
        n_attempts: Maximum number of random-restart attempts.
        max_iter: Maximum alternating fit-assign iterations per attempt.
        rmse_threshold: Early-stop RMSE threshold (ppm). Set to 0.0 to
            disable.
        area_weight: Weight for area-consistency cost term.
        width_weight: Weight for linewidth-consistency cost term.
        r1_weight: Weight for R1-consistency cost term.
        isotope_partitions: Per-isotope ``(signal_indices, label_set)`` pairs.
            Signal indices are relative to the first experiment's signal list
            (all experiments are assumed to share the same signal ordering).
            When ``None``, all signals and labels form one partition.

    Returns:
        Mean RMSE (ppm) across experiments after restoring and re-fitting
        the best assignment, together with the shared assignment list.

    Raises:
        RuntimeError: If no attempt converges.
    """
    if not records:
        raise ValueError("fit_with_hungarian_assignment_multi: no records.")

    # All experiments must share the same signal count and ordering.
    n_signals = len(records[0]["experiment"].signals)
    for rec in records[1:]:
        if len(rec["experiment"].signals) != n_signals:
            raise ValueError(
                "fit_with_hungarian_assignment_multi: all experiments must "
                "have the same number of signals."
            )

    logger.info(
        "Multi-experiment Hungarian: %d experiments, %d signals each",
        len(records), n_signals,
    )

    AssignMap = dict

    def _apply_assignment_all(
        trial_recs: list[dict], assignment_map: AssignMap
    ) -> None:
        for rec in trial_recs:
            for idx, label in assignment_map.items():
                rec["experiment"].signals[idx].assignment = label

    def _current_assignment_map(rec: dict) -> AssignMap:
        return {
            i: s.assignment
            for i, s in enumerate(rec["experiment"].signals)
        }

    def _build_cost_partition(
        trial_recs: list[dict],
        part_sig_idxs: list[int],
        part_labels: list[str],
        avg_preds: list[dict],
    ) -> np.ndarray:
        """Sum cost matrices across all experiments for one partition."""
        n_sig = len(part_sig_idxs)
        n_lbl = len(part_labels)
        cost_total = np.zeros((n_sig, n_lbl))

        for rec, avg_pred in zip(trial_recs, avg_preds):
            exp = rec["experiment"]
            mol = rec["molecule"]
            part_signals = [exp.signals[i] for i in part_sig_idxs]
            cost = np.zeros((n_sig, n_lbl))

            norm_exp_areas = norm_group_sizes = None
            if area_weight > 0.0:
                exp_areas = np.array(
                    [sig.area for sig in part_signals], dtype=float
                )
                tot = exp_areas.sum()
                norm_exp_areas = exp_areas / tot if tot > 0 else exp_areas
                gsizes = np.array(
                    [
                        sum(1 for n in mol.nuclei if n.chem_label == cl)
                        for cl in part_labels
                    ], dtype=float,
                )
                tot_gs = gsizes.sum()
                norm_group_sizes = (
                    gsizes / tot_gs if tot_gs > 0 else gsizes
                )

            norm_inv_r6 = None
            if (
                (width_weight > 0.0 or r1_weight > 0.0)
                and getattr(mol, "paramagnetic_centre", None) is not None
            ):
                centre = np.asarray(mol.paramagnetic_centre, dtype=float)
                ir6 = np.array(
                    [
                        np.mean([
                            1.0 / max(
                                float(np.linalg.norm(n.coord - centre)),
                                1e-6,
                            ) ** 6
                            for n in mol.nuclei if n.chem_label == cl
                        ])
                        for cl in part_labels
                    ], dtype=float,
                )
                tot_ir6 = ir6.sum()
                norm_inv_r6 = ir6 / tot_ir6 if tot_ir6 > 0 else ir6

            norm_exp_widths = norm_exp_r1 = None
            if width_weight > 0.0 and norm_inv_r6 is not None:
                ws = np.array(
                    [sig.width for sig in part_signals], dtype=float
                )
                tot_w = ws.sum()
                norm_exp_widths = ws / tot_w if tot_w > 0 else ws
            if r1_weight > 0.0 and norm_inv_r6 is not None:
                raw = np.array(
                    [
                        float(sig.r1)
                        if sig.r1 is not None
                        and not np.isnan(float(sig.r1))
                        else 0.0
                        for sig in part_signals
                    ], dtype=float,
                )
                tot_r = raw.sum()
                norm_exp_r1 = raw / tot_r if tot_r > 0 else raw

            for i, sig in enumerate(part_signals):
                for j, cl in enumerate(part_labels):
                    cost[i, j] = (sig.shift - avg_pred.get(cl, 0.0)) ** 2
                    if area_weight > 0.0 and norm_exp_areas is not None:
                        cost[i, j] += area_weight * (
                            norm_exp_areas[i] - norm_group_sizes[j]
                        ) ** 2
                    if (
                        width_weight > 0.0
                        and norm_exp_widths is not None
                    ):
                        cost[i, j] += width_weight * (
                            norm_exp_widths[i] - norm_inv_r6[j]
                        ) ** 2
                    if r1_weight > 0.0 and norm_exp_r1 is not None:
                        cost[i, j] += r1_weight * (
                            norm_exp_r1[i] - norm_inv_r6[j]
                        ) ** 2

            cost_total += cost

        return cost_total

    best_rmse = np.inf
    best_assignment_map: AssignMap | None = None
    attempt = 0

    while best_rmse > rmse_threshold and attempt < n_attempts:
        logger.debug(
            "Multi-exp Hungarian attempt %d / %d",
            attempt + 1, n_attempts,
        )

        # Deep-copy all records for this attempt
        trial_recs = [
            {
                "molecule": copy.deepcopy(rec["molecule"]),
                "susc_model": copy.deepcopy(rec["susc_model"]),
                "experiment": copy.deepcopy(rec["experiment"]),
                "average_labels": rec["average_labels"],
            }
            for rec in records
        ]

        # Warm start or random permutation
        initial_map = _current_assignment_map(records[0])
        if attempt == 0:
            current_map = dict(initial_map)
        else:
            all_labels = list(initial_map.values())
            perm = np.random.permutation(len(all_labels))
            current_map = {
                idx: all_labels[perm[k]]
                for k, idx in enumerate(initial_map)
            }
        _apply_assignment_all(trial_recs, current_map)

        converged = False
        final_map: AssignMap = dict(current_map)

        for iteration in range(max_iter):
            # Fit each model independently to its own experiment
            rmses = []
            avg_preds: list[dict] = []
            for rec in trial_recs:
                rec["susc_model"].fit_to(
                    rec["molecule"],
                    rec["experiment"],
                    average_labels=rec["average_labels"],
                )
                rmses.append(rec["susc_model"].rmse)

                pred_para = rec["susc_model"].model(
                    rec["susc_model"].final_var_values,
                    rec["molecule"].nuclei,
                )
                dia = {
                    nuc.label: nuc.shift.dia
                    for nuc in rec["molecule"].nuclei
                }
                pred_total = {
                    lbl: pred_para[lbl] + dia[lbl]
                    for lbl in pred_para
                }

                cl_to_group: dict[str, list[str]] = {}
                for group in rec["average_labels"]:
                    nuc = next(
                        n for n in rec["molecule"].nuclei
                        if n.label == group[0]
                    )
                    cl_to_group[nuc.chem_label] = group

                ap: dict[str, float] = {}
                for cl, group in cl_to_group.items():
                    ap[cl] = float(
                        np.mean([pred_total[lbl] for lbl in group])
                    )
                avg_preds.append(ap)

            # Resolve partitions
            all_labels_set = set(avg_preds[0].keys())
            _partitions: list[tuple[list[int], list[str]]]
            if isotope_partitions is not None:
                _partitions = [
                    (
                        sig_idxs,
                        sorted(lbl_set & all_labels_set),
                    )
                    for sig_idxs, lbl_set in isotope_partitions
                    if lbl_set & all_labels_set
                ]
            else:
                _partitions = [
                    (list(range(n_signals)), sorted(all_labels_set))
                ]

            new_map: AssignMap = dict(current_map)
            all_row_idx: list[int] = []

            for part_sig_idxs, part_labels in _partitions:
                if not part_sig_idxs or not part_labels:
                    continue
                cost = _build_cost_partition(
                    trial_recs, part_sig_idxs, part_labels, avg_preds
                )
                row_idx, col_idx = linear_sum_assignment(cost)
                for local_i, lbl_j in zip(row_idx, col_idx):
                    global_i = part_sig_idxs[local_i]
                    new_map[global_i] = part_labels[lbl_j]
                    all_row_idx.append(global_i)

            final_map = dict(new_map)

            mean_rmse = float(np.mean(rmses))
            logger.debug(
                "  Iter %d/%d: mean RMSE = %.6f",
                iteration + 1, max_iter, mean_rmse,
            )

            solver_slice_new = {i: new_map[i] for i in all_row_idx}
            solver_slice_cur = {i: current_map[i] for i in all_row_idx}
            if solver_slice_new == solver_slice_cur:
                converged = True
                logger.debug(
                    "  Converged at iteration %d", iteration + 1
                )
                break

            current_map = new_map
            _apply_assignment_all(trial_recs, current_map)

        if converged:
            attempt_rmse = float(
                np.mean([rec["susc_model"].rmse for rec in trial_recs])
            )
            if best_assignment_map is None or attempt_rmse < best_rmse:
                best_rmse = attempt_rmse
                best_assignment_map = dict(final_map)
            logger.info(
                "Multi-exp attempt %d converged, mean RMSE = %.6f",
                attempt + 1, attempt_rmse,
            )
        else:
            logger.info(
                "Multi-exp attempt %d did not converge within %d iterations,"
                " discarding",
                attempt + 1, max_iter,
            )
        attempt += 1

    if best_assignment_map is None:
        raise RuntimeError(
            f"Multi-experiment Hungarian: no attempt converged within "
            f"{max_iter} iterations across {n_attempts} restarts."
        )

    # Apply best assignment to all caller-owned experiments and refit
    for rec in records:
        for idx, label in best_assignment_map.items():
            rec["experiment"].signals[idx].assignment = label
        rec["susc_model"].fit_to(
            rec["molecule"],
            rec["experiment"],
            average_labels=rec["average_labels"],
        )

    best_assignment = [
        best_assignment_map[i] for i in sorted(best_assignment_map)
    ]
    mean_rmse = float(
        np.mean([rec["susc_model"].rmse for rec in records])
    )
    return mean_rmse, best_assignment
