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


def generate_assignment_permutations(
    experiment: Experiment,
    groups: list[list[str]] | None = None,
) -> list[list[str]]:
    """Generate assignment permutations consistent with grouping constraints.

    Args:
        experiment: Reference experiment whose assignment labels are permuted.
        groups: Groups of assignment labels that may be permuted within each group.
            Labels not present in any group are treated as fixed (singletons).

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

    # Validate grouped labels.
    missing = [lab for lab in grouped_labels if lab not in exp_labels]
    if missing:
        raise ValueError(
            "Grouped assignment label(s) not present in experiment: "
            + ", ".join(missing)
        )

    # Add fixed assignments as singleton groups.
    fixed = [[lab] for lab in exp_labels if lab not in grouped_labels]
    group_list = group_list + fixed

    # Generate all permutations subject to grouping constraints.
    per_group = [permutations(group) for group in group_list]
    perms = [list(chain.from_iterable(e)) for e in product(*per_group, repeat=1)]

    # Map each label in group_list order to its position in the experiment.
    l2i = {label: idx for idx, label in enumerate(exp_labels)}
    group_to_exp = [
        l2i[lab] for lab in (list(np.concatenate(group_list)) if group_list else [])
    ]

    # Reorder each permuted assignment list to match the original experiment ordering.
    order = np.argsort(group_to_exp)
    all_new_assignments = [[new_assgn[o] for o in order] for new_assgn in perms]

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
            sum to 1, then ``width_weight * (norm_width - norm_inv_r6)**2`` is
            added to the cost. Requires ``molecule.paramagnetic_centre`` to be
            set. Set to 0.0 (default) to skip.
        r1_weight: Weight for the R1-consistency term, analogous to
            ``width_weight`` but using experimental R1 values. Signals with
            no R1 data (``None`` or ``NaN``) are excluded from this term.
            Set to 0.0 (default) to skip.

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

    def _apply_assignment(exp: Experiment, assignment: list[str]) -> None:
        for i, chem_label in enumerate(assignment):
            exp.signals[i].assignment = chem_label

    def _current_assignment(exp: Experiment) -> list[str]:
        return [sig.assignment for sig in exp.signals]

    best_rmse = np.inf
    best_assignment: list[str] | None = None
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
        # Subsequent attempts: random permutation to escape local minima.
        initial = _current_assignment(experiment)
        if attempt == 0:
            current_assignment = list(initial)
        else:
            perm = np.random.permutation(len(initial))
            current_assignment = [initial[i] for i in perm]
        _apply_assignment(trial_experiment, current_assignment)

        converged = False
        final_assignment = list(current_assignment)
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
            pred_total = {label: pred_para[label] + dia[label] for label in pred_para}

            # Map chem_labels to their averaging groups
            cl_to_group: dict[str, list[str]] = {}
            for group in average_labels:
                nuc = next(n for n in molecule.nuclei if n.label == group[0])
                cl_to_group[nuc.chem_label] = group

            # Average predicted shifts within each group
            avg_pred: dict[str, float] = {}
            for cl, group in cl_to_group.items():
                shifts = [pred_total[lbl] for lbl in group]
                avg_pred[cl] = float(np.mean(shifts))

            # Build cost matrix and solve with Hungarian algorithm
            labels_ordered = sorted(avg_pred)
            n_sig = len(trial_experiment.signals)
            n_lbl = len(labels_ordered)
            cost = np.zeros((n_sig, n_lbl))

            # Precompute area term if requested
            if area_weight > 0.0:
                exp_areas = np.array(
                    [sig.area for sig in trial_experiment.signals], dtype=float
                )
                total_exp_area = exp_areas.sum()
                norm_exp_areas = (
                    exp_areas / total_exp_area if total_exp_area > 0 else exp_areas
                )

                group_sizes = np.array(
                    [
                        sum(
                            1
                            for n in molecule.nuclei
                            if n.chem_label == cl
                        )
                        for cl in labels_ordered
                    ],
                    dtype=float,
                )
                total_group_size = group_sizes.sum()
                norm_group_sizes = (
                    group_sizes / total_group_size
                    if total_group_size > 0
                    else group_sizes
                )

            # Precompute mean(1/r^6) per label for width/R1 terms
            norm_inv_r6 = None
            if (width_weight > 0.0 or r1_weight > 0.0) and molecule.paramagnetic_centre is not None:
                centre = np.asarray(molecule.paramagnetic_centre, dtype=float)
                inv_r6_per_label = np.array(
                    [
                        np.mean(
                            [
                                1.0 / max(float(np.linalg.norm(n.coord - centre)), 1e-6) ** 6
                                for n in molecule.nuclei
                                if n.chem_label == cl
                            ]
                        )
                        for cl in labels_ordered
                    ],
                    dtype=float,
                )
                total_inv_r6 = inv_r6_per_label.sum()
                norm_inv_r6 = (
                    inv_r6_per_label / total_inv_r6 if total_inv_r6 > 0 else inv_r6_per_label
                )

            # Precompute normalised experimental widths
            norm_exp_widths = None
            if width_weight > 0.0 and norm_inv_r6 is not None:
                exp_widths = np.array(
                    [sig.width for sig in trial_experiment.signals], dtype=float
                )
                total_width = exp_widths.sum()
                norm_exp_widths = (
                    exp_widths / total_width if total_width > 0 else exp_widths
                )

            # Precompute normalised experimental R1 values
            norm_exp_r1 = None
            if r1_weight > 0.0 and norm_inv_r6 is not None:
                raw_r1 = np.array(
                    [
                        float(sig.r1) if sig.r1 is not None and not np.isnan(float(sig.r1)) else 0.0
                        for sig in trial_experiment.signals
                    ],
                    dtype=float,
                )
                total_r1 = raw_r1.sum()
                norm_exp_r1 = raw_r1 / total_r1 if total_r1 > 0 else raw_r1

            for i, sig in enumerate(trial_experiment.signals):
                for j, cl in enumerate(labels_ordered):
                    cost[i, j] = (sig.shift - avg_pred[cl]) ** 2
                    if area_weight > 0.0:
                        cost[i, j] += area_weight * (
                            norm_exp_areas[i] - norm_group_sizes[j]
                        ) ** 2
                    if width_weight > 0.0 and norm_exp_widths is not None:
                        cost[i, j] += width_weight * (
                            norm_exp_widths[i] - norm_inv_r6[j]
                        ) ** 2
                    if r1_weight > 0.0 and norm_exp_r1 is not None:
                        cost[i, j] += r1_weight * (
                            norm_exp_r1[i] - norm_inv_r6[j]
                        ) ** 2

            _, col_idx = linear_sum_assignment(cost)
            new_assignment = [labels_ordered[j] for j in col_idx]
            final_assignment = list(new_assignment)

            # Check for convergence
            if new_assignment == current_assignment:
                converged = True
                logger.debug(
                    "  Converged at iteration %d",
                    iteration + 1,
                )
                break

            current_assignment = new_assignment
            _apply_assignment(trial_experiment, current_assignment)

        # Only record and update best_rmse when the attempt converged.
        # If max_iter was hit without convergence, trial state is
        # unreliable so we discard this attempt entirely.
        if converged:
            attempt_rmse = trial_model.rmse
            if best_assignment is None or attempt_rmse < best_rmse:
                best_rmse = attempt_rmse
                best_assignment = list(final_assignment)
            logger.info(
                "Attempt %d converged, RMSE = %.6f", attempt + 1, attempt_rmse
            )
        else:
            logger.info(
                "Attempt %d did not converge within %d iterations, discarding",
                attempt + 1,
                max_iter,
            )

        attempt += 1

    if best_assignment is None:
        raise RuntimeError(
            f"Hungarian assignment: no attempt converged within {max_iter} "
            f"iterations across {n_attempts} restarts."
            "Fallback to an alternative method if feasible, "
            "or consider increasing max_iter and/or n_attempts."
        )

    # Restore the best assignment into the caller-owned experiment and
    # perform exactly one final fit on the caller-owned model so that the
    # externally visible state corresponds to the selected best attempt.
    _apply_assignment(experiment, best_assignment)
    susc_model.fit_to(
        molecule,
        experiment,
        average_labels=average_labels,
    )

    return susc_model.rmse, best_assignment
