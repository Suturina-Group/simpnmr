# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Generate constrained assignment permutations.

Provides helpers to enumerate assignment label permutations subject to grouping
constraints for downstream fitting workflows.
"""

from itertools import chain, permutations, product

import numpy as np

from simpnmr.core.domain.experiment import Experiment


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
