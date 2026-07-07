# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Regression test: comment-format experiment files must tag signals with the
``# isotope`` header, so multi-isotope experiments filter correctly per isotope
(otherwise other-isotope peaks leak into a spectrum figure)."""

import pytest

from simpnmr.io.csv.exp import _load_legacy_experiments


@pytest.mark.unit
def test_legacy_experiment_tags_signals_with_isotope(tmp_path):
    csv = (
        "#temperature 302.15\n"
        "#magnetic_field 4.7\n"
        "#isotope 13C\n"
        "assignment,shift (ppm),width (Hz),area ()\n"
        "C3,1201.49,619.89,42.03\n"
        "C1,913.24,38.09,1.00\n"
    )
    path = tmp_path / "exp_13C.csv"
    path.write_text(csv)

    experiments = _load_legacy_experiments(str(path))
    signals = [s for e in experiments for s in e.signals]
    assert signals, "expected signals to be loaded"
    assert all(s.isotope == "13C" for s in signals)


@pytest.mark.unit
def test_legacy_experiment_without_isotope_header_is_untagged(tmp_path):
    csv = (
        "#temperature 302.15\n"
        "#magnetic_field 4.7\n"
        "assignment,shift (ppm),width (Hz),area ()\n"
        "aax,82.89,587.31,4108.48\n"
    )
    path = tmp_path / "exp.csv"
    path.write_text(csv)

    signals = [s for e in _load_legacy_experiments(str(path)) for s in e.signals]
    assert signals and all(s.isotope is None for s in signals)
