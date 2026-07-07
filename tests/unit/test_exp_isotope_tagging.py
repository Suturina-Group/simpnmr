# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Regression test: comment-format experiment files must tag signals with the
``# isotope`` header, so multi-isotope experiments filter correctly per isotope
(otherwise other-isotope peaks leak into a spectrum figure)."""

import pytest

from simpnmr.io.csv.exp import _load_legacy_experiments, _load_wide_experiments


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


@pytest.mark.unit
def test_wide_experiment_tags_signals_from_trailing_isotope_column(tmp_path):
    """A trailing (global) isotope column must be read, not dropped by the
    trailing-empty-cell strip on the temperature/field rows."""
    csv = (
        "temperature (K),298,298,298,\n"
        "magnetic_field (T),11.75,11.75,11.75,\n"
        "assignment,shift (ppm),width (Hz),area (),isotope\n"
        "C3,1201.49,619.89,42.03,13C\n"
        "H1,10.5,50.0,3.0,1H\n"
    )
    path = tmp_path / "1H_and_13C_exp_298.csv"
    path.write_text(csv)

    signals = {
        s.assignment: s.isotope
        for e in _load_wide_experiments(str(path))
        for s in e.signals
    }
    assert signals == {"C3": "13C", "H1": "1H"}
