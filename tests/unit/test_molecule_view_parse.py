# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Regression tests for XYZ element parsing in the GUI molecule viewer.

SimpNMR structure files carry labelled atoms ("Dy1", "O1", "C2_up") in the
element column. The 3Dmol viewer must recover the bare element symbol,
otherwise every atom renders as the unknown-element colour with bogus,
distance-inferred bonds.
"""

import pytest

pytest.importorskip("simpnmr.gui.molecule_view")

from simpnmr.gui.molecule_view import parse_xyz


def _write(tmp_path, text):
    p = tmp_path / "structure.xyz"
    p.write_text(text)
    return p


def test_labelled_atoms_resolve_to_real_elements(tmp_path):
    xyz = (
        "8\n\n"
        "Dy1   0.0 0.0 0.0\n"   # two-letter element + index
        "O1    1.0 0.0 0.0\n"
        "N12   0.0 1.0 0.0\n"   # multi-digit index
        "C3    1.0 1.0 1.0\n"
        "C2_up 2.0 0.0 0.0\n"   # index + textual label suffix
        "Cl1   0.0 2.0 0.0\n"   # two-letter element that must beat "C"
        "Fe    3.0 0.0 0.0\n"   # bare two-letter symbol
        "H     0.0 0.0 3.0\n"   # bare one-letter symbol
    )
    elems = [a.element for a in parse_xyz(_write(tmp_path, xyz)).atoms]
    assert elems == ["Dy", "O", "N", "C", "C", "Cl", "Fe", "H"]


def test_atomic_numbers_still_parse(tmp_path):
    xyz = "2\n\n66 0.0 0.0 0.0\n6 1.0 0.0 0.0\n"
    elems = [a.element for a in parse_xyz(_write(tmp_path, xyz)).atoms]
    assert elems == ["Dy", "C"]
