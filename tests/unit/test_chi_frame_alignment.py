# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Guards for susceptibility-source <-> hyperfine geometry alignment.

A large Kabsch RMSD means the two structures do not correspond (wrong atom
order or a different molecule), which would silently mis-orient the
susceptibility tensor. Such cases must be rejected, while a matching geometry
must pass through as an identity rotation.
"""

import numpy as np
import pytest

from simpnmr.core.util.transform import get_rotation_and_transformation

_CHI = np.diag([1.0, -0.5, -0.5])
_COORDS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])


def _call(chi_source, dft):
    return get_rotation_and_transformation(
        chi_tensor=_CHI,
        temperature=300.0,
        chi_source_coords=chi_source,
        dft_coords=dft,
    )


@pytest.mark.unit
def test_matching_geometry_gives_identity_rotation():
    rot, _ = _call(_COORDS, _COORDS)
    assert np.allclose(rot, np.eye(3))


@pytest.mark.unit
def test_large_alignment_rmsd_warns_loudly(caplog):
    scrambled = np.array([[0, 0, 0], [5, 5, 5], [-5, 3, 2], [4, -6, 1.0]])
    with caplog.at_level("WARNING"):
        _call(scrambled, _COORDS)
    assert any(
        "do not correspond well" in rec.message for rec in caplog.records
    )


@pytest.mark.unit
def test_different_atom_count_is_rejected():
    with pytest.raises(ValueError, match="different lengths"):
        _call(_COORDS[:3], _COORDS)
