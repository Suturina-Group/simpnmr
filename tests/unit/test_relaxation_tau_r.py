# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Unit tests for solvent-based τ_R estimation in the predict relaxation block.

Covers the config keys (``tau_r_method``/``tau_r_solvent``/``tau_r_eta``/…)
added so that a prediction can estimate the rotational correlation time from
the molecular shape and solvent viscosity instead of requiring an explicit
``tR``.
"""

import numpy as np
import pytest

from simpnmr.cfg.config import PredictConfig
from simpnmr.core.phys.tau_c import get_viscosity, run_ellipsoid


@pytest.mark.unit
def test_relaxation_block_exposes_tau_r_keys():
    """The predict relaxation block accepts the τ_R estimation keys."""
    keys = PredictConfig.KEYWORDS["relaxation"]
    for k in (
        "tau_r_method",
        "tau_r_solvent",
        "tau_r_eta",
        "tau_r_shell",
        "tau_r_sigma",
    ):
        assert k in keys


@pytest.mark.unit
def test_tau_r_method_accepts_valid_models():
    cfg = PredictConfig()
    cfg.relaxation_tau_r_method = "ellipsoid"
    assert cfg.relaxation_tau_r_method == "ellipsoid"
    cfg.relaxation_tau_r_method = "beadshell"
    assert cfg.relaxation_tau_r_method == "beadshell"


@pytest.mark.unit
def test_tau_r_method_rejects_unknown_model():
    cfg = PredictConfig()
    with pytest.raises(ValueError, match="ellipsoid.*beadshell"):
        cfg.relaxation_tau_r_method = "stokes"


@pytest.mark.unit
def test_tau_r_solvent_and_eta_validation():
    cfg = PredictConfig()
    cfg.relaxation_tau_r_solvent = "CDCl3"
    assert cfg.relaxation_tau_r_solvent == "CDCl3"
    with pytest.raises(ValueError, match="positive"):
        cfg.relaxation_tau_r_eta = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        cfg.relaxation_tau_r_shell = -0.5


@pytest.mark.unit
def test_ellipsoid_tau_r_is_positive_for_known_solvent():
    """The hydrodynamic estimate used by predict yields a positive τ_R."""
    eta = get_viscosity("CDCl3", 302.15)
    assert eta > 0
    atoms = [
        ("Fe", np.array([0.0, 0.0, 0.0])),
        ("N", np.array([2.0, 0.0, 0.0])),
        ("C", np.array([0.0, 2.0, 0.0])),
        ("C", np.array([0.0, 0.0, 2.0])),
    ]
    result = run_ellipsoid(atoms, eta, 302.15)
    assert result["tau_iso"] > 0
