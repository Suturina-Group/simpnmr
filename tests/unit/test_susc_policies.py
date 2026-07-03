# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest

from simpnmr.app.policies.susc import (
    normalize_susc_fit_input_units,
    resolve_susc_fit_variables,
)
from simpnmr.cfg.config import FitSuscConfig
from simpnmr.core.const.physics import KB, MU0, MUB, NA


@pytest.mark.unit
def test_fit_susc_config_input_units_defaults_to_a3():
    config = FitSuscConfig()

    assert config.susc_fit_input_units == "A3"


@pytest.mark.unit
def test_normalize_susc_fit_input_units_accepts_supported_aliases():
    assert normalize_susc_fit_input_units(None) == "A3"
    assert normalize_susc_fit_input_units("") == "A3"
    assert normalize_susc_fit_input_units("Å^3") == "A3"
    assert normalize_susc_fit_input_units("cm3 mol-1") == "cm3 mol-1"
    assert normalize_susc_fit_input_units("reduced") == "reduced"


@pytest.mark.unit
def test_resolve_susc_fit_variables_defaults_to_a3_values():
    fit_vars, fix_vars = resolve_susc_fit_variables(
        raw_variables={
            "iso": ["fit", 0.2],
            "ax": ["fit", -0.1],
            "rh_over_ax": ["fix", 0.25],
        },
        input_units=None,
        temperature=300.0,
        spin=0.5,
    )

    assert fit_vars == {"iso": 0.2, "ax": -0.1}
    assert fix_vars == {"rh_over_ax": 0.25}


@pytest.mark.unit
def test_resolve_susc_fit_variables_converts_cm3_mol_minus_1_to_a3():
    fit_vars, fix_vars = resolve_susc_fit_variables(
        raw_variables={
            "iso": ["fit", 1.0],
            "ax": ["fix", -2.0],
        },
        input_units="cm3 mol-1",
        temperature=300.0,
        spin=1.5,
    )

    scale = 1.0 / (1e-24 * NA / (4.0 * np.pi))

    assert fit_vars["iso"] == pytest.approx(scale)
    assert fix_vars["ax"] == pytest.approx(-2.0 * scale)


@pytest.mark.unit
def test_resolve_susc_fit_variables_converts_reduced_per_temperature():
    fit_vars, fix_vars = resolve_susc_fit_variables(
        raw_variables={
            "iso": ["fit", 0.5],
            "ax": ["fit", -0.2],
            "rh_over_ax": ["fix", 1.0 / 3.0],
        },
        input_units="reduced",
        temperature=200.0,
        spin=1.0,
    )

    curie_prefactor = (MU0 * MUB**2 * 1.0 * (1.0 + 1.0)) / (3.0 * KB) * 1e30
    scale = curie_prefactor / 200.0

    assert fit_vars["iso"] == pytest.approx(0.5 * scale)
    assert fit_vars["ax"] == pytest.approx(-0.2 * scale)
    assert fix_vars["rh_over_ax"] == pytest.approx(1.0 / 3.0)


@pytest.mark.unit
def test_resolve_susc_fit_variables_requires_spin_for_reduced_units():
    with pytest.raises(ValueError, match="requires hyperfine:spin"):
        resolve_susc_fit_variables(
            raw_variables={"iso": ["fit", 0.5]},
            input_units="reduced",
            temperature=200.0,
            spin=None,
        )
