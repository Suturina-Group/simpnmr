# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration tests for canonical susceptibility-fitting workflows.

These tests exercise public, user-facing ``fit_susc`` examples that represent
stable happy-path configurations for the main supported fitting setups.
"""

import os
import subprocess
from pathlib import Path

import pytest


def _cli_env(tmp_path: Path) -> dict[str, str]:
    mpl_cache = tmp_path / "matplotlib"
    xdg_cache = tmp_path / "xdg-cache"
    mpl_cache.mkdir()
    xdg_cache.mkdir()

    return {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(mpl_cache),
        "XDG_CACHE_HOME": str(xdg_cache),
    }


@pytest.mark.integration
def test_fit_susc_with_pdip_hfc_isoaxrh_and_permutation_assignment(tmp_path: Path):
    """Run the canonical ``fit_susc`` workflow with PDIP hyperfine input.

    This public happy-path case uses point-dipole hyperfine data from XYZ
    coordinates together with the ``isoaxrh`` susceptibility fit model and
    permutation-based assignment.
    """
    cwd = Path("examples/DyL1/SIMULATIONS/Fitting")
    cmd = ["simpnmr", "--hide", "fit_susc", "DyL1_1H_Fitting.yml"]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, env=_cli_env(tmp_path)
    )

    if result.returncode != 0 and "missing" in result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{result.stderr}")

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected_output = cwd / "DyL1_1H_Fitting" / "susceptibility_tensor.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"


@pytest.mark.integration
def test_fit_susc_with_qc_hfc_split_model_and_hungarian_assignment(tmp_path: Path):
    """Run the canonical variable-temperature ``fit_susc`` workflow.

    This public happy-path case uses DFT-derived hyperfine input together with
    the ``split`` susceptibility fit model and Hungarian-based assignment. It
    exercises the variable-temperature fitting path, where the workflow fits a
    shared susceptibility model across multiple experiments recorded at
    different temperatures.
    """
    cwd = Path("examples/P3FeCl/SIMULATIONS/Fitting")
    cmd = ["simpnmr", "--hide", "fit_susc", "P3FeCl_VT_Fitting.yml"]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, env=_cli_env(tmp_path)
    )

    if result.returncode != 0 and "missing" in result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{result.stderr}")

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected_output = cwd / "P3FeCl_VT_Fitting" / "susceptibility_tensor.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"
