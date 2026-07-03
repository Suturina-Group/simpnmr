# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration tests for canonical predict example workflows.

These tests exercise public, user-facing prediction examples that represent
stable happy-path configurations for the main supported input combinations.
"""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_predict_with_qc_hfc_and_qc_susceptibility():
    """Run the canonical P3FeCl prediction example end-to-end.

    This example covers the public happy-path combination of QC-derived
    hyperfine input with QC-derived susceptibility input, including the
    relaxation-enabled prediction workflow.
    """
    cwd = Path("examples/P3FeCl/SIMULATIONS/Prediction")
    if not (cwd / "P3FeCl_Prediction.yml").exists():
        pytest.skip(
            "example config not available (examples/**/SIMULATIONS/ is "
            f"gitignored): {cwd / 'P3FeCl_Prediction.yml'}"
        )
    cmd = ["simpnmr", "--hide", "predict", "P3FeCl_Prediction.yml"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    if result.returncode != 0 and "missing" in result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{result.stderr}")

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected_output = cwd / "P3FeCl_13C_Prediction" / "susceptibility_tensor.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"


@pytest.mark.integration
def test_predict_with_pdip_hfc_and_csv_susceptibility():
    """Run the canonical DyL1 prediction example end-to-end.

    This example covers the public happy-path combination of point-dipole
    hyperfine input from XYZ coordinates with CSV-based susceptibility input,
    including the relaxation-enabled prediction workflow.
    """
    cwd = Path("examples/DyL1/SIMULATIONS/Prediction")
    if not (cwd / "DyL1_1H_Prediction.yml").exists():
        pytest.skip(
            "example config not available (examples/**/SIMULATIONS/ is "
            f"gitignored): {cwd / 'DyL1_1H_Prediction.yml'}"
        )
    cmd = ["simpnmr", "--hide", "predict", "DyL1_1H_Prediction.yml"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    if result.returncode != 0 and "missing" in result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{result.stderr}")

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected_output = cwd / "DyL1_1H_Prediction" / "susceptibility_tensor.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"
