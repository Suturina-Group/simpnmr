# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration test for the predict relaxation example workflow.

Runs the CLI prediction example and asserts successful execution and expected outputs.
"""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_examples_predict_relax():
    cwd = Path("examples/prediction/predict_relax")
    cmd = ["simpnmr", "predict", "input.yml"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    if result.returncode != 0 and "missing" in result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{result.stderr}")

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected_output = cwd / "prediction_relaxation" / "susceptibility_tensor.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"
