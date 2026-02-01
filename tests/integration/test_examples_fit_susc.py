# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration test for the fit_susc example workflow.

Runs the CLI example and asserts successful execution and expected outputs.
"""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_examples_fit_susc():
    cwd = Path("examples/fitting/fit_susc")
    cmd = ["simpnmr", "fit_susc", "input.yml"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected_output = cwd / "proton" / "susceptibility_tensor.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"
