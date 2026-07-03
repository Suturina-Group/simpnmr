# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration test for the canonical correlation-time-fitting workflow.

This module exercises the public, user-facing ``fit_corr_time`` example as a
stable happy-path integration case for the CLI pipeline.
"""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_fit_corr_time():
    """Run the canonical ``fit_corr_time`` CLI workflow.

    This integration test executes the public example configuration from the
    examples tree and asserts that the pipeline completes successfully and
    produces the expected diagnostics CSV artifact.
    """
    cwd = Path("examples/FeH/SIMULATIONS/Fit_Correlation_Time")
    if not (cwd / "FeH_fit_corr_time.yml").exists():
        pytest.skip(
            "example config not available (examples/**/SIMULATIONS/ is "
            f"gitignored): {cwd / 'FeH_fit_corr_time.yml'}"
        )
    cmd = ["simpnmr", "--hide", "fit_corr_time", "FeH_fit_corr_time.yml"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    if result.returncode != 0 and "missing" in result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{result.stderr}")

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected_output = cwd / "FeH_Correlation_Time_Fit" / "corr_time_fit_diagnostics.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"
