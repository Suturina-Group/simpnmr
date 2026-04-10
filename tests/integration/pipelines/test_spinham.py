# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration test for the spin-Hamiltonian CLI workflow."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_get_sh_with_isoaxrh_fit_csv_runs_successfully():
    """Run the ``get_sh`` CLI workflow on an ``isoaxrh`` fit result.

    This test exercises the CLI-driven spin-Hamiltonian extraction path using
    an internal susceptibility-fit CSV fixture and verifies successful command
    execution together with creation of the expected output CSV.
    """
    cwd = Path("tests/data/pipelines/spinham")
    cmd = ["simpnmr", "get_sh", "--spin", "2.0", "isoaxrh_fit.csv"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    if result.returncode != 0 and "missing" in result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{result.stderr}")

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )
    expected_output = cwd / "spin_hamiltonian_from_chiT.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"
