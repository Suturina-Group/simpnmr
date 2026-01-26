import subprocess

import pytest


@pytest.mark.parametrize(
    "subcommand",
    [
        "fit_susc",
        "predict",
        "fit_corr_time",
        "calc_pcs_iso",
    ],
)
def test_simpnmr_subcommand_help(subcommand):
    result = subprocess.run(["simpnmr", subcommand, "--help"], capture_output=True)
    assert result.returncode == 0
