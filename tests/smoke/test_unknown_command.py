import subprocess


def test_simpnmr_unknown_subcommand():
    result = subprocess.run(
        ["simpnmr", "definitely_not_a_command"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert result.stderr.strip() != ""
