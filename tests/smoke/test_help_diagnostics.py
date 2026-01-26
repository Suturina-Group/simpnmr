import subprocess


def test_simpnmr_help_with_diagnostics():
    result = subprocess.run(["simpnmr", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"simpnmr --help failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
