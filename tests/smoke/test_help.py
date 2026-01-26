import subprocess


def test_simpnmr_help():
    result = subprocess.run(["simpnmr", "--help"], capture_output=True)
    assert result.returncode == 0
