# simpnmr/io/qc/detect.py
from __future__ import annotations

from typing import Literal

from simpnmr.io.qc.backends.gaussian.detect import is_gaussian_log
from simpnmr.io.qc.backends.orca.detect import is_orca_output

Backend = Literal["orca", "gaussian"]


def detect_backend(file_name: str) -> Backend:
    if is_orca_output(file_name):
        return "orca"
    if is_gaussian_log(file_name):
        return "gaussian"
    raise ValueError(f"Unsupported QC file: {file_name}")
