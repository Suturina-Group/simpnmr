# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Regression test: the Gaussian-16 shielding reader keeps the full 3x3 tensor.

Gaussian prints the (generally non-symmetric) shielding tensor as labelled
components (XX/YX/ZX ...). The reader must store all nine so the anisotropy and
orientation survive, not just the isotropic value.
"""

import numpy as np
import pytest

from simpnmr.io.qc.backends.gaussian.shield import read_gaussian16_log_cs

# Two real nuclei from a Gaussian TPSSh shielding log (Cu, P).
_BLOCK = """ SCF GIAO Magnetic shielding tensor (ppm):
      1  Cu   Isotropic =  1099.6202   Anisotropy =  1263.2798
   XX=   904.8946   YX=   493.4538   ZX=    56.5242
   XY=   499.7929   YY=  1675.5683   ZY=   146.3755
   XZ=    51.8887   YZ=   173.9594   ZZ=   718.3978
   Eigenvalues:   651.1318   705.9221  1941.8067
      2  P    Isotropic =   272.8978   Anisotropy =    60.8571
   XX=   261.7705   YX=    30.9458   ZX=    -4.8015
   XY=    29.8129   YY=   287.2030   ZY=    24.2309
   XZ=    29.3078   YZ=    -1.5534   ZZ=   269.7199
   Eigenvalues:   240.9347   264.2895   313.4692
"""


@pytest.fixture
def log(tmp_path):
    p = tmp_path / "shield.log"
    p.write_text(_BLOCK)
    return str(p)


def test_reads_full_tensor_and_scalars(log):
    cs_iso, cs_aniso, cs_tensor = read_gaussian16_log_cs(log)
    assert set(cs_tensor) == {"Cu1", "P2"}
    assert cs_iso["Cu1"] == pytest.approx(1099.6202)
    assert cs_aniso["Cu1"] == pytest.approx(1263.2798)

    t = cs_tensor["Cu1"]
    assert t.shape == (3, 3)
    # component -> tensor[axis(first), axis(second)]
    assert t[0, 0] == pytest.approx(904.8946)   # XX
    assert t[0, 1] == pytest.approx(499.7929)   # XY
    assert t[1, 0] == pytest.approx(493.4538)   # YX
    assert t[2, 2] == pytest.approx(718.3978)   # ZZ
    # isotropic value is the trace/3 of the stored tensor
    assert np.trace(t) / 3.0 == pytest.approx(cs_iso["Cu1"])
    # Gaussian shielding tensors are generally non-symmetric
    assert t[0, 1] != t[1, 0]


def test_parses_negative_components(log):
    _, _, cs_tensor = read_gaussian16_log_cs(log)
    # P2 has ZX = -4.8015
    assert cs_tensor["P2"][2, 0] == pytest.approx(-4.8015)
