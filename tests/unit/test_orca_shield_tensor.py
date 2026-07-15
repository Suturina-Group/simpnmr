# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Regression test: the ORCA output shielding reader keeps the full 3x3 tensor.

ORCA prints Diamagnetic, Paramagnetic and Total shielding tensors per nucleus.
The reader must store the raw (non-symmetric) *Total* tensor, and its keys must
align with the summary-table isotropic/anisotropic values.
"""

import numpy as np
import pytest

from simpnmr.io.qc.backends.orca.shield import read_orca5_output_cs

_BLOCK = """CHEMICAL SHIELDING SUMMARY (ppm)
--------------------------------


  Nucleus  Element    Isotropic     Anisotropy
  -------  -------  ------------   ------------
     51       H           30.579          6.697
     52       H           31.337          5.100

 --------------
 Nucleus  51H :
 --------------

Diamagnetic contribution to the shielding tensor (ppm) :
            41.764         -5.725        17.585
           -27.174         53.156       -11.693
            11.346         -0.452        37.631

Paramagnetic contribution to the shielding tensor (ppm):
           -10.313          5.179       -13.878
            25.818        -23.730        12.164
            -7.297          2.041        -6.770

Total shielding tensor (ppm):
            31.451         -0.546         3.707
            -1.356         29.426         0.471
             4.048          1.588        30.861
"""


@pytest.fixture
def out(tmp_path):
    p = tmp_path / "shield.out"
    p.write_text(_BLOCK)
    return str(p)


def test_reads_total_tensor_raw(out):
    cs_iso, cs_aniso, cs_tensor = read_orca5_output_cs(out)
    assert cs_iso["H51"] == pytest.approx(30.579)
    assert cs_aniso["H51"] == pytest.approx(6.697)

    t = cs_tensor["H51"]
    assert t.shape == (3, 3)
    # the *Total* tensor, not Diamagnetic (41.764) or Paramagnetic (-10.313)
    assert t[0, 0] == pytest.approx(31.451)
    assert t[0, 1] == pytest.approx(-0.546)
    assert t[1, 0] == pytest.approx(-1.356)
    # iso from summary equals trace/3 of the stored tensor
    assert np.trace(t) / 3.0 == pytest.approx(cs_iso["H51"], abs=1e-3)
    # stored raw, not symmetrised
    assert t[0, 1] != t[1, 0]


def test_tensor_keys_align_with_summary(out):
    cs_iso, _, cs_tensor = read_orca5_output_cs(out)
    # H51 has a tensor block; H52 is summary-only here
    assert "H51" in cs_tensor
    assert set(cs_tensor).issubset(set(cs_iso))
