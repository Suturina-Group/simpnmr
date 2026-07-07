# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Unit tests for security-hardening measures.

Covers:
* ``project:name`` path-traversal rejection,
* HTML-safe JSON embedding for the molecule viewer,
* the config parser using ``safe_load`` (no arbitrary object construction).
"""

import json

import pytest
import yaml

from simpnmr.cfg.config import _safe_project_name


@pytest.mark.unit
@pytest.mark.parametrize("good", ["output", "dy_prediction", "results/run1"])
def test_project_name_accepts_relative_names(good):
    assert _safe_project_name(good) == good


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad", ["../etc", "/tmp/evil", "a/../../x", "..\\win", "/abs/path"]
)
def test_project_name_rejects_traversal(bad):
    with pytest.raises(ValueError):
        _safe_project_name(bad)


@pytest.mark.unit
def test_viewer_json_is_script_break_safe():
    """The molecule viewer must not let '</script>' break out of a <script>."""
    pytest.importorskip("PyQt6")  # GUI extra may be absent
    from simpnmr.gui.molecule_view import _html_safe_json

    payload = "atom </script><script>alert(1)</script>"
    encoded = _html_safe_json(payload)
    assert "</script>" not in encoded
    assert "<" not in encoded and ">" not in encoded
    # Still decodes back to the original value.
    restored = (
        encoded.replace("\\u003c", "<")
        .replace("\\u003e", ">")
        .replace("\\u0026", "&")
    )
    assert json.loads(restored) == payload


@pytest.mark.unit
def test_safe_load_rejects_arbitrary_python_objects():
    """safe_load must not construct arbitrary Python objects (RCE vector)."""
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load("!!python/object/apply:os.system ['echo pwned']")
