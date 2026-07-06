# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the SimpNMR desktop application (Windows).

Build (from the repository root, on Windows):

    pyinstaller packaging/windows/simpnmr.spec --noconfirm

Produces a one-folder application under ``dist/SimpNMR/`` whose entry point is
``SimpNMR.exe``. The one-folder layout is deliberate: the app bundles Qt
WebEngine (Chromium) for the 3Dmol.js viewer, and a one-file build would unpack
that ~hundreds of MB to a temp directory on every launch.

This spec is invoked by ``build.ps1`` and by the CI Windows build job.
"""

import os

from PyInstaller.utils.hooks import collect_all

# Resolve the repository root relative to this spec file. PyInstaller executes
# the spec with ``__file__`` available, but SPECPATH is the robust way.
REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))

block_cipher = None

# --- Data files ------------------------------------------------------------
# The 3Dmol.js viewer source is loaded at runtime via
# ``Path(__file__).parent / "3Dmol-min.js"`` in simpnmr/gui/molecule_view.py,
# so it must land next to that module inside the bundle.
datas = [
    (os.path.join(REPO_ROOT, "simpnmr", "gui", "3Dmol-min.js"), "simpnmr/gui"),
]
binaries = []
hiddenimports = [
    # pathos multiprocessing stack (used by the fitting pipelines) — pulled in
    # dynamically, so declare them explicitly.
    "pathos",
    "pathos.pools",
    "pathos.multiprocessing",
    "multiprocess",
    "dill",
]

# Qt WebEngine ships a large set of resources, translations and the QtWebEngine
# process executable; collect_all makes sure they are all bundled.
for pkg in ("PyQt6.QtWebEngineWidgets", "PyQt6.QtWebEngineCore"):
    _d, _b, _h = collect_all(pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h


a = Analysis(
    [os.path.join(REPO_ROOT, "simpnmr", "gui", "app.py")],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SimpNMR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed app — no console window
    disable_windowed_traceback=False,
    icon=os.path.join(SPECPATH, "simpnmr.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SimpNMR",
)
