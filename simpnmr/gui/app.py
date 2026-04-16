# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Unified GUI for the predict and fit_susc workflows.

Launch with:
    simpnmr gui
or:
    python -m simpnmr.gui.app
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QColor, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from simpnmr.gui.molecule_view import MoleculeView
    _WEBENGINE_OK = True
except ImportError:
    _WEBENGINE_OK = False

try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as _FigureCanvas,
        NavigationToolbar2QT as _NavigationToolbar,
    )
    from matplotlib.figure import Figure as _Figure
    from simpnmr.io.csv.spec import read_spectrum_with_peaks as _read_spectrum
    _MATPLOTLIB_QT_OK = True
except ImportError:
    _MATPLOTLIB_QT_OK = False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _file_picker(parent, label: str, mode: str = "file") -> QWidget:
    """Row widget: line edit + browse button."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit()
    edit.setPlaceholderText("(not set)")
    btn = QPushButton("Browse…")

    def _browse():
        if mode == "file":
            path, _ = QFileDialog.getOpenFileName(parent, f"Select {label}")
        elif mode == "save":
            path, _ = QFileDialog.getSaveFileName(parent, f"Select {label}")
        else:
            path = QFileDialog.getExistingDirectory(parent, f"Select {label}")
        if path:
            edit.setText(path)

    btn.clicked.connect(_browse)
    layout.addWidget(edit, 1)
    layout.addWidget(btn)
    row._edit = edit
    return row


def _section(title: str) -> QGroupBox:
    box = QGroupBox(title)
    box.setLayout(QFormLayout())
    box.layout().setFieldGrowthPolicy(
        QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
    )
    return box


def _var_row(fix_val: str = "0.01") -> QWidget:
    """fit/fix combo + initial value field."""
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    combo = QComboBox()
    combo.addItems(["fit", "fix"])
    edit = QLineEdit(fix_val)
    edit.setFixedWidth(80)
    lay.addWidget(combo)
    lay.addWidget(edit)
    lay.addStretch()
    row._combo = combo
    row._edit = edit
    return row


# ---------------------------------------------------------------------------
# Config form
# ---------------------------------------------------------------------------

class ConfigForm(QScrollArea):
    """Scrollable form that serialises / deserialises the YAML config.

    Sections are grouped into:
      - shared: visible and enabled in both modes
      - predict-only: enabled only in predict mode
      - fit-only: enabled only in fit_susc mode
    """

    MODE_PREDICT = "predict"
    MODE_FIT = "fit_susc"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setSpacing(8)
        self.setWidget(container)

        self._mode = self.MODE_PREDICT
        self._build_form()
        self._update_mode()

    # ------------------------------------------------------------------
    # Form construction
    # ------------------------------------------------------------------

    def _build_form(self):
        lay = self._layout

        # ── Mode selector ─────────────────────────────────────────────
        mode_box = QGroupBox("Workflow")
        mode_lay = QHBoxLayout(mode_box)
        self._rb_predict = QRadioButton("Predict")
        self._rb_fit = QRadioButton("Fit susceptibility")
        self._rb_predict.setChecked(True)
        self._bg = QButtonGroup(self)
        self._bg.addButton(self._rb_predict)
        self._bg.addButton(self._rb_fit)
        mode_lay.addWidget(self._rb_predict)
        mode_lay.addWidget(self._rb_fit)
        mode_lay.addStretch()
        self._rb_predict.toggled.connect(self._on_mode_changed)
        lay.addWidget(mode_box)

        # ── Hyperfine ─────────────────────────────────────────────────
        sec = _section("Hyperfine")
        self._hfc_file = _file_picker(self, "Hyperfine file")
        self._hfc_method = QComboBox()
        self._hfc_method.addItems(["dft", "pdip", "csv"])
        self._hfc_spin = QLineEdit()
        self._hfc_spin.setPlaceholderText("e.g. 2.5")
        self._hfc_orbit = QLineEdit()
        self._hfc_orbit.setPlaceholderText("e.g. 0  (TM) or 5  (Dy) — optional for spin_only")
        self._hfc_total_J = QLineEdit()
        self._hfc_total_J.setPlaceholderText("e.g. 7.5  (lanthanide) — omit for pure spin Curie")
        self._hfc_centre = QLineEdit()
        self._hfc_centre.setPlaceholderText("x, y, z  (Å)  — required for pdip / relaxation")
        sec.layout().addRow("File:", self._hfc_file)
        sec.layout().addRow("Method:", self._hfc_method)
        sec.layout().addRow("Spin S:", self._hfc_spin)
        sec.layout().addRow("Orbit L:", self._hfc_orbit)
        sec.layout().addRow("Total J:", self._hfc_total_J)
        sec.layout().addRow("Paramagnetic centre:", self._hfc_centre)
        lay.addWidget(sec)

        # ── Chemical labels ───────────────────────────────────────────
        sec = _section("Chemical labels")
        self._chem_labels = _file_picker(self, "Chemical labels CSV")
        sec.layout().addRow("File:", self._chem_labels)
        lay.addWidget(sec)

        # ── Nuclei ────────────────────────────────────────────────────
        sec = _section("Nuclei")
        self._nuclei_include = QLineEdit()
        self._nuclei_include.setPlaceholderText("e.g. H  or  H, C")
        sec.layout().addRow("Include:", self._nuclei_include)
        lay.addWidget(sec)

        # ── Experiment ────────────────────────────────────────────────
        sec = _section("Experiment")
        self._exp_file = _file_picker(self, "Experiment CSV")
        self._exp_spectrum = _file_picker(self, "Spectrum CSV")
        self._exp_reference = QLineEdit()
        self._exp_reference.setPlaceholderText("ppm  (optional)")
        sec.layout().addRow("Peak file:", self._exp_file)
        sec.layout().addRow("Spectrum file:", self._exp_spectrum)
        sec.layout().addRow("Exp. reference (ppm):", self._exp_reference)
        lay.addWidget(sec)

        # ── Diamagnetic ───────────────────────────────────────────────
        sec = _section("Diamagnetic shifts (optional)")
        self._dia_method = QComboBox()
        self._dia_method.addItems(["csv", "dft"])
        self._dia_file = _file_picker(self, "Diamagnetic shifts file")
        sec.layout().addRow("Method:", self._dia_method)
        sec.layout().addRow("File:", self._dia_file)

        # Reference sub-section
        self._dia_ref_method = QComboBox()
        self._dia_ref_method.addItems(["(none)", "dft", "csv", "values"])
        self._dia_ref_file = _file_picker(self, "Reference file")
        # values table: each row is "isotope: value", e.g. "1H: 31.74"
        self._dia_ref_values = QTextEdit()
        self._dia_ref_values.setPlaceholderText(
            "One entry per line, e.g.:\n1H: 31.74\n13C: 188.07"
        )
        self._dia_ref_values.setFixedHeight(64)
        self._dia_ref_method.currentTextChanged.connect(
            self._on_dia_ref_method_changed
        )
        sec.layout().addRow("Ref method:", self._dia_ref_method)
        sec.layout().addRow("Ref file:", self._dia_ref_file)
        sec.layout().addRow("Ref values:", self._dia_ref_values)
        lay.addWidget(sec)
        self._on_dia_ref_method_changed("(none)")

        # ── PREDICT-ONLY: Susceptibility source ───────────────────────
        self._sec_susc_src = _section("Susceptibility source  [predict only]")
        self._susc_method = QComboBox()
        self._susc_method.addItems(
            ["file", "spin_only", "sh", "reduced_chi"]
        )
        self._susc_file = _file_picker(self, "Susceptibility file")
        self._susc_format = QComboBox()
        self._susc_format.addItems(
            ["(auto)", "orca_nev", "orca_cas", "csv", "txt", "molcas"]
        )
        self._susc_temps = QLineEdit()
        self._susc_temps.setPlaceholderText(
            "e.g. 298.15  or  298.15, 310.0"
        )

        # SH parameter fields
        self._susc_sh_widget = QWidget()
        _sh_form = QFormLayout(self._susc_sh_widget)
        _sh_form.setContentsMargins(0, 0, 0, 0)
        self._sh_gx = QLineEdit()
        self._sh_gx.setPlaceholderText("e.g. 2.0023")
        self._sh_gy = QLineEdit()
        self._sh_gy.setPlaceholderText("e.g. 2.0023")
        self._sh_gz = QLineEdit()
        self._sh_gz.setPlaceholderText("e.g. 2.100")
        self._sh_D = QLineEdit()
        self._sh_D.setPlaceholderText("cm⁻¹")
        self._sh_E_over_D = QLineEdit()
        self._sh_E_over_D.setPlaceholderText("0 – 1/3")
        self._sh_alpha = QLineEdit("0.0")
        self._sh_beta = QLineEdit("0.0")
        self._sh_gamma = QLineEdit("0.0")
        _sh_form.addRow("gx:", self._sh_gx)
        _sh_form.addRow("gy:", self._sh_gy)
        _sh_form.addRow("gz:", self._sh_gz)
        _sh_form.addRow("D (cm⁻¹):", self._sh_D)
        _sh_form.addRow("E/D:", self._sh_E_over_D)
        _sh_form.addRow("α (°):", self._sh_alpha)
        _sh_form.addRow("β (°):", self._sh_beta)
        _sh_form.addRow("γ (°):", self._sh_gamma)

        # Reduced-chiT parameter fields
        self._susc_rc_widget = QWidget()
        _rc_form = QFormLayout(self._susc_rc_widget)
        _rc_form.setContentsMargins(0, 0, 0, 0)
        self._rc_iso = QLineEdit()
        self._rc_iso.setPlaceholderText(
            "scalar or comma-separated (one per T)"
        )
        self._rc_ax = QLineEdit()
        self._rc_ax.setPlaceholderText(
            "scalar or comma-separated (one per T)"
        )
        self._rc_rh_over_ax = QLineEdit()
        self._rc_rh_over_ax.setPlaceholderText("0 – 1/3")
        self._rc_alpha = QLineEdit("0.0")
        self._rc_beta = QLineEdit("0.0")
        self._rc_gamma = QLineEdit("0.0")
        _rc_form.addRow("Δχ_iso·T:", self._rc_iso)
        _rc_form.addRow("Δχ_ax·T:", self._rc_ax)
        _rc_form.addRow("Δχ_rh·T / Δχ_ax·T:", self._rc_rh_over_ax)
        _rc_form.addRow("α (°):", self._rc_alpha)
        _rc_form.addRow("β (°):", self._rc_beta)
        _rc_form.addRow("γ (°):", self._rc_gamma)

        self._susc_method.currentTextChanged.connect(
            self._on_susc_method_changed
        )
        self._sec_susc_src.layout().addRow("Method:", self._susc_method)
        self._sec_susc_src.layout().addRow("File:", self._susc_file)
        self._sec_susc_src.layout().addRow("Format:", self._susc_format)
        self._sec_susc_src.layout().addRow(
            "Temperature(s) (K):", self._susc_temps
        )
        self._susc_sh_widget.setVisible(False)
        self._susc_rc_widget.setVisible(False)
        self._sec_susc_src.layout().addRow(self._susc_sh_widget)
        self._sec_susc_src.layout().addRow(self._susc_rc_widget)
        lay.addWidget(self._sec_susc_src)

        # ── FIT-ONLY: Assignment ──────────────────────────────────────
        self._sec_assign = _section("Assignment  [fit only]")
        self._assign_method = QComboBox()
        self._assign_method.addItems(["fixed", "hungarian", "permute"])
        self._assign_area_w = QLineEdit("0.0")
        self._assign_width_w = QLineEdit("0.0")
        self._assign_r1_w = QLineEdit("0.0")
        self._sec_assign.layout().addRow("Method:", self._assign_method)
        self._sec_assign.layout().addRow("Area weight:", self._assign_area_w)
        self._sec_assign.layout().addRow("Width weight:", self._assign_width_w)
        self._sec_assign.layout().addRow("R1 weight:", self._assign_r1_w)
        lay.addWidget(self._sec_assign)

        # ── FIT-ONLY: Susceptibility fit ──────────────────────────────
        self._sec_susc_fit = _section("Susceptibility fit  [fit only]")
        self._susc_type = QComboBox()
        self._susc_type.addItems(["isoaxrh", "full", "split", "eigen", "isoeigen"])
        self._susc_iso = _var_row()
        self._susc_ax = _var_row()
        self._susc_rh = _var_row(fix_val="0.0")
        self._susc_rh_over_ax = _var_row(fix_val="0.0")
        self._average_shifts = QCheckBox("Average equivalent shift groups")
        self._sec_susc_fit.layout().addRow("Fit type:", self._susc_type)
        self._sec_susc_fit.layout().addRow("iso:", self._susc_iso)
        self._sec_susc_fit.layout().addRow("ax:", self._susc_ax)
        self._sec_susc_fit.layout().addRow("rh:", self._susc_rh)
        self._sec_susc_fit.layout().addRow("rh/ax (fixed ratio):", self._susc_rh_over_ax)
        self._sec_susc_fit.layout().addRow("", self._average_shifts)
        lay.addWidget(self._sec_susc_fit)

        # ── Relaxation ────────────────────────────────────────────────
        sec = _section("Relaxation (optional)")
        self._relax_model = QComboBox()
        self._relax_model.addItems(["(none)", "sbm", "curie", "sbm curie"])
        self._relax_temp = QLineEdit()
        self._relax_temp.setPlaceholderText("K  (leave blank = from experiment)")
        self._relax_b0 = QLineEdit()
        self._relax_b0.setPlaceholderText("T  (leave blank = from experiment)")
        self._relax_T1e = QLineEdit()
        self._relax_T1e.setPlaceholderText("s  e.g. 0.2e-12")
        self._relax_T2e = QLineEdit()
        self._relax_T2e.setPlaceholderText("s  e.g. 0.2e-12")
        self._relax_tR = QLineEdit()
        self._relax_tR.setPlaceholderText("s  e.g. 140e-12")
        self._relax_min_lw = QLineEdit()
        self._relax_min_lw.setPlaceholderText("Hz  (optional floor, default 0)")
        sec.layout().addRow("Model:", self._relax_model)
        sec.layout().addRow("Temperature (K):", self._relax_temp)
        sec.layout().addRow("B₀ (T):", self._relax_b0)
        sec.layout().addRow("T1e:", self._relax_T1e)
        sec.layout().addRow("T2e:", self._relax_T2e)
        sec.layout().addRow("τR:", self._relax_tR)
        sec.layout().addRow("Min linewidth (Hz):", self._relax_min_lw)
        lay.addWidget(sec)

        # ── FIT-ONLY: Fit relaxation (τR heatmaps) ────────────────────
        self._sec_fit_relax = _section("Fit relaxation / τR heatmap  [fit only]")
        self._tau_r_method = QComboBox()
        self._tau_r_method.addItems(["(none)", "ellipsoid", "beadshell"])
        self._tau_r_solvent = QComboBox()
        self._tau_r_solvent.addItems([
            "(none)", "D2O", "CDCl3", "CD2Cl2", "DMSO-d6", "CD3OD",
            "acetone-d6", "C6D6", "toluene-d8", "THF-d8", "CD3CN",
            "DMF-d7", "pyridine-d5", "water", "methanol", "ethanol",
        ])
        self._tau_r_eta = QLineEdit()
        self._tau_r_eta.setPlaceholderText("Pa·s  (overrides solvent)")
        self._tau_r_fixed = QLineEdit()
        self._tau_r_fixed.setPlaceholderText("s  (manual overlay)")
        self._sec_fit_relax.layout().addRow("τR method:", self._tau_r_method)
        self._sec_fit_relax.layout().addRow("Solvent:", self._tau_r_solvent)
        self._sec_fit_relax.layout().addRow("η (Pa·s):", self._tau_r_eta)
        self._sec_fit_relax.layout().addRow("τR fixed (s):", self._tau_r_fixed)
        lay.addWidget(self._sec_fit_relax)

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _on_mode_changed(self, predict_checked: bool):
        self._mode = self.MODE_PREDICT if predict_checked else self.MODE_FIT
        self._update_mode()

    def _update_mode(self):
        is_predict = self._mode == self.MODE_PREDICT
        self._sec_susc_src.setEnabled(is_predict)
        self._sec_assign.setEnabled(not is_predict)
        self._sec_susc_fit.setEnabled(not is_predict)
        self._sec_fit_relax.setEnabled(not is_predict)
        # Susceptibility file sub-fields depend on method when in predict mode
        if is_predict:
            self._on_susc_method_changed(self._susc_method.currentText())

    def _on_susc_method_changed(self, method: str):
        is_file = method == "file"
        self._susc_file.setEnabled(is_file)
        self._susc_format.setEnabled(is_file)
        self._susc_sh_widget.setVisible(method == "sh")
        self._susc_rc_widget.setVisible(method == "reduced_chi")

    def _on_dia_ref_method_changed(self, method: str):
        is_none = method == "(none)"
        is_values = method == "values"
        self._dia_ref_file.setEnabled(not is_none and not is_values)
        self._dia_ref_values.setEnabled(is_values)

    # ------------------------------------------------------------------
    # Public mode accessors
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
        if mode == self.MODE_FIT:
            self._rb_fit.setChecked(True)
        else:
            self._rb_predict.setChecked(True)

    # ------------------------------------------------------------------
    # Serialise → YAML dict
    # ------------------------------------------------------------------

    def to_yaml_dict(self, project_name: str | None = None) -> dict:
        d: dict = {}
        is_predict = self._mode == self.MODE_PREDICT

        if project_name:
            d["project"] = {"name": project_name}

        # Hyperfine
        hfc_file = self._hfc_file._edit.text().strip()
        if hfc_file:
            hfc: dict = {
                "method": self._hfc_method.currentText(),
                "file": hfc_file,
            }
            for key, field in [
                ("spin", self._hfc_spin),
                ("orbit", self._hfc_orbit),
                ("total_momentum_J", self._hfc_total_J),
            ]:
                val = field.text().strip()
                if val:
                    hfc[key] = float(val)
            centre = self._hfc_centre.text().strip()
            if centre:
                hfc["paramagnetic_centre"] = [
                    float(x) for x in centre.replace(",", " ").split()
                ]
            d["hyperfine"] = hfc

        # Chem labels
        cl_file = self._chem_labels._edit.text().strip()
        if cl_file:
            d["chem_labels"] = {"file": cl_file}

        # Nuclei
        nuclei = self._nuclei_include.text().strip()
        if nuclei:
            items = [x.strip() for x in nuclei.split(",") if x.strip()]
            d["nuclei"] = {"include": items[0] if len(items) == 1 else items}

        # Experiment
        exp_file = self._exp_file._edit.text().strip()
        if exp_file:
            exp: dict = {"files": exp_file}
            spec = self._exp_spectrum._edit.text().strip()
            if spec:
                exp["spectrum_files"] = spec
            ref = self._exp_reference.text().strip()
            if ref:
                exp["exp_reference"] = float(ref)
            d["experiment"] = exp

        # Diamagnetic
        dia_file = self._dia_file._edit.text().strip()
        if dia_file:
            d["diamagnetic"] = {
                "method": self._dia_method.currentText(),
                "file": dia_file,
            }
            ref_method = self._dia_ref_method.currentText()
            if ref_method != "(none)":
                ref_block: dict = {"method": ref_method}
                if ref_method == "values":
                    vals: dict = {}
                    for line in self._dia_ref_values.toPlainText().splitlines():
                        line = line.strip()
                        if ":" in line:
                            iso, val = line.split(":", 1)
                            try:
                                vals[iso.strip()] = float(val.strip())
                            except ValueError:
                                pass
                    if vals:
                        ref_block["values"] = vals
                else:
                    rf = self._dia_ref_file._edit.text().strip()
                    if rf:
                        ref_block["file"] = rf
                d["diamagnetic_ref"] = ref_block

        # Predict-only: Susceptibility source
        if is_predict:
            susc_method = self._susc_method.currentText()
            temps_raw = self._susc_temps.text().strip()
            temps = (
                [
                    float(t.strip())
                    for t in temps_raw.replace(",", " ").split()
                    if t.strip()
                ]
                if temps_raw else []
            )
            temps_val = temps[0] if len(temps) == 1 else temps

            def _floats(text):
                parts = [
                    p.strip()
                    for p in text.replace(",", " ").split()
                    if p.strip()
                ]
                vals = [float(p) for p in parts]
                return vals[0] if len(vals) == 1 else vals

            if susc_method == "spin_only":
                susc_block: dict = {"method": "spin_only"}
                if temps:
                    susc_block["temperatures"] = temps_val
            elif susc_method == "sh":
                susc_block = {"method": "sh"}
                if temps:
                    susc_block["temperatures"] = temps_val
                sh_params = {}
                for key, widget in [
                    ("gx", self._sh_gx), ("gy", self._sh_gy),
                    ("gz", self._sh_gz), ("D", self._sh_D),
                    ("alpha", self._sh_alpha),
                    ("beta", self._sh_beta), ("gamma", self._sh_gamma),
                ]:
                    v = widget.text().strip()
                    if v:
                        sh_params[key] = float(v)
                ed_text = self._sh_E_over_D.text().strip()
                if ed_text:
                    ed = float(ed_text)
                    if not (0.0 <= ed <= 1.0 / 3.0):
                        QMessageBox.warning(
                            self, "Invalid E/D",
                            f"E/D = {ed:.4f} is outside [0, 1/3]. "
                            "Please enter a value between 0 and 0.3333.",
                        )
                        return None
                    sh_params["E_over_D"] = ed
                if sh_params:
                    susc_block["sh"] = sh_params
            elif susc_method == "reduced_chi":
                susc_block = {"method": "reduced_chi"}
                if temps:
                    susc_block["temperatures"] = temps_val
                rc_params = {}
                for key, widget in [
                    ("chi_iso_T", self._rc_iso),
                    ("chi_ax_T", self._rc_ax),
                    ("alpha", self._rc_alpha),
                    ("beta", self._rc_beta),
                    ("gamma", self._rc_gamma),
                ]:
                    v = widget.text().strip()
                    if v:
                        rc_params[key] = _floats(v)
                roa_text = self._rc_rh_over_ax.text().strip()
                if roa_text:
                    roa = float(roa_text)
                    if not (0.0 <= roa <= 1.0 / 3.0):
                        QMessageBox.warning(
                            self, "Invalid Δχ_rh/Δχ_ax",
                            f"Δχ_rh/Δχ_ax = {roa:.4f} is outside [0, 1/3]. "
                            "Please enter a value between 0 and 0.3333.",
                        )
                        return None
                    rc_params["rh_over_ax"] = roa
                if rc_params:
                    susc_block["reduced_chi"] = rc_params
            else:
                susc_block = {}
                sf = self._susc_file._edit.text().strip()
                if sf:
                    susc_block["file"] = sf
                fmt = self._susc_format.currentText()
                if fmt != "(auto)":
                    susc_block["format"] = fmt
                if temps:
                    susc_block["temperatures"] = temps_val
            if susc_block:
                d["susceptibility"] = susc_block

        # Fit-only: Assignment
        if not is_predict:
            d["assignment"] = {"method": self._assign_method.currentText()}
            for key, field in [
                ("area_weight", self._assign_area_w),
                ("width_weight", self._assign_width_w),
                ("r1_weight", self._assign_r1_w),
            ]:
                val = field.text().strip()
                if val and float(val) != 0.0:
                    d["assignment"][key] = float(val)

        # Fit-only: Susceptibility fit
        if not is_predict:
            def _var(row):
                return [row._combo.currentText(), float(row._edit.text() or "0")]

            susc_type = self._susc_type.currentText()
            variables: dict = {
                "iso": _var(self._susc_iso),
                "ax": _var(self._susc_ax),
            }
            if susc_type == "isoaxrh":
                variables["rh_over_ax"] = _var(self._susc_rh_over_ax)
            else:
                variables["rh"] = _var(self._susc_rh)
            susc_fit: dict = {"type": susc_type, "variables": variables}
            if self._average_shifts.isChecked():
                susc_fit["average_shifts"] = "all"
            d["susc_fit"] = susc_fit

        # Relaxation (shared)
        model = self._relax_model.currentText()
        if model != "(none)":
            relax: dict = {"model": model}
            for key, field in [
                ("temperature", self._relax_temp),
                ("magnetic_field_tesla", self._relax_b0),
                ("T1e", self._relax_T1e),
                ("T2e", self._relax_T2e),
                ("tR", self._relax_tR),
                ("min_linewidth_hz", self._relax_min_lw),
            ]:
                val = field.text().strip()
                if val:
                    relax[key] = float(val)
            d["relaxation"] = relax

        # Fit-only: Fit relaxation / tau_r heatmap
        if not is_predict:
            tau_r_method = self._tau_r_method.currentText()
            if tau_r_method != "(none)":
                fit_relax: dict = {"tau_r_method": tau_r_method}
                solvent = self._tau_r_solvent.currentText()
                if solvent != "(none)":
                    fit_relax["tau_r_solvent"] = solvent
                eta = self._tau_r_eta.text().strip()
                if eta:
                    fit_relax["tau_r_eta"] = float(eta)
                fixed = self._tau_r_fixed.text().strip()
                if fixed:
                    fit_relax["tau_r_fixed"] = float(fixed)
                d["fit_relaxation"] = fit_relax

        return d

    # ------------------------------------------------------------------
    # Deserialise ← YAML dict
    # ------------------------------------------------------------------

    def from_yaml_dict(self, d: dict):
        # Auto-detect mode
        if "susc_fit" in d:
            self.set_mode(self.MODE_FIT)
        elif "susceptibility" in d or not d:
            self.set_mode(self.MODE_PREDICT)

        hfc = d.get("hyperfine", {})
        self._hfc_file._edit.setText(str(hfc.get("file", "")))
        idx = self._hfc_method.findText(str(hfc.get("method", "dft")))
        if idx >= 0:
            self._hfc_method.setCurrentIndex(idx)
        for key, field in [
            ("spin", self._hfc_spin),
            ("orbit", self._hfc_orbit),
            ("total_momentum_J", self._hfc_total_J),
        ]:
            val = hfc.get(key, "")
            field.setText(str(val) if val != "" else "")
        centre = hfc.get("paramagnetic_centre")
        if centre:
            self._hfc_centre.setText(", ".join(str(x) for x in centre))

        cl = d.get("chem_labels", {})
        self._chem_labels._edit.setText(str(cl.get("file", "")))

        nuclei = d.get("nuclei", {})
        inc = nuclei.get("include", [])
        self._nuclei_include.setText(
            ", ".join(inc) if isinstance(inc, list) else str(inc)
        )

        exp = d.get("experiment", {})
        files = exp.get("files", "")
        self._exp_file._edit.setText(
            str(files[0]) if isinstance(files, list) else str(files)
        )
        spec = exp.get("spectrum_files", "")
        self._exp_spectrum._edit.setText(
            str(spec[0]) if isinstance(spec, list) else str(spec)
        )
        self._exp_reference.setText(str(exp.get("exp_reference", "")))

        dia = d.get("diamagnetic", {})
        self._dia_file._edit.setText(str(dia.get("file", "")))
        dia_m = str(dia.get("method", "csv") or "csv")
        idx = self._dia_method.findText(dia_m)
        if idx >= 0:
            self._dia_method.setCurrentIndex(idx)

        dia_ref = d.get("diamagnetic_ref", {})
        ref_m = str(dia_ref.get("method", "(none)") or "(none)")
        idx = self._dia_ref_method.findText(ref_m)
        self._dia_ref_method.setCurrentIndex(idx if idx >= 0 else 0)
        ref_file = dia_ref.get("file", "")
        self._dia_ref_file._edit.setText(str(ref_file) if ref_file else "")
        ref_vals = dia_ref.get("values", {})
        if ref_vals:
            self._dia_ref_values.setPlainText(
                "\n".join(f"{iso}: {val}" for iso, val in ref_vals.items())
            )
        else:
            self._dia_ref_values.setPlainText("")

        # Susceptibility source (predict)
        susc = d.get("susceptibility", {})
        if susc:
            method = str(susc.get("method", "file") or "file")
            idx = self._susc_method.findText(method)
            if idx >= 0:
                self._susc_method.setCurrentIndex(idx)
            self._susc_file._edit.setText(str(susc.get("file", "")))
            fmt = str(susc.get("format", "(auto)") or "(auto)")
            idx = self._susc_format.findText(fmt)
            if idx >= 0:
                self._susc_format.setCurrentIndex(idx)
            temps = susc.get("temperatures", "")
            if isinstance(temps, list):
                self._susc_temps.setText(", ".join(str(t) for t in temps))
            elif temps:
                self._susc_temps.setText(str(temps))
            sh = susc.get("sh", {})
            for key, widget in [
                ("gx", self._sh_gx), ("gy", self._sh_gy),
                ("gz", self._sh_gz), ("D", self._sh_D),
                ("alpha", self._sh_alpha),
                ("beta", self._sh_beta), ("gamma", self._sh_gamma),
            ]:
                if key in sh:
                    widget.setText(str(sh[key]))
            if "E_over_D" in sh:
                self._sh_E_over_D.setText(str(sh["E_over_D"]))
            rc = susc.get("reduced_chi", {})
            for key, widget in [
                ("chi_iso_T", self._rc_iso),
                ("chi_ax_T", self._rc_ax),
                ("alpha", self._rc_alpha),
                ("beta", self._rc_beta),
                ("gamma", self._rc_gamma),
            ]:
                if key in rc:
                    v = rc[key]
                    if isinstance(v, list):
                        widget.setText(", ".join(str(x) for x in v))
                    else:
                        widget.setText(str(v))
            if "rh_over_ax" in rc:
                self._rc_rh_over_ax.setText(str(rc["rh_over_ax"]))

        # Assignment (fit)
        assign = d.get("assignment", {})
        idx = self._assign_method.findText(str(assign.get("method", "fixed")))
        if idx >= 0:
            self._assign_method.setCurrentIndex(idx)
        self._assign_area_w.setText(str(assign.get("area_weight", "0.0")))
        self._assign_width_w.setText(str(assign.get("width_weight", "0.0")))
        self._assign_r1_w.setText(str(assign.get("r1_weight", "0.0")))

        # Susceptibility fit (fit)
        susc_fit = d.get("susc_fit", {})
        idx = self._susc_type.findText(str(susc_fit.get("type", "isoaxrh")))
        if idx >= 0:
            self._susc_type.setCurrentIndex(idx)
        variables = susc_fit.get("variables", {})
        for key, row in [
            ("iso", self._susc_iso), ("ax", self._susc_ax),
            ("rh", self._susc_rh), ("rh_over_ax", self._susc_rh_over_ax),
        ]:
            val = variables.get(key)
            if val and isinstance(val, list) and len(val) == 2:
                idx = row._combo.findText(str(val[0]))
                if idx >= 0:
                    row._combo.setCurrentIndex(idx)
                row._edit.setText(str(val[1]))
        self._average_shifts.setChecked(bool(susc_fit.get("average_shifts")))

        # Relaxation (shared)
        relax = d.get("relaxation", {})
        model = str(relax.get("model", "(none)") or "(none)")
        idx = self._relax_model.findText(model)
        self._relax_model.setCurrentIndex(idx if idx >= 0 else 0)
        self._relax_temp.setText(str(relax.get("temperature", "")))
        self._relax_b0.setText(str(relax.get("magnetic_field_tesla", "")))
        self._relax_T1e.setText(str(relax.get("T1e", "")))
        self._relax_T2e.setText(str(relax.get("T2e", "")))
        self._relax_tR.setText(str(relax.get("tR", "")))
        self._relax_min_lw.setText(str(relax.get("min_linewidth_hz", "")))

        # Fit relaxation (fit)
        fit_relax = d.get("fit_relaxation", {})
        method = str(fit_relax.get("tau_r_method", "(none)"))
        idx = self._tau_r_method.findText(method)
        self._tau_r_method.setCurrentIndex(idx if idx >= 0 else 0)
        solvent = str(fit_relax.get("tau_r_solvent", "(none)"))
        idx = self._tau_r_solvent.findText(solvent)
        self._tau_r_solvent.setCurrentIndex(idx if idx >= 0 else 0)
        self._tau_r_eta.setText(str(fit_relax.get("tau_r_eta", "")))
        self._tau_r_fixed.setText(str(fit_relax.get("tau_r_fixed", "")))


# ---------------------------------------------------------------------------
# Log panel
# ---------------------------------------------------------------------------

class LogPanel(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Menlo", 10))
        self.setMaximumHeight(180)

    def append_line(self, text: str, color: str = "#cccccc"):
        self.setTextColor(QColor(color))
        self.append(text.rstrip())
        self.moveCursor(QTextCursor.MoveOperation.End)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class SimpNMRWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SimpNMR")
        self.resize(1300, 860)

        self._process: QProcess | None = None
        self._yaml_path: Path | None = None

        self._build_ui()

    def _build_ui(self):
        # ── Toolbar ───────────────────────────────────────────────────
        toolbar_widget = QWidget()
        toolbar = QHBoxLayout(toolbar_widget)
        toolbar.setContentsMargins(6, 4, 6, 4)

        btn_new = QPushButton("New")
        btn_open = QPushButton("Open YAML…")
        btn_save = QPushButton("Save YAML")
        btn_save_as = QPushButton("Save As…")
        self._btn_run = QPushButton("▶  Run")
        self._btn_run.setStyleSheet(
            "QPushButton { background: #2a7; color: white; font-weight: bold;"
            " border-radius: 4px; padding: 4px 16px; }"
            "QPushButton:disabled { background: #555; }"
        )
        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(
            "QPushButton { background: #a33; color: white; font-weight: bold;"
            " border-radius: 4px; padding: 4px 16px; }"
            "QPushButton:disabled { background: #555; }"
        )
        self._hide_cb = QCheckBox("--hide (no interactive plots)")
        self._hide_cb.setChecked(True)

        for w in [btn_new, btn_open, btn_save, btn_save_as,
                  QLabel(" "), self._btn_run, self._btn_stop,
                  QLabel(" "), self._hide_cb]:
            toolbar.addWidget(w)
        toolbar.addStretch()

        btn_new.clicked.connect(self._new)
        btn_open.clicked.connect(self._open_yaml)
        btn_save.clicked.connect(self._save_yaml)
        btn_save_as.clicked.connect(self._save_yaml_as)
        self._btn_run.clicked.connect(self._run)
        self._btn_stop.clicked.connect(self._stop)

        # ── Left column: form + log ───────────────────────────────────
        self._form = ConfigForm()
        # Keep title in sync whenever mode radio changes
        self._form._rb_predict.toggled.connect(lambda _: self._update_title())
        self._form._rb_fit.toggled.connect(lambda _: self._update_title())
        self._form.setMinimumWidth(400)
        self._form.setMaximumWidth(540)

        self._log = LogPanel()

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        left_layout.addWidget(self._form, 1)
        left_layout.addWidget(QLabel("Log output:"))
        left_layout.addWidget(self._log)

        # ── Right panel: molecule viewer + placeholder ────────────────
        right_widget = QSplitter(Qt.Orientation.Vertical)

        if _WEBENGINE_OK:
            mol_container = QWidget()
            mol_layout = QVBoxLayout(mol_container)
            mol_layout.setContentsMargins(0, 0, 0, 0)
            mol_layout.setSpacing(2)

            mol_toolbar = QWidget()
            mol_tb_lay = QHBoxLayout(mol_toolbar)
            mol_tb_lay.setContentsMargins(4, 2, 4, 2)
            self._btn_load_mol = QPushButton("Load structure…")
            self._lbl_mol_path = QLabel("No structure loaded")
            self._lbl_mol_path.setStyleSheet("color: #888; font-size: 11px;")
            self._mol_elem_combo = QComboBox()
            self._mol_elem_combo.addItem("All")
            self._mol_elem_combo.setToolTip("Show labels for selected element only")
            self._mol_elem_combo.setEnabled(False)
            self._mol_elem_combo.currentTextChanged.connect(self._on_mol_elem_filter)

            self._iso_check = QCheckBox("PCS iso:")
            self._iso_check.setChecked(True)
            self._iso_check.setEnabled(False)
            self._iso_check.setToolTip("Show / hide PCS isosurface")
            self._iso_spin = QDoubleSpinBox()
            self._iso_spin.setRange(0.0, 100000.0)
            self._iso_spin.setValue(100.0)
            self._iso_spin.setSuffix(" ppm")
            self._iso_spin.setSingleStep(10.0)
            self._iso_spin.setDecimals(1)
            self._iso_spin.setFixedWidth(95)
            self._iso_spin.setToolTip(
                "PCS isosurface isovalue (ppm). Blue = positive, red = negative."
            )
            self._iso_spin.setEnabled(False)
            self._iso_check.toggled.connect(self._on_iso_toggled)
            self._iso_spin.valueChanged.connect(self._on_isovalue_changed)

            mol_tb_lay.addWidget(self._btn_load_mol)
            mol_tb_lay.addWidget(self._lbl_mol_path, 1)
            mol_tb_lay.addWidget(self._mol_elem_combo)
            mol_tb_lay.addWidget(self._iso_check)
            mol_tb_lay.addWidget(self._iso_spin)
            self._btn_load_mol.clicked.connect(self._load_structure_manual)

            self._mol_view = MoleculeView()
            mol_layout.addWidget(mol_toolbar)
            mol_layout.addWidget(self._mol_view, 1)
            right_widget.addWidget(mol_container)
        else:
            self._mol_view = None
            self._mol_elem_combo = None
            self._iso_check = None
            self._iso_spin = None
            lbl = QLabel("Install PyQt6-WebEngine to enable molecule viewer")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#888; background:#2a2a2a;")
            right_widget.addWidget(lbl)

        if _MATPLOTLIB_QT_OK:
            self._spectrum_panel = _SpectrumPanel()
            spec_container = QWidget()
            spec_lay = QVBoxLayout(spec_container)
            spec_lay.setContentsMargins(0, 0, 0, 0)
            spec_lay.setSpacing(0)
            spec_toolbar = _NavigationToolbar(
                self._spectrum_panel, spec_container
            )
            spec_lay.addWidget(spec_toolbar)
            spec_lay.addWidget(self._spectrum_panel)
            right_widget.addWidget(spec_container)
        else:
            self._spectrum_panel = None
            right_widget.addWidget(_make_placeholder())

        right_widget.setStretchFactor(0, 3)   # molecule viewer
        right_widget.setStretchFactor(1, 1)   # spectrum panel

        # ── Main splitter ─────────────────────────────────────────────
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(1, 3)

        # ── Assemble ──────────────────────────────────────────────────
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(toolbar_widget)
        root.addWidget(main_splitter, 1)
        self.setCentralWidget(central)

    # ── File actions ──────────────────────────────────────────────────

    def _new(self):
        self._form.from_yaml_dict({})
        self._yaml_path = None
        self._update_title()

    def _open_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open YAML", "", "YAML files (*.yml *.yaml)"
        )
        if not path:
            return
        self._yaml_path = Path(path)
        try:
            with open(path) as f:
                d = yaml.safe_load(f)
            self._form.from_yaml_dict(d or {})
            self._update_title()
            self._log.append_line(f"Loaded {path}", "#88cc88")
        except Exception as e:
            QMessageBox.critical(self, "Error loading YAML", str(e))

    def _save_yaml(self):
        if self._yaml_path is None:
            self._save_yaml_as()
            return
        self._write_yaml(self._yaml_path)

    def _save_yaml_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save YAML", "", "YAML files (*.yml *.yaml)"
        )
        if not path:
            return
        self._yaml_path = Path(path)
        self._write_yaml(self._yaml_path)

    def _write_yaml(self, path: Path):
        try:
            d = self._form.to_yaml_dict(project_name=self._project_name())
            with open(path, "w") as f:
                yaml.dump(d, f, default_flow_style=False, allow_unicode=True)
            self._update_title()
            self._log.append_line(f"Saved {path}", "#88cc88")
        except Exception as e:
            QMessageBox.critical(self, "Error saving YAML", str(e))

    def _project_name(self) -> str | None:
        """Derive project folder name: <yaml_stem>_predict or <yaml_stem>_fit."""
        if self._yaml_path is None:
            return None
        suffix = "predict" if self._form.mode == ConfigForm.MODE_PREDICT else "fit"
        return f"{self._yaml_path.stem}_{suffix}"

    def _update_title(self):
        mode_label = (
            "predict" if self._form.mode == ConfigForm.MODE_PREDICT else "fit_susc"
        )
        name = self._yaml_path.name if self._yaml_path else "new"
        proj = self._project_name()
        proj_hint = f"  →  {proj}/" if proj else ""
        self.setWindowTitle(f"SimpNMR — {mode_label}  [{name}]{proj_hint}")

    # ── Run / stop ────────────────────────────────────────────────────

    def _run(self):
        if self._yaml_path is None:
            self._save_yaml_as()
            if self._yaml_path is None:
                return
        else:
            self._write_yaml(self._yaml_path)

        mode = self._form.mode
        work_dir = str(self._yaml_path.parent)

        self._log.clear()
        self._log.append_line(
            f"Running: simpnmr {mode} {self._yaml_path.name}", "#aaaaff"
        )

        hide = self._hide_cb.isChecked()
        entry = "from simpnmr.cli.main import interface; interface()"
        if hide:
            cmd = [sys.executable, "-c", entry, "--hide", mode, str(self._yaml_path)]
        else:
            cmd = [sys.executable, "-c", entry, mode, str(self._yaml_path)]

        self._process = QProcess(self)
        self._process.setWorkingDirectory(work_dir)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)
        self._process.start(cmd[0], cmd[1:])

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)

    def _stop(self):
        if self._process:
            self._process.kill()

    def _on_stdout(self):
        data = self._process.readAllStandardOutput().data().decode(errors="replace")
        for line in data.splitlines():
            self._log.append_line(line, "#cccccc")

    def _on_stderr(self):
        data = self._process.readAllStandardError().data().decode(errors="replace")
        for line in data.splitlines():
            color = (
                "#ff8888" if "ERROR" in line or "Traceback" in line
                else "#ffcc66"
            )
            self._log.append_line(line, color)

    def _on_finished(self, exit_code: int, _):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._update_title()
        if exit_code == 0:
            self._log.append_line("✓ Finished successfully", "#88cc88")
            self._try_load_structure()
            self._try_load_spectrum()
        else:
            self._log.append_line(f"✗ Exited with code {exit_code}", "#ff8888")

    def _try_load_structure(self):
        """Auto-load chemcraft_structure.xyz (or structure.xyz) after a run."""
        if self._mol_view is None or self._yaml_path is None:
            return
        proj = self._project_name()
        if proj is None:
            return
        work = self._yaml_path.parent / proj
        # Find the first available cube file to show PCS isosurface
        cube_paths = sorted(work.glob("pcs_isosurf_*.cube"))
        cube_data = Path(cube_paths[0]).read_text(encoding="utf-8") \
            if cube_paths else None
        for candidate in ["chemcraft_structure.xyz", "structure.xyz"]:
            path = work / candidate
            if path.exists():
                self._load_structure(path, cube_data=cube_data)
                return

    def _try_load_spectrum(self):
        """Load shift_vs_intensity CSVs from the output dir after a run."""
        if self._spectrum_panel is None or self._yaml_path is None:
            return
        proj = self._project_name()
        if proj is None:
            return
        self._spectrum_panel.load_output_dir(self._yaml_path.parent / proj)

    def _load_structure(self, path: Path, cube_data: str | None = None):
        if self._mol_view is None:
            return
        try:
            isoval_ppm = self._iso_spin.value() if self._iso_spin else 1.0
            mol = self._mol_view.load_xyz_file(
                path,
                cube_data=cube_data,
                default_isoval=isoval_ppm * 1000.0,  # ppm → ppb
            )
            self._lbl_mol_path.setText(path.name)
            self._log.append_line(f"Structure loaded: {path.name}", "#88cc88")
            self._populate_elem_combo(mol)
            has_cube = cube_data is not None
            if self._iso_check is not None:
                self._iso_check.setEnabled(has_cube)
            if self._iso_spin is not None:
                self._iso_spin.setEnabled(has_cube)
        except Exception as e:
            self._log.append_line(f"Could not load structure: {e}", "#ffcc66")

    def _on_iso_toggled(self, checked: bool) -> None:
        if self._mol_view is None:
            return
        if checked and self._iso_spin is not None:
            self._mol_view.set_isosurface_isovalue(
                self._iso_spin.value() * 1000.0
            )
        else:
            self._mol_view.set_isosurface_isovalue(0.0)

    def _on_isovalue_changed(self, value: float) -> None:
        if self._mol_view is None:
            return
        checked = self._iso_check.isChecked() if self._iso_check else True
        if checked:
            self._mol_view.set_isosurface_isovalue(value * 1000.0)

    def _populate_elem_combo(self, mol) -> None:
        if self._mol_elem_combo is None:
            return
        self._mol_elem_combo.blockSignals(True)
        self._mol_elem_combo.clear()
        self._mol_elem_combo.addItem("All")
        self._mol_elem_combo.addItem("None")
        labelled_elems = sorted({a.element for a in mol.atoms if a.label})
        for elem in labelled_elems:
            self._mol_elem_combo.addItem(elem)
        self._mol_elem_combo.setEnabled(bool(labelled_elems))
        self._mol_elem_combo.blockSignals(False)

    def _on_mol_elem_filter(self, text: str) -> None:
        if self._mol_view is None:
            return
        if text == "None":
            self._mol_view.run_js("viewer.removeAllLabels(); viewer.render();")
        else:
            elem = "all" if text == "All" else text
            self._mol_view.run_js(f"filterLabels({json.dumps(elem)})")

    def _load_structure_manual(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open XYZ structure", "", "XYZ files (*.xyz)"
        )
        if path:
            self._load_structure(Path(path))

    def closeEvent(self, event):
        if (self._process
                and self._process.state() != QProcess.ProcessState.NotRunning):
            self._process.kill()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Spectrum panel
# ---------------------------------------------------------------------------

def _make_placeholder() -> QWidget:
    w = QWidget()
    w.setStyleSheet("background: #2a2a2a;")
    return w


if _MATPLOTLIB_QT_OK:
    class _SpectrumPanel(_FigureCanvas):
        """Embedded matplotlib canvas showing predicted spectra."""

        def __init__(self, parent=None):
            self._fig = _Figure(tight_layout=True)
            super().__init__(self._fig)
            self.setParent(parent)
            self.setMinimumHeight(180)
            self._show_placeholder()

        def _show_placeholder(self):
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Spectrum will appear here after run",
                    ha='center', va='center', transform=ax.transAxes,
                    color='gray', fontsize=10)
            ax.set_axis_off()
            self.draw()

        def load_output_dir(self, out_dir: Path) -> None:
            """Display spectra written by the pipeline in *out_dir*.

            Reads ``shift_vs_intensity_*.csv`` files produced by
            ``plot_pred_spectrum``.  Each file becomes one sub-panel.
            Peak labels use the same ``_annotate_peaks_with_barrier``
            logic as the saved PDF, and re-run on every zoom so labels
            never overlap and always stay visible.
            """
            paths = sorted(out_dir.glob("shift_vs_intensity_*.csv"))
            if not paths:
                self._show_placeholder()
                return

            # Build {isotope: {chem_label: count}} from peak_data_*.csv
            counts_map: dict[str, dict[str, int]] = {}
            for pd_path in out_dir.glob("peak_data_*.csv"):
                try:
                    import pandas as _pd
                    df = _pd.read_csv(pd_path, comment="#")
                    if "chem_label" in df.columns and "count" in df.columns:
                        iso_col = df.get("isotope")
                        for _, row in df.iterrows():
                            iso_key = (
                                str(row["isotope"])
                                if iso_col is not None
                                else ""
                            )
                            counts_map.setdefault(iso_key, {})[
                                str(row["chem_label"])
                            ] = int(row["count"])
                except Exception:
                    pass

            self._fig.clear()
            n = len(paths)
            axes = self._fig.subplots(1, n, squeeze=False)[0]

            for ax, path in zip(axes, paths):
                try:
                    sp = _read_spectrum(str(path))
                except Exception:
                    ax.set_axis_off()
                    continue

                iso = sp["isotope"]
                temp = sp["temperature"]
                title_parts = []
                if iso:
                    title_parts.append(iso)
                if temp is not None:
                    title_parts.append(f"{temp:.1f} K")
                subtitle = (
                    "  ".join(title_parts) if title_parts else path.stem
                )
                xlabel = f"{iso} δ (ppm)" if iso else "δ (ppm)"

                lbl_counts = counts_map.get(iso or "", {})
                peak_n = [
                    lbl_counts.get(lbl)
                    for lbl in sp["peak_labels"]
                ]

                data = {
                    "x": sp["shift"],
                    "y": sp["intensity"],
                    "peak_x": sp["peak_shifts"],
                    "peak_lbl": sp["peak_labels"],
                    "peak_n": peak_n,
                    "subtitle": subtitle,
                    "xlabel": xlabel,
                    "_busy": False,
                }
                self._draw_ax(ax, data, invert=True)

            self.draw()

        # ── screen-optimised constants ────────────────────────────────
        _TRACE_COLOR  = "#2c7bb6"
        _LABEL_COLOR  = "#1a1a2e"
        _TICK_COLOR   = "#2c7bb6"
        _LABEL_FS     = 8          # pt — comfortable for screen
        _TICK_FRAC    = 0.08       # tick height as fraction of y-range

        def _draw_ax(self, ax, data: dict, invert: bool) -> None:
            """Clear *ax* and redraw trace + screen-optimised annotations."""
            import numpy as _np
            xlim = ax.get_xlim()
            ax.cla()
            # ax.cla() resets the callback registry — reconnect immediately so
            # subsequent zoom/pan events (including Home) keep labels live.
            def _on_xlim(ax_=ax, data_=data):
                if data_["_busy"]:
                    return
                data_["_busy"] = True
                self._draw_ax(ax_, data_, invert=False)
                data_["_busy"] = False
            ax.callbacks.connect("xlim_changed", _on_xlim)
            x, y = data["x"], data["y"]

            # ── spectrum trace ────────────────────────────────────────
            ax.plot(x, y, lw=1.0, color=self._TRACE_COLOR)
            if invert:
                ax.invert_xaxis()
            else:
                ax.set_xlim(xlim)

            # ── chrome ───────────────────────────────────────────────
            ax.set_yticks([])
            ax.set_title(data["subtitle"], fontsize=9, pad=4)
            ax.set_xlabel(data["xlabel"], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["right", "top", "left"]].set_visible(False)

            # ── peak annotations ─────────────────────────────────────
            peak_x   = data["peak_x"]
            peak_lbl = data["peak_lbl"]
            peak_n   = data.get("peak_n") or []
            if not (peak_x and peak_lbl):
                self.draw_idle()
                return

            xmin, xmax = ax.get_xlim()
            # keep only peaks visible in the current view
            counts_full = peak_n if len(peak_n) == len(peak_x) else [None] * len(peak_x)
            visible = [
                (px, lbl, n)
                for px, lbl, n in zip(peak_x, peak_lbl, counts_full)
                if min(xmin, xmax) <= px <= max(xmin, xmax)
            ]
            if not visible:
                self.draw_idle()
                return

            vis_px, vis_lbl, vis_n = zip(*visible)
            vis_px  = list(vis_px)
            vis_lbl = list(vis_lbl)
            vis_n   = list(vis_n)

            # compute minimum separation that avoids visual character overlap
            # (labels are 90° rotated, so their horizontal footprint ≈ 1 em)
            inv = ax.transData.inverted()
            em_ppm = abs(
                inv.transform((self._LABEL_FS / 72 * self._fig.dpi, 0))[0]
                - inv.transform((0, 0))[0]
            ) * 1.15

            # resolve overlaps — nudge only when peaks are actually too close
            adj = list(vis_px)
            for _ in range(300):
                moved = False
                for i in range(len(adj) - 1):
                    gap = adj[i] - adj[i + 1]   # reversed axis: left > right
                    if abs(gap) < em_ppm:
                        delta = (em_ppm - abs(gap)) / 2
                        adj[i]     += delta
                        adj[i + 1] -= delta
                        moved = True
                if not moved:
                    break

            # draw dashed line from peak to label + count below baseline
            y_range = y.max() - y.min() if y.max() != y.min() else 1.0
            tick_h  = self._TICK_FRAC * y_range
            lbl_y   = y.max() + tick_h * 1.3
            cnt_y   = y.min() - tick_h * 0.6   # sits between baseline and trace

            for px, lx, lbl, n in zip(vis_px, adj, vis_lbl, vis_n):
                peak_y = y[_np.argmin(_np.abs(x - px))]
                ax.plot(
                    [px, lx], [peak_y, lbl_y],
                    color=self._TICK_COLOR, lw=0.6,
                    linestyle="--", alpha=0.5, clip_on=False,
                )
                ax.text(
                    lx, lbl_y, lbl,
                    fontsize=self._LABEL_FS, rotation=90,
                    va="bottom", ha="center",
                    color=self._LABEL_COLOR, clip_on=False,
                )
                if n is not None:
                    ax.text(
                        px, cnt_y, str(n),
                        fontsize=self._LABEL_FS - 1, rotation=0,
                        va="top", ha="center",
                        color=self._TICK_COLOR, clip_on=False,
                        alpha=0.75,
                    )

            # give labels and counts headroom
            ax.set_ylim(
                bottom=y.min() - tick_h * 1.5,
                top=lbl_y,
            )
            self.draw_idle()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SimpNMRWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
