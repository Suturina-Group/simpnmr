# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Unified GUI for the predict and fit_susc workflows.

Launch with:
    simpnmr gui
or:
    python -m simpnmr.gui.app
"""

from __future__ import annotations

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
        self._susc_method.addItems(["file", "spin_only"])
        self._susc_file = _file_picker(self, "Susceptibility file")
        self._susc_format = QComboBox()
        self._susc_format.addItems(["(auto)", "orca_nev", "orca_cas", "csv", "txt", "molcas"])
        self._susc_temps = QLineEdit()
        self._susc_temps.setPlaceholderText("e.g. 298.15  or  298.15, 310.0")
        self._susc_method.currentTextChanged.connect(self._on_susc_method_changed)
        self._sec_susc_src.layout().addRow("Method:", self._susc_method)
        self._sec_susc_src.layout().addRow("File:", self._susc_file)
        self._sec_susc_src.layout().addRow("Format:", self._susc_format)
        self._sec_susc_src.layout().addRow("Temperature(s) (K):", self._susc_temps)
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
                [float(t.strip()) for t in temps_raw.replace(",", " ").split() if t.strip()]
                if temps_raw else []
            )
            if susc_method == "spin_only":
                susc_block: dict = {"method": "spin_only"}
                if temps:
                    susc_block["temperatures"] = temps[0] if len(temps) == 1 else temps
            else:
                susc_block = {}
                sf = self._susc_file._edit.text().strip()
                if sf:
                    susc_block["file"] = sf
                fmt = self._susc_format.currentText()
                if fmt != "(auto)":
                    susc_block["format"] = fmt
                if temps:
                    susc_block["temperatures"] = temps[0] if len(temps) == 1 else temps
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

        # ── Right panel: placeholders ─────────────────────────────────
        right_widget = QSplitter(Qt.Orientation.Vertical)
        right_widget.addWidget(_make_placeholder())
        right_widget.addWidget(_make_placeholder())

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
        else:
            self._log.append_line(f"✗ Exited with code {exit_code}", "#ff8888")

    def closeEvent(self, event):
        if (self._process
                and self._process.state() != QProcess.ProcessState.NotRunning):
            self._process.kill()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Placeholder panel
# ---------------------------------------------------------------------------

def _make_placeholder() -> QWidget:
    w = QWidget()
    w.setStyleSheet("background: #2a2a2a;")
    return w


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
