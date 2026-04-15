# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""GUI prototype for the fit_susc workflow.

Launch with:
    simpnmr gui
or:
    python -m simpnmr.gui.fit_susc_app
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QColor, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
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
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_picker(parent, label: str, mode: str = "file") -> QWidget:
    """Row widget: label + line edit + browse button."""
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_placeholder() -> QWidget:
    """Empty panel for right-side sections not yet assigned."""
    w = QWidget()
    w.setStyleSheet("background: #2a2a2a;")
    return w


# ---------------------------------------------------------------------------
# Config form
# ---------------------------------------------------------------------------

class ConfigForm(QScrollArea):
    """Scrollable form that builds / parses the YAML config."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setSpacing(8)
        self.setWidget(container)

        self._build_form()

    def _build_form(self):
        lay = self._layout

        # ── Project ──────────────────────────────────────────────────────
        sec = _section("Project")
        self._project_name = QLineEdit()
        self._project_name.setPlaceholderText("output folder name")
        sec.layout().addRow("Name:", self._project_name)
        lay.addWidget(sec)

        # ── Hyperfine ─────────────────────────────────────────────────────
        sec = _section("Hyperfine")
        self._hfc_file = _file_picker(self, "Hyperfine file")
        self._hfc_method = QComboBox()
        self._hfc_method.addItems(["dft", "pdip"])
        self._hfc_spin = QLineEdit()
        self._hfc_spin.setPlaceholderText("e.g. 2.0  (leave blank = auto)")
        self._hfc_centre = QLineEdit()
        self._hfc_centre.setPlaceholderText("x, y, z  (Å)")
        sec.layout().addRow("File:", self._hfc_file)
        sec.layout().addRow("Method:", self._hfc_method)
        sec.layout().addRow("Spin S:", self._hfc_spin)
        sec.layout().addRow("Paramagnetic centre:", self._hfc_centre)
        lay.addWidget(sec)

        # ── Experiment ────────────────────────────────────────────────────
        sec = _section("Experiment")
        self._exp_file = _file_picker(self, "Experiment CSV")
        self._chem_labels = _file_picker(self, "Chemical labels CSV")
        self._dia_file = _file_picker(self, "Diamagnetic shifts CSV")
        sec.layout().addRow("Experiment file:", self._exp_file)
        sec.layout().addRow("Chem labels:", self._chem_labels)
        sec.layout().addRow("Diamagnetic file:", self._dia_file)
        lay.addWidget(sec)

        # ── Nuclei & Assignment ───────────────────────────────────────────
        sec = _section("Nuclei & Assignment")
        self._nuclei_include = QLineEdit()
        self._nuclei_include.setPlaceholderText("e.g. H, C")
        self._assign_method = QComboBox()
        self._assign_method.addItems(["fixed", "hungarian", "permutation"])
        sec.layout().addRow("Include nuclei:", self._nuclei_include)
        sec.layout().addRow("Assignment method:", self._assign_method)
        lay.addWidget(sec)

        # ── Susceptibility fit ────────────────────────────────────────────
        sec = _section("Susceptibility fit")
        self._susc_type = QComboBox()
        self._susc_type.addItems(
            ["isoaxrh", "full", "split", "eigen", "isoeigen"]
        )
        self._susc_iso = self._var_row()
        self._susc_ax = self._var_row()
        self._susc_rh = self._var_row(fix_val="0.0")
        self._susc_rh_over_ax = self._var_row(fix_val="0.0")
        self._average_shifts = QCheckBox("Average equivalent shift groups")
        sec.layout().addRow("Fit type:", self._susc_type)
        sec.layout().addRow("iso:", self._susc_iso)
        sec.layout().addRow("ax:", self._susc_ax)
        sec.layout().addRow("rh:", self._susc_rh)
        sec.layout().addRow("rh/ax (fixed ratio):", self._susc_rh_over_ax)
        sec.layout().addRow("", self._average_shifts)
        lay.addWidget(sec)

        # ── Relaxation ────────────────────────────────────────────────────
        sec = _section("Relaxation (optional)")
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
        self._tau_r_fixed.setPlaceholderText("s  (manual override)")
        sec.layout().addRow("τR method:", self._tau_r_method)
        sec.layout().addRow("Solvent:", self._tau_r_solvent)
        sec.layout().addRow("η (Pa·s):", self._tau_r_eta)
        sec.layout().addRow("τR fixed (s):", self._tau_r_fixed)
        lay.addWidget(sec)

    def _var_row(self, fix_val: str = "0.01") -> QWidget:
        """fit/fix toggle + initial value."""
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

    # ── Serialize to YAML dict ─────────────────────────────────────────────

    def to_yaml_dict(self) -> dict:
        d: dict = {}

        name = self._project_name.text().strip()
        if name:
            d["project"] = {"name": name}

        hfc_file = self._hfc_file._edit.text().strip()
        if hfc_file:
            hfc = {"method": self._hfc_method.currentText(), "file": hfc_file}
            spin = self._hfc_spin.text().strip()
            if spin:
                hfc["spin"] = float(spin)
            centre = self._hfc_centre.text().strip()
            if centre:
                hfc["paramagnetic_centre"] = [
                    float(x) for x in centre.replace(",", " ").split()
                ]
            d["hyperfine"] = hfc

        exp_file = self._exp_file._edit.text().strip()
        if exp_file:
            d["experiment"] = {"files": [exp_file]}

        cl_file = self._chem_labels._edit.text().strip()
        if cl_file:
            d["chem_labels"] = {"file": cl_file}

        dia_file = self._dia_file._edit.text().strip()
        if dia_file:
            d["diamagnetic"] = {"method": "csv", "file": dia_file}

        nuclei = self._nuclei_include.text().strip()
        if nuclei:
            d["nuclei"] = {"include": [x.strip() for x in nuclei.split(",")]}

        d["assignment"] = {"method": self._assign_method.currentText()}

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
        susc: dict = {"type": susc_type, "variables": variables}
        if self._average_shifts.isChecked():
            susc["average_shifts"] = "all"
        d["susc_fit"] = susc

        tau_r_method = self._tau_r_method.currentText()
        if tau_r_method != "(none)":
            relax: dict = {"tau_r_method": tau_r_method}
            solvent = self._tau_r_solvent.currentText()
            if solvent != "(none)":
                relax["tau_r_solvent"] = solvent
            eta = self._tau_r_eta.text().strip()
            if eta:
                relax["tau_r_eta"] = float(eta)
            fixed = self._tau_r_fixed.text().strip()
            if fixed:
                relax["tau_r_fixed"] = float(fixed)
            d["fit_relaxation"] = relax

        return d

    # ── Load from YAML dict ────────────────────────────────────────────────

    def from_yaml_dict(self, d: dict):
        proj = d.get("project", {})
        self._project_name.setText(str(proj.get("name", "")))

        hfc = d.get("hyperfine", {})
        self._hfc_file._edit.setText(str(hfc.get("file", "")))
        idx = self._hfc_method.findText(str(hfc.get("method", "dft")))
        if idx >= 0:
            self._hfc_method.setCurrentIndex(idx)
        spin = hfc.get("spin", "")
        self._hfc_spin.setText(str(spin) if spin else "")
        centre = hfc.get("paramagnetic_centre")
        if centre:
            self._hfc_centre.setText(", ".join(str(x) for x in centre))

        exp = d.get("experiment", {})
        files = exp.get("files", [])
        self._exp_file._edit.setText(str(files[0]) if files else "")

        cl = d.get("chem_labels", {})
        self._chem_labels._edit.setText(str(cl.get("file", "")))

        dia = d.get("diamagnetic", {})
        self._dia_file._edit.setText(str(dia.get("file", "")))

        nuclei = d.get("nuclei", {})
        inc = nuclei.get("include", [])
        self._nuclei_include.setText(
            ", ".join(inc) if isinstance(inc, list) else str(inc)
        )

        assign = d.get("assignment", {})
        idx = self._assign_method.findText(str(assign.get("method", "fixed")))
        if idx >= 0:
            self._assign_method.setCurrentIndex(idx)

        susc = d.get("susc_fit", {})
        idx = self._susc_type.findText(str(susc.get("type", "isoaxrh")))
        if idx >= 0:
            self._susc_type.setCurrentIndex(idx)
        variables = susc.get("variables", {})
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
        avg = susc.get("average_shifts")
        self._average_shifts.setChecked(bool(avg))

        relax = d.get("fit_relaxation", {})
        method = str(relax.get("tau_r_method", "(none)"))
        idx = self._tau_r_method.findText(method)
        self._tau_r_method.setCurrentIndex(idx if idx >= 0 else 0)
        solvent = str(relax.get("tau_r_solvent", "(none)"))
        idx = self._tau_r_solvent.findText(solvent)
        self._tau_r_solvent.setCurrentIndex(idx if idx >= 0 else 0)
        self._tau_r_eta.setText(str(relax.get("tau_r_eta", "")))
        self._tau_r_fixed.setText(str(relax.get("tau_r_fixed", "")))


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

class FitSuscWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SimpNMR — fit_susc")
        self.resize(1300, 800)

        self._process: QProcess | None = None
        self._yaml_path: Path | None = None

        self._build_ui()

    def _build_ui(self):
        # ── Toolbar row ───────────────────────────────────────────────────
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

        # ── Left column: form + log ───────────────────────────────────────
        self._form = ConfigForm()
        self._form.setMinimumWidth(380)
        self._form.setMaximumWidth(520)

        self._log = LogPanel()

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        left_layout.addWidget(self._form, 1)
        left_layout.addWidget(QLabel("Log output:"))
        left_layout.addWidget(self._log)

        # ── Right panel: placeholders ─────────────────────────────────────
        right_widget = QSplitter(Qt.Orientation.Vertical)
        right_widget.addWidget(_make_placeholder())
        right_widget.addWidget(_make_placeholder())

        # ── Main splitter: left column | right column ─────────────────────
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(1, 3)

        # ── Assemble ─────────────────────────────────────────────────────
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(toolbar_widget)
        root.addWidget(main_splitter, 1)

        self.setCentralWidget(central)

    # ── File actions ──────────────────────────────────────────────────────

    def _new(self):
        self._form.from_yaml_dict({})
        self._yaml_path = None
        self.setWindowTitle("SimpNMR — fit_susc  [new]")

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
            self.setWindowTitle(
                f"SimpNMR — fit_susc  [{self._yaml_path.name}]"
            )
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
            d = self._form.to_yaml_dict()
            with open(path, "w") as f:
                yaml.dump(d, f, default_flow_style=False, allow_unicode=True)
            self.setWindowTitle(f"SimpNMR — fit_susc  [{path.name}]")
            self._log.append_line(f"Saved {path}", "#88cc88")
        except Exception as e:
            QMessageBox.critical(self, "Error saving YAML", str(e))

    # ── Run / stop ────────────────────────────────────────────────────────

    def _run(self):
        # Save to a temp file if not saved yet
        if self._yaml_path is None:
            self._save_yaml_as()
            if self._yaml_path is None:
                return
        else:
            self._write_yaml(self._yaml_path)

        # Determine output directory from project name
        d = self._form.to_yaml_dict()
        project_name = d.get("project", {}).get("name", "")
        work_dir = str(self._yaml_path.parent)
        out_dir = (
            str(self._yaml_path.parent / project_name)
            if project_name else work_dir
        )

        self._log.clear()
        self._log.append_line(
            f"Running: simpnmr fit_susc {self._yaml_path.name}", "#aaaaff"
        )

        cmd = [sys.executable, "-m", "simpnmr.cli.main_entry", "fit_susc"]
        if self._hide_cb.isChecked():
            cmd = [sys.executable, "-c",
                   "from simpnmr.cli.main import interface; interface()",
                   "--hide", "fit_susc", str(self._yaml_path)]
        else:
            cmd = [sys.executable, "-c",
                   "from simpnmr.cli.main import interface; interface()",
                   "fit_susc", str(self._yaml_path)]

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
        data = self._process.readAllStandardOutput().data().decode(
            errors="replace"
        )
        for line in data.splitlines():
            self._log.append_line(line, "#cccccc")

    def _on_stderr(self):
        data = self._process.readAllStandardError().data().decode(
            errors="replace"
        )
        for line in data.splitlines():
            color = (
                "#ff8888" if "ERROR" in line or "Traceback" in line
                else "#ffcc66"
            )
            self._log.append_line(line, color)

    def _on_finished(self, exit_code: int, _):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        if exit_code == 0:
            self._log.append_line("✓ Finished successfully", "#88cc88")
        else:
            self._log.append_line(f"✗ Exited with code {exit_code}", "#ff8888")

    def closeEvent(self, event):
        if (self._process
                and self._process.state()
                != QProcess.ProcessState.NotRunning):
            self._process.kill()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = FitSuscWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
