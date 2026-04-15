# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""
molecule_view.py
================
Drop-in PyQt6 widget for simpNMR that reads an XYZ file (atomic numbers
*or* element symbols, with optional per-atom quoted labels) and renders
it interactively using 3Dmol.js inside a QWebEngineView.

Features
--------
* Parses standard XYZ (count + comment header) AND headerless variants.
* Accepts either atomic numbers (e.g. 66 for Dy) or element symbols.
* Reads optional custom labels from a quoted token in column 5+:
      6   1.234  -0.567  0.890   "ceq"
* Groups atoms by label and color-codes them in the viewer.
* Shows ONE framed label per group, anchored to a chosen representative
  atom of the group (configurable to avoid overlaps).
* Exposes a `MoleculeView(QWidget)` you can embed in any PyQt6 layout.
* Two-way bridge via QWebChannel: emits an `atomClicked` signal carrying
  the atom's index, element, group label and coordinates so simpNMR can
  link 3D picks to a chemical-shift table.

Dependencies
------------
    pip install PyQt6 PyQt6-WebEngine
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QVBoxLayout, QWidget


# ---------------------------------------------------------------------------
# 1. XYZ parsing
# ---------------------------------------------------------------------------

PERIODIC_TABLE = [
    'n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
    'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
    'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
    'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
    'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
    'Ra', 'Ac', 'Th', 'Pa', 'U',
]


@dataclass
class Atom:
    index: int            # 0-based — what 3Dmol uses for selection
    serial: int           # 1-based — convenient for users
    element: str
    x: float
    y: float
    z: float
    label: Optional[str] = None   # custom label from xyz, if any


@dataclass
class Molecule:
    atoms: list[Atom] = field(default_factory=list)
    source_path: Optional[str] = None

    @property
    def elements(self) -> list[str]:
        return sorted({a.element for a in self.atoms})

    def grouped_by_label(self) -> dict[str, list[Atom]]:
        g: dict[str, list[Atom]] = defaultdict(list)
        for a in self.atoms:
            if a.label:
                g[a.label].append(a)
        return dict(g)

    def to_xyz_string(self, comment: str = "") -> str:
        lines = [str(len(self.atoms)), comment]
        for a in self.atoms:
            lines.append(f"{a.element:<3s} {a.x:14.6f} {a.y:14.6f} {a.z:14.6f}")
        return "\n".join(lines) + "\n"


_LABEL_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')


def parse_xyz(path: str | Path) -> Molecule:
    """Parse an XYZ file. Tolerates:
       - missing 2-line header (atom count + comment)
       - atomic numbers OR element symbols in column 1
       - optional quoted custom label in any later column
    """
    raw = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not raw:
        raise ValueError(f"{path}: file is empty")

    # Detect the standard XYZ header (line 1 = integer count alone)
    first = raw[0].split()
    has_header = (len(first) == 1 and first[0].isdigit()
                  and int(first[0]) <= len(raw) - 2)
    body = raw[2:] if has_header else raw

    atoms: list[Atom] = []
    for i, ln in enumerate(body):
        parts = ln.split(None, 4)             # keep remainder intact for the label
        if len(parts) < 4:
            continue
        col0 = parts[0]
        if col0.isdigit():
            element = PERIODIC_TABLE[int(col0)]
        else:
            element = col0[0].upper() + col0[1:].lower()
        x, y, z = (float(parts[1]), float(parts[2]), float(parts[3]))
        label = None
        if len(parts) == 5:
            m = _LABEL_RE.search(parts[4])
            if m:
                label = m.group(1) or m.group(2)
            else:
                # fall back: a bare token after the coords is treated as a label
                token = parts[4].strip().split()
                if token:
                    label = token[0]
        atoms.append(Atom(index=i, serial=i + 1, element=element,
                          x=x, y=y, z=z, label=label))

    return Molecule(atoms=atoms, source_path=str(path))


# ---------------------------------------------------------------------------
# 2. Default label color palette
# ---------------------------------------------------------------------------

DEFAULT_PALETTE = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#1b9e77', '#f781bf', '#a65628', '#ffd92f', '#17becf',
    '#bcbd22', '#8c564b', '#e377c2', '#7f7f7f', '#2ca02c',
]


def assign_label_colors(labels: list[str],
                        custom: Optional[dict[str, str]] = None
                        ) -> dict[str, str]:
    """Map labels to hex colors. Honors any user-supplied overrides."""
    custom = custom or {}
    out: dict[str, str] = {}
    pal_iter = iter(DEFAULT_PALETTE * 4)
    for lbl in sorted(labels):
        out[lbl] = custom[lbl] if lbl in custom else next(pal_iter)
    return out


# ---------------------------------------------------------------------------
# 3. HTML generation for the 3Dmol.js viewer
# ---------------------------------------------------------------------------

VIEWER_HTML_TEMPLATE = """\
<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<style>
html,body{{margin:0;height:100%;background:#fff;font-family:sans-serif}}
#v{{width:100%;height:100vh;position:relative}}
</style></head>
<body><div id="v"></div>
<script>
const xyz       = {xyz_json};
const colored   = {colored_json};
const labels    = {labels_json};
const baseStyle = {base_style_json};
const labelStyle= {label_style_json};

let viewer = $3Dmol.createViewer("v", {{backgroundColor:"white"}});
viewer.addModel(xyz, "xyz");
viewer.setStyle({{}}, baseStyle);

// Override style for labelled hydrogens (color them by group)
colored.forEach(c => {{
  viewer.setStyle({{index: c.index}},
    {{sphere:{{scale: c.scale, color: c.color}}, stick:{{radius:0.12}}}});
}});

// Single framed label per group, on a representative atom
labels.forEach(L => {{
  viewer.addLabel(L.text, {{
    position: {{x:L.x, y:L.y, z:L.z + 0.6}},
    fontSize: labelStyle.fontSize,
    fontColor: labelStyle.fontColor,
    backgroundColor: 'white',
    backgroundOpacity: 0.95,
    borderThickness: labelStyle.borderThickness,
    borderColor: L.color,
    inFront: true
  }});
}});

// QWebChannel bridge for click-back into Python
new QWebChannel(qt.webChannelTransport, function(channel) {{
  window.pyObj = channel.objects.pyObj;
  viewer.setClickable({{}}, true, function(atom) {{
    if (window.pyObj) pyObj.onAtomClick(JSON.stringify({{
      index: atom.index, serial: atom.serial,
      elem: atom.elem, x: atom.x, y: atom.y, z: atom.z
    }}));
  }});
  viewer.render();
}});

viewer.zoomTo();
viewer.render();
</script></body></html>"""


def build_viewer_html(mol: Molecule,
                      label_colors: Optional[dict[str, str]] = None,
                      rep_index: Optional[dict[str, int]] = None,
                      base_sphere_scale: float = 0.25,
                      group_sphere_scale: float = 0.275,
                      label_font_size: int = 28,
                      title: str = "Molecule") -> str:
    """Build a self-contained HTML page that renders `mol` with grouped
    label colors and one framed label per group.

    Parameters
    ----------
    label_colors : dict[label -> "#rrggbb"], optional
        Override the auto-assigned palette for any label.
    rep_index : dict[label -> int], optional
        Which member of the group to anchor the label on (0-based).
        Use this to break visual overlaps between groups.
    """
    groups = mol.grouped_by_label()
    colors = assign_label_colors(list(groups.keys()), label_colors)
    rep_index = rep_index or {}

    colored = [{'index': a.index,
                'color': colors[a.label],
                'scale': group_sphere_scale}
               for a in mol.atoms if a.label]

    labels = []
    for tag, members in groups.items():
        idx = rep_index.get(tag, 0) % len(members)
        rep = members[idx]
        labels.append({'text': tag, 'color': colors[tag],
                       'x': rep.x, 'y': rep.y, 'z': rep.z})

    base_style = {'sphere': {'scale': base_sphere_scale},
                  'stick':  {'radius': 0.12}}
    label_style = {'fontSize': label_font_size,
                   'fontColor': 'black',
                   'borderThickness': 2.0}

    return VIEWER_HTML_TEMPLATE.format(
        title=title,
        xyz_json=json.dumps(mol.to_xyz_string()),
        colored_json=json.dumps(colored),
        labels_json=json.dumps(labels),
        base_style_json=json.dumps(base_style),
        label_style_json=json.dumps(label_style),
    )


# ---------------------------------------------------------------------------
# 4. PyQt6 widget
# ---------------------------------------------------------------------------

class _Bridge(QObject):
    """JS -> Python bridge object exposed as `pyObj` in the page."""
    atomClicked = pyqtSignal(dict)

    @pyqtSlot(str)
    def onAtomClick(self, payload_json: str):
        self.atomClicked.emit(json.loads(payload_json))


class MoleculeView(QWidget):
    """Embeddable PyQt6 widget that renders a molecule via 3Dmol.js.

    Signals
    -------
    atomClicked(dict)
        Emitted when the user clicks an atom in the 3D viewer. The dict
        contains: index, serial, elem, x, y, z, and (added by Python)
        the atom's custom `label` if any.
    """

    atomClicked = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._mol: Optional[Molecule] = None

        self._web = QWebEngineView(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._web)

        self._bridge = _Bridge()
        self._bridge.atomClicked.connect(self._on_atom_clicked)
        self._channel = QWebChannel(self)
        self._channel.registerObject("pyObj", self._bridge)
        self._web.page().setWebChannel(self._channel)

        self._web.setHtml("<html><body style='background:#fff'></body></html>")

    # ---- public API ----
    def load_xyz_file(self, path: str | Path,
                      label_colors: Optional[dict[str, str]] = None,
                      rep_index: Optional[dict[str, int]] = None,
                      **viewer_opts) -> Molecule:
        mol = parse_xyz(path)
        self.show_molecule(mol, label_colors=label_colors,
                           rep_index=rep_index, **viewer_opts)
        return mol

    def show_molecule(self, mol: Molecule, **viewer_opts) -> None:
        self._mol = mol
        html = build_viewer_html(mol, **viewer_opts)
        # Write to a temp file so the embedded qwebchannel.js loads correctly
        tmp = tempfile.NamedTemporaryFile('w', suffix='.html',
                                          delete=False, encoding='utf-8')
        tmp.write(html); tmp.close()
        self._web.load(QUrl.fromLocalFile(tmp.name))

    def run_js(self, code: str, callback=None) -> None:
        page = self._web.page()
        if callback:
            page.runJavaScript(code, callback)
        else:
            page.runJavaScript(code)

    def highlight_atom(self, index: int, color: str = 'yellow') -> None:
        self.run_js(f"""
            viewer.setStyle({{index:{index}}},
              {{sphere:{{scale:0.55, color:{json.dumps(color)}}},
                stick:{{radius:0.12}}}});
            viewer.render();
        """)

    # ---- internal ----
    def _on_atom_clicked(self, payload: dict) -> None:
        if self._mol and 0 <= payload.get('index', -1) < len(self._mol.atoms):
            payload['label'] = self._mol.atoms[payload['index']].label
        self.atomClicked.emit(payload)


# ---------------------------------------------------------------------------
# 5. Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Run as `python molecule_view.py path/to/file.xyz`."""
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow

    if len(sys.argv) < 2:
        print("usage: python molecule_view.py file.xyz")
        sys.exit(1)

    app = QApplication(sys.argv)
    win = QMainWindow()
    view = MoleculeView(win)
    win.setCentralWidget(view)

    # Optional: pin specific labels to specific group members to avoid overlap
    rep = {'ceqp': 2}
    view.load_xyz_file(sys.argv[1], rep_index=rep)

    def on_pick(atom):
        lbl = atom.get('label') or '(no label)'
        print(f"clicked {atom['elem']}#{atom['serial']}  label={lbl}  "
              f"xyz=({atom['x']:.3f}, {atom['y']:.3f}, {atom['z']:.3f})")
    view.atomClicked.connect(on_pick)

    win.resize(900, 700)
    win.setWindowTitle(f"MoleculeView — {sys.argv[1]}")
    win.show()
    sys.exit(app.exec())
