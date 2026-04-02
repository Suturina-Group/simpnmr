Standalone CLI utilities
========================

This page documents selected SimpNMR command-line utilities that operate
*directly on input files* and do not require YAML pipeline configuration files.

These commands are intended for quick calculations, post-processing, and
exploratory analysis, and can be used independently of the YAML-driven
workflows documented elsewhere.


``calc_pcs_iso``
----------------

The ``calc_pcs_iso`` command computes isotropic pseudocontact shift (PCS)
fields directly from a magnetic susceptibility tensor and a molecular
structure.

This command is intentionally lightweight and does not require a YAML input
file. The susceptibility backend and method are determined automatically from
the input file.

**Automatic method selection**

If the susceptibility source is an ORCA output file, the most advanced
available susceptibility method is selected automatically (e.g. NEVPT2 is
preferred over CASSCF). Explicit method selection is not required.

**Required input**

- A susceptibility source file
  (e.g. CSV file or ORCA output containing susceptibility data)
- A temperature value (in Kelvin)
- A molecular structure file (XYZ format)
- The label of the paramagnetic centre (as defined in the structure file)

**Typical usage**

::

   simpnmr calc_pcs_iso susceptibility.csv 298.0 structure.xyz Fe1

When an ORCA output file is provided instead of a CSV file, the susceptibility
backend and method are inferred automatically from the file contents.

The command produces PCS values on a three-dimensional grid, suitable for
visualisation or further analysis.



``label_groups``
-----------------

The ``label_groups`` command automatically identifies methyl (CH\ :sub:`3`\)
and *tert*-butyl (C(CH\ :sub:`3`\)\ :sub:`3`\) groups in an XYZ structure
file using distance-based bond detection, and assigns group labels to each
atom.

This is useful for generating ``chem_labels`` CSV files when equivalent proton
groups (e.g. *tert*-butyl protons) should be treated as a single resonance in
the spectrum.

**Required input**

- An XYZ file (standard or Chemcraft format)

**Typical usage**

::

   label_groups molecule.xyz

**Outputs**

- ``<input>_labeled.xyz`` — original XYZ with group tags appended in quotes,
  e.g. ``H   1.23  4.56  7.89  "tBu2"``
- ``<input>_labels.csv`` — ``atom_label,chem_label,isotope`` CSV for all C
  and H atoms, ready to use as ``chem_labels.file`` in a SimpNMR YAML.
  The ``isotope`` column is auto-populated (H → ``1H``, C → ``13C``)

**Group naming**

- *tert*-butyl groups are labelled ``tBu1``, ``tBu2``, … (sorted by central
  carbon index)
- Standalone methyl groups are labelled ``Me1``, ``Me2``, …

All atoms belonging to the same group receive the same ``chem_label``, so
their shifts are averaged during prediction.


``get_sh``
----------

The ``get_sh`` command derives effective spin-Hamiltonian parameters from a
previously performed magnetic susceptibility (χT) regression.

It is intended as a post-processing utility and does not perform fitting
itself. No YAML configuration file is required.

**Required input**

- A CSV file containing χT regression results
- The spin quantum number of the paramagnetic centre

**Typical usage**

::

   simpnmr get_sh --spin 2.0 chiT_regression.csv

The command extracts the effective g-tensor (and, if applicable, zero-field
splitting parameters) from the regression data and prints the results to
standard output.

