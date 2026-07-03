.. _standalone_cli:

Standalone CLI utilities
========================

This page documents selected SimpNMR command-line utilities that operate
*directly on input files* and do not require YAML pipeline configuration files.

These commands are intended for quick calculations, post-processing, and
exploratory analysis, and can be used independently of the YAML-driven
workflows documented elsewhere.


``average_conformers``
----------------------

The ``average_conformers`` command reads N quantum-chemistry output files
representing different conformers of the same molecule, computes
population-weighted averages of the hyperfine A tensors and of the
ensemble-averaged ⟨r⁻⁶⟩ distances to the paramagnetic centre, and writes a
canonical SimpNMR molecule CSV that can be used directly as a
``method: csv`` hyperfine input.

**When to use this**

Use this command when you have multiple low-energy conformers that interconvert
rapidly on the NMR timescale and want a single effective input that correctly
captures:

- Conformer-averaged paramagnetic shifts (A tensor average — exact because
  shifts are linear in A).
- Conformer-averaged relaxation rates (⟨r⁻⁶⟩ average — correct because r⁻⁶
  is non-linear in r, so this is more accurate than averaging coordinates).

**Required input**

- Two or more QC output files with identical atom order
- The atom label of the paramagnetic centre (e.g. ``Fe1``)

**Typical usage**

Equal weights (default)::

   simpnmr average_conformers conformer1.out conformer2.out conformer3.out \
       --centre Fe1

Boltzmann-weighted (unnormalised weights are accepted)::

   simpnmr average_conformers conformer1.out conformer2.out conformer3.out \
       --centre Fe1 \
       --weights 0.60 0.30 0.10

Custom output file name::

   simpnmr average_conformers conf*.out --centre Fe1 --output hfc_avg.csv

**Options**

``--centre <label>``
   Atom label of the paramagnetic centre in the QC files (required).

``--weights <w1 w2 ...>``
   Population weight for each conformer. Values need not sum to 1 — they are
   normalised automatically. Omit for equal weights.

``--output <file>``
   Output CSV file name. Defaults to
   ``conformer_avg_<stem_of_first_file>.csv``.

**Output**

A canonical molecule CSV with:

- Coordinates from the **first (reference) conformer** — labelled in the
  file header so you know which geometry was used.
- Averaged A tensors (``A_fc_iso``, ``A_sd_*``, and ``A_orb_*`` if present).
- An extra ``r_inv6 (Å^-6)`` column containing the weighted ⟨r⁻⁶⟩ per
  nucleus. When this column is present the relaxation fitting pipeline uses
  these values directly instead of computing 1/r⁶ from the reference
  coordinates.

**Using the output as CSV hyperfine input**

::

   hyperfine:
     method: csv
     file: conformer_avg_conformer1.csv
     spin: 2.5
     paramagnetic_centre: [0.0, 0.0, 0.0]  # not used for r⁻⁶, but required
                                             # for pdip fallback checks

.. note::

   All conformer files must contain the same atoms in the same order. The
   backend is detected automatically (ORCA or Gaussian). If orbital hyperfine
   data (``A(ORB)``) are present in any conformer, all conformers are expected
   to provide them; missing orbital tensors in a conformer are treated as zero.


``extract_hfc``
---------------

The ``extract_hfc`` command reads a quantum-chemistry output file, extracts
hyperfine coupling (A) tensors, and writes them to a canonical SimpNMR
molecule CSV file.  The output can be used directly as a ``method: csv``
hyperfine input in any pipeline YAML (see :ref:`input_files`).

**Supported input formats**

- ORCA output or property files (``*.out``, ``*.txt``)
- Gaussian log files (``*.log``)

The backend is detected automatically from the file contents.

**Required input**

- A quantum-chemistry output file containing A-tensor data

**Typical usage**

::

   simpnmr extract_hfc path/to/orca_hfc.out

This writes ``hyperfine_<filename>.csv`` in the current directory.

**Output columns**

The CSV contains atom labels, Cartesian coordinates, and the split
hyperfine tensor in ``ppm Å⁻³``:

.. code-block:: text

   atom_label (),chem_label (),x (Å),y (Å),z (Å),
   A_fc_iso (ppm Å^-3),
   A_sd_xx (ppm Å^-3), A_sd_xy (ppm Å^-3), A_sd_xz (ppm Å^-3),
   A_sd_yy (ppm Å^-3), A_sd_yz (ppm Å^-3), A_sd_zz (ppm Å^-3)

When orbital hyperfine data (``A(ORB)``) are present in the source file,
six additional ``A_orb_*`` columns are appended automatically.

**Using the output as CSV hyperfine input**

Point the ``hyperfine`` block of your YAML to the generated file::

   hyperfine:
     method: csv
     file: hyperfine_orca_hfc.csv
     spin: 2.5       # required for csv method
     orbit: 0        # optional
     total_momentum_J: null  # optional

.. note::

   Hyperfine tensors in the output file are expressed in the same coordinate
   frame as the QC calculation.  If your susceptibility tensor is in a
   different frame (e.g. the χ eigenframe), you must rotate the tensors
   manually before using this file as input.


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


``calc_tau_c``
--------------

The ``calc_tau_c`` command estimates the isotropic rotational correlation time
τ\ :sub:`R` from molecular coordinates using a hydrodynamic model.  Viscosity
is taken from a built-in solvent database (with Arrhenius temperature
correction) or supplied directly.

**Available models**

*Perrin ellipsoid* (default, ``--method ellipsoid``)
   Fits the molecule to a triaxial ellipsoid and applies the Perrin analytical
   rotational diffusion tensor.  Fast and suitable for compact, roughly
   ellipsoidal molecules.

*Bead-shell* (``--method beadshell``)
   Covers the molecular surface with minibeads and builds a Rotne–Prager–Yamakawa
   hydrodynamic interaction matrix.  More accurate for extended or irregular
   shapes.

**Required input**

- A coordinate file (XYZ or PDB format)
- One or more temperatures in Kelvin

**Typical usage**

::

   simpnmr calc_tau_c molecule.xyz 298.0 --solvent D2O

Multiple temperatures can be given in a single call::

   simpnmr calc_tau_c molecule.xyz 280.0 298.0 310.0 --solvent D2O

To supply a custom viscosity instead of using the solvent database::

   simpnmr calc_tau_c molecule.xyz 298.0 --eta 1.1e-3

**Options**

``--solvent <name>``
   Solvent name (e.g. ``D2O``, ``CDCl3``, ``DMSO-d6``, ``CD3OD``).
   Viscosity is automatically corrected to each temperature using Arrhenius
   scaling.  Run ``simpnmr calc_tau_c --help`` to see all available solvents.

``--eta <float>``
   Explicit solvent viscosity in Pa·s.  Overrides ``--solvent`` and is applied
   unchanged at every temperature.

``--method ellipsoid|beadshell``
   Hydrodynamic model (default: ``ellipsoid``).

``--shell <float>``
   Thickness of the solvation shell added to all van der Waals radii (Å,
   default: 0.0).

``--sigma <float>``
   Minibead radius for the bead-shell model (Å, default: 0.6).

**Output**

A summary table is printed to standard output::

   T (K)   η (mPa·s)    τ_R (ps)   D_iso (rad²/s)  Anisotropy
   ---------------------------------------------------------------
   298.0      1.1000      152.3      1.0948e+07       1.450

The computed τ\ :sub:`R` values can be fed directly into the
``fit_relaxation:tau_r_fixed`` field (see :ref:`fit_relaxation block <fit-relaxation-block>`)
or automated via ``tau_r_method`` (see below).


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

