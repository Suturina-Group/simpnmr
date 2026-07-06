.. _tutorial_fe:

Fe(II) complex: DFT-calculated hyperfine tensors
================================================

This tutorial works through susceptibility fitting and spin-Hamiltonian
extraction on an **iron(II) complex**, using **hyperfine coupling (HFC) tensors
from a DFT calculation**.

For a transition metal such as Fe(II), the unpaired electrons occupy valence
d-orbitals that delocalise significantly onto the ligands. This covalency
produces a sizeable Fermi-contact hyperfine contribution that depends on the
electronic structure, not just the geometry — so the point-dipole approximation
used for lanthanides (see :doc:`dy_point_dipole`) is not sufficient. Instead,
SimpNMR reads the full hyperfine tensors from a quantum-chemistry output
(``hyperfine: method: dft``), and here also takes the ab-initio susceptibility
from an ORCA calculation.

.. contents:: On this page
   :local:
   :depth: 1
   :backlinks: none

Prerequisites
-------------

Install ``simpnmr`` by following the :doc:`../user_guide/installation` guide,
then check it is available:

.. code-block:: bash

   simpnmr --version

Command line or the desktop app
-------------------------------

The workflows below can be run from the terminal (the ``simpnmr`` command) or
from the **desktop GUI**, which presents the same options as a form and can open
and save the ``run.yml`` shown here. To use the GUI, install the optional GUI
dependencies and launch it:

.. code-block:: bash

   pip install "simpnmr[gui]"
   simpnmr-gui

The window has a toolbar (**New**, **Open YAML…**, **Save YAML**, **Save As…**,
**▶ Run**, **■ Stop**); a form on the left whose sections mirror the ``run.yml``
blocks, with a **Workflow** selector (**Predict** / **Fit susceptibility**) at
the top; and a 3D viewer with a live log panel on the right. The general pattern
is: **Open YAML…** → choose your ``run.yml`` → set the **Workflow** mode → click
**▶ Run**.

The complex and its input files
-------------------------------

Create a working directory with the ``data/`` sub-folders:

.. code-block:: bash

   mkdir fe-tutorial
   cd fe-tutorial
   mkdir -p data/chi data/hfc data/labels data/dia data/para

**Ab-initio inputs (downloads).** The hyperfine tensors and the susceptibility
come from quantum-chemistry calculations whose output files are large. Download
each and save it under the path shown:

* :download:`hyperfine.out <../_downloads/examples/05_VT_Susceptibility_Fitting/data/hfc/hyperfine.out>`
  → ``data/hfc/hyperfine.out`` (DFT hyperfine tensors)
* :download:`susceptibility.out <../_downloads/examples/05_VT_Susceptibility_Fitting/data/chi/susceptibility.out>`
  → ``data/chi/susceptibility.out`` (ORCA NEVPT2 susceptibility)

**The diamagnetic reference shifts** (ppm):

.. code-block:: text
   :caption: data/dia/diamagnetic_shifts.csv

   chem_label,shift
   H1,1.87
   H2_u,1.26
   H2_d,1.35
   H3,1.09
   H4,7.46
   H5,7.26
   H6,7.26
   C1,10.81
   C2_up,19.30
   C2_down,21.01
   C3,33.07
   C4,133.6
   C5,128.46
   C6,129.47
   C7,134.48

**The chemical-label map** — note the extra ``chem_math_label`` column, which
gives a LaTeX label used in figures:

.. dropdown:: data/labels/chemical_labels.csv

   .. code-block:: text
      :caption: data/labels/chemical_labels.csv

      atom_label,chem_label,chem_math_label
      H1,H3,$\mathbf{H}_{\mathit{3}}$
      H2,H3,$\mathbf{H}_{\mathit{3}}$
      H3,H3,$\mathbf{H}_{\mathit{3}}$
      H4,H3,$\mathbf{H}_{\mathit{3}}$
      H5,H3,$\mathbf{H}_{\mathit{3}}$
      H6,H3,$\mathbf{H}_{\mathit{3}}$
      H7,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H8,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H9,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H10,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H11,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H12,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H13,H1,$\mathbf{H}_{\mathit{1}}$
      H14,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H15,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H16,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H17,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H18,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H19,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H20,H1,$\mathbf{H}_{\mathit{1}}$
      H21,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H22,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H23,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H24,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H25,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H26,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H27,H1,$\mathbf{H}_{\mathit{1}}$
      H28,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H29,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H30,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H31,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H32,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H33,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H34,H1,$\mathbf{H}_{\mathit{1}}$
      H35,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H36,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H37,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H38,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H39,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H40,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H41,H1,$\mathbf{H}_{\mathit{1}}$
      H42,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H43,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H44,H2_d,$\mathbf{H}_{\mathit{2}}^{\mathit{D}}$
      H45,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H46,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H47,H2_u,$\mathbf{H}_{\mathit{2}}^{\mathit{U}}$
      H48,H4,$\mathbf{H}_{\mathit{4}}$
      H49,H5,$\mathbf{H}_{\mathit{5}}$
      H50,H6,$\mathbf{H}_{\mathit{6}}$
      H51,H5,$\mathbf{H}_{\mathit{5}}$
      H52,H4,$\mathbf{H}_{\mathit{4}}$
      H53,H1,$\mathbf{H}_{\mathit{1}}$
      C1,C3,$\mathbf{C}_{\mathit{3}}$
      C2,C3,$\mathbf{C}_{\mathit{3}}$
      C3,C3,$\mathbf{C}_{\mathit{3}}$
      C4,C1,$\mathbf{C}_{\mathit{1}}$
      C5,C2_up,$\mathbf{C}_{\mathit{2}}^{\mathit{U}}$
      C6,C2_down,$\mathbf{C}_{\mathit{2}}^{\mathit{D}}$
      C7,C1,$\mathbf{C}_{\mathit{1}}$
      C8,C2_up,$\mathbf{C}_{\mathit{2}}^{\mathit{U}}$
      C9,C2_down,$\mathbf{C}_{\mathit{2}}^{\mathit{D}}$
      C10,C1,$\mathbf{C}_{\mathit{1}}$
      C11,C2_up,$\mathbf{C}_{\mathit{2}}^{\mathit{U}}$
      C12,C2_down,$\mathbf{C}_{\mathit{2}}^{\mathit{D}}$
      C13,C1,$\mathbf{C}_{\mathit{1}}$
      C14,C2_up,$\mathbf{C}_{\mathit{2}}^{\mathit{U}}$
      C15,C2_down,$\mathbf{C}_{\mathit{2}}^{\mathit{D}}$
      C16,C1,$\mathbf{C}_{\mathit{1}}$
      C17,C2_down,$\mathbf{C}_{\mathit{2}}^{\mathit{D}}$
      C18,C2_up,$\mathbf{C}_{\mathit{2}}^{\mathit{U}}$
      C19,C1,$\mathbf{C}_{\mathit{1}}$
      C20,C2_down,$\mathbf{C}_{\mathit{2}}^{\mathit{D}}$
      C21,C2_up,$\mathbf{C}_{\mathit{2}}^{\mathit{U}}$
      C22,C7,$\mathbf{C}_{\mathit{7}}$
      C23,C4,$\mathbf{C}_{\mathit{4}}$
      C24,C5,$\mathbf{C}_{\mathit{5}}$
      C25,C6,$\mathbf{C}_{\mathit{6}}$
      C26,C5,$\mathbf{C}_{\mathit{5}}$
      C27,C4,$\mathbf{C}_{\mathit{4}}$

**The variable-temperature peak lists** — twelve measured spectra, one per
temperature. Save each block under ``data/para/`` with the file name in its
caption:

.. dropdown:: The twelve peak lists (data/para/\*K.csv)

   .. code-block:: text
      :caption: data/para/experiment_248K.csv

      #temperature 247.25
      #magnetic_field 11.75
      #isotope 1H
      assignment,shift (ppm),width (Hz),area ()
      H3, 213.40, 4163.02, 5.52
      H4, 46.69, 49.02, 1.98
      H5, 22.19, 15.12, 1.70
      H6, 20.51, 17.03, 1.00
      H2_d, 6.46, 441.82, 16.54
      H1, -30.65, 1611.66, 4.07
      H2_u, -50.50, 524.48, 15.14
      C3,1201.49,619.89,42.03
      C1,913.24,38.09,1.00
      C2_down,483.66,325.13,75.84
      C4,193.71,149.55,13.39
      C5,154.41,73.57,13.83
      C7,149.54,21.06,8.48
      C6,146.38,22.21,5.29
      C2_up,126.16,167.98,22.97

   .. code-block:: text
      :caption: data/para/experiment_253K.csv

      #temperature 252.27
      #magnetic_field 11.75
      #isotope 1H
      assignment,shift (ppm),width (Hz),area ()
      H3, 208.31, 3323.97, 5.61
      H4, 45.41, 45.64, 2.02
      H5, 21.69, 17.27, 1.92
      H6, 20.09, 17.66, 1.00
      H2_d, 6.34, 351.95, 17.21
      H1, -28.56, 1285.87, 4.22
      H2_u, -48.37, 443.41, 15.15
      C3,1177.84,357.63,8.74
      C1,897.17,96.99,1.00
      C2_down,472.97,300.41,23.97
      C4,190.93,66.2,3.53
      C5,153.26,42.86,2.73
      C7,148.32,38.17,2.99
      C6,145.01,22.94,1.83
      C2_up,123.13,165.07,9.55

   .. code-block:: text
      :caption: data/para/experiment_258K.csv

      #temperature 257.27
      #magnetic_field 11.75
      #isotope 1H
      assignment,shift (ppm),width (Hz),area ()
      H3, 203.54, 2645.10, 5.87
      H4, 44.20, 44.55, 2.12
      H5, 21.22, 16.77, 2.07
      H6, 19.69, 18.31, 1.00
      H2_d, 6.22, 287.93, 18.10
      H1, -26.53, 1107.38, 5.17
      H2_u, -46.38, 375.62, 15.65
      C3,1154.78,568.39,6.62
      C1,875.33,113.24,1.00
      C2_down,463.02,266.45,15.85
      C4,189.63,162.95,3.36
      C5,152.39,23.7,2.33
      C7,147.79,35.15,1.60
      C6,144.37,16.2,1.24
      C2_up,120.46,166.45,7.28

   .. code-block:: text
      :caption: data/para/experiment_263K.csv

      #temperature 262.29
      #magnetic_field 11.75
      #isotope 1H
      assignment,shift (ppm),width (Hz),area ()
      H3, 198.84, 1867.32, 3.13
      H4, 43.06, 42.73, 2.28
      H5, 20.77, 17.33, 2.15
      H6, 19.32, 18.13, 1.00
      H2_d, 6.11, 232.57, 19.09
      H1, -24.69, 915.38, 5.41
      H2_u, -44.51, 321.08, 16.88
      C3,1132.51,578.88,5.13
      C1,858.34,230.18,1.32
      C2_down,453.51,203.81,9.97
      C4,188.53,65,1.70
      C5,152.23,22.92,1.64
      C7,147.76,26.34,1.44
      C6,144.22,23.85,1.00
      C2_up,118.43,251.87,4.40

   .. code-block:: text
      :caption: data/para/experiment_268K.csv

      #temperature 267.28
      #magnetic_field 11.75
      #isotope 1H
      assignment,shift (ppm),width (Hz),area ()
      H3, 194.41, 1627.96, 5.20
      H4, 41.97, 41.14, 2.23
      H5, 20.34, 17.15, 2.07
      H6, 18.96, 17.54, 1.00
      H2_d, 6.00, 193.89, 18.84
      H1, -22.95, 806.53, 5.85
      H2_u, -42.75, 302.60, 16.91
      C3,1111.88,439.64,6.30
      C1,841.88,664.71,5.25
      C2_down,444.26,222.78,10.19
      C4,187.46,54.6,1.62
      C5,151.59,23.53,1.73
      C7,147.22,24.11,1.58
      C6,143.63,22.18,1.00
      C2_up,116.12,175.81,6.01

   .. code-block:: text
      :caption: data/para/experiment_273K.csv

      #temperature 272.25
      #magnetic_field 11.75
      #isotope 1H
      assignment,shift (ppm),width (Hz),area ()
      H3,190.20,1443.71,6.12
      H4,40.94,43.31,2.15
      H5,19.94,17.29,1.98
      H6,18.62,17.15,1.00
      H2_d,5.90,170.71,18.19
      H1,-21.31,673.93,5.30
      H2_u,-41.08,257.85,15.70
      C3,1091.07,432.16,7.39
      C1,826.9,704.95,8.19
      C2_down,435.26,204.83,5.59
      C4,185.66,66.4,1.80
      C5,150.64,34.56,1.82
      C7,146.4,13.55,1.76
      C6,142.74,15.56,1.00
      C2_up,113.57,181.71,7.88

   .. code-block:: text
      :caption: data/para/experiment_278K.csv

      #temperature 277.29
      #magnetic_field 11.75
      #isotope 1H
      assignment,shift (ppm),width (Hz),area ()
      H3, 186.13, 1210.87, 6.12
      H4, 39.96, 38.34, 2.00
      H5, 19.56, 16.75, 1.96
      H6, 18.29, 16.86, 1.00
      H2_d, 5.80, 144.71, 17.70
      H1, -19.78, 578.36, 5.31
      H2_u, -39.50, 244.28, 16.18
      C3,1071.64,443.31,6.46
      C1,811.52,582.7,7.11
      C2_down,427.15,186.53,6.63
      C4,184.49,55.4,1.81
      C5,150.39,54.63,2.30
      C7,146.25,19.84,1.52
      C6,142.54,21,1.00
      C2_up,111.87,195.02,5.80

   .. code-block:: text
      :caption: data/para/experiment_283K.csv

      #temperature 282.30
      #magnetic_field 11.75
      #isotope 1H
      assignment,shift (ppm),width (Hz),area ()
      H3, 182.23, 1034.66, 6.30
      H4, 39.03, 39.05, 2.07
      H5, 19.19, 16.99, 2.04
      H6, 17.99, 17.14, 1.00
      H2_d, 5.72, 127.28, 17.99
      H1, -18.34, 509.89, 5.42
      H2_u, -38.01, 227.07, 16.54
      C3,1053.06,432.7,5.12
      C1,796.93,513.77,5.64
      C2_down,418.7,185.03,8.57
      C4,183,50.9,1.88
      C5,149.81,24.94,2.10
      C7,145.77,24.7,1.72
      C6,142.05,19.35,1.00
      C2_up,109.91,207.75,5.95

   .. code-block:: text
      :caption: data/para/experiment_288K.csv

      #temperature 287.30
      #magnetic_field 11.75
      #isotope 1H
      assignment,shift (ppm),width (Hz),area ()
      H3, 178.55, 913.27, 6.32
      H4, 38.14, 38.99, 2.03
      H5, 18.85, 16.37, 1.95
      H6, 17.69, 17.59, 1.00
      H2_d, 5.64, 112.10, 17.52
      H1, -17.00, 459.31, 5.50
      H2_u, -36.62, 215.61, 16.43
      C3,1035.11,438.69,6.05
      C1,782.9,374.19,7.14
      C2_down,410.9,191.69,6.49
      C4,181.97,102.4,3.13
      C5,149.26,20.29,1.96
      C7,145.36,26.46,1.60
      C6,141.61,17.44,1.00
      C2_up,107.77,188.25,9.24

   .. code-block:: text
      :caption: data/para/experiment_293K.csv

      #temperature 292.28
      #magnetic_field 11.75
      #isotope 1H
      assignment,shift (ppm),width (Hz),area ()
      H3, 174.94, 742.30, 5.41
      H4, 37.29, 38.36, 2.08
      H5, 18.52, 16.69, 2.03
      H6, 17.41, 17.78, 1.00
      H2_d, 5.56, 103.15, 18.04
      H1, -15.71, 410.35, 5.62
      H2_u, -35.27, 194.43, 16.28
      C3,1017.69,379.37,4.73
      C1,769.07,467.49,5.35
      C2_down,403.47,168.78,5.69
      C4,180.92,53.1,1.91
      C5,148.68,13.24,1.69
      C7,144.85,21.6,1.28
      C6,141.11,14.84,1.00
      C2_up,105.93,173.96,5.94

   .. code-block:: text
      :caption: data/para/experiment_298K.csv

      #temperature 297.22
      #magnetic_field 11.75
      #isotope 1H
      assignment,shift (ppm),width (Hz),area ()
      H3, 171.39, 712.65, 6.52
      H4, 36.48, 39.37, 2.04
      H5, 18.20, 17.53, 1.97
      H6, 17.15, 18.05, 1.00
      H2_d, 5.49, 92.86, 17.77
      H1, -14.49, 386.48, 5.83
      H2_u, -33.99, 198.25, 16.58
      C3,1000.56,577.26,8.88
      C1,756,390.25,5.41
      C2_down,396.22,159.62,5.59
      C4,179.74,71.6,1.80
      C5,148.28,22.5,1.91
      C7,144.55,26.29,1.27
      C6,140.81,18.95,1.00
      C2_up,104.28,212.78,6.73

   .. code-block:: text
      :caption: data/para/experiment_303K.csv

      #temperature 302.17
      #magnetic_field 11.75
      #isotope 1H
      assignment,shift (ppm),width (Hz),area ()
      H3,168.06,639.92,6.39
      H4, 35.71, 39.49, 2.04
      H5, 17.90, 17.39, 1.92
      H6, 16.89, 19.03, 1.00
      H2_d, 5.42, 90.16, 17.77
      H1, -13.36, 348.60, 5.50
      H2_u, -32.78, 177.80, 16.25
      C3,984.3,358.71,5.85
      C1,742.99,327.08,6.98
      C2_down,389.21,146.89,6.44
      C4,178.7,40.6,1.58
      C5,147.84,17.22,1.82
      C7,144.22,38.28,1.77
      C6,140.48,13.69,1.00
      C2_up,102.65,206.67,7.48

1. Variable-temperature susceptibility fitting
----------------------------------------------

This workflow fits the temperature dependence of the susceptibility across the
twelve measured spectra. The configuration reads hyperfine tensors from the DFT
output (``hyperfine: method: dft``) and the ab-initio susceptibility from the
ORCA output (``susc_vt: ab_initio_file``):

.. code-block:: yaml
   :caption: run.yml

   project:
     name: fe_vt_fit

   hyperfine:
     method: dft
     file: data/hfc/hyperfine.out
     paramagnetic_centre: [9.122358, 5.108520, 11.946464]
     spin: 2.0

   nuclei:
     include_groups: ['H4', 'H5', 'H6', 'C4', 'C5', 'C6', 'C7']

   assignment:
     method: fixed

   diamagnetic:
     method: csv
     file: data/dia/diamagnetic_shifts.csv

   experiment:
     files: [data/para/*K.csv]

   chem_labels:
     file: data/labels/chemical_labels.csv

   susc_vt:
     method: vt_2nd_order
     tip_type: fix_tip_from_ab_initio
     ab_initio_file: data/chi/susceptibility.out
     ab_initio_format: orca_nev
     variables:
       iso:
         intercept: [fit, 0.01]
         slope: [fit, 0.01]
       ax:
         intercept: [fit, 0.01]
         slope: [fit, 0.01]
       rh:
         intercept: [fix, 0.0]
         slope: [fix, 0.0]

   susc_fit:
     type: isoaxrh
     variables:
       iso: [fit, 0.01]
       ax: [fit, 0.01]
       rh_over_ax: [fix, 0.00]
     average_shifts: 'all'

Key points of this configuration:

``hyperfine: method: dft``
   Reads the full hyperfine tensors from ``hyperfine.out`` instead of computing
   them from geometry. ``spin: 2.0`` is the total spin *S* = 2 of high-spin
   Fe(II).

``susc_vt``
   Models the temperature dependence of the susceptibility to second order
   (``vt_2nd_order``), fixing the temperature-independent paramagnetism (TIP)
   from the ab-initio susceptibility in ``susceptibility.out``
   (``ab_initio_format: orca_nev``).

Run the variable-temperature fit:

.. tab-set::

   .. tab-item:: Command line
      :sync: cli

      .. code-block:: bash

         simpnmr --hide fit_susc --susc_units 'cm3 mol-1' run.yml

      ``--hide`` saves the figures without displaying them interactively;
      ``--susc_units`` states the units of the susceptibility input explicitly.

   .. tab-item:: Desktop app
      :sync: gui

      Launch ``simpnmr-gui``, **Open YAML…** this ``run.yml``, and set the
      **Workflow** to **Fit susceptibility**. The ``susc_vt`` block populates
      the variable-temperature section of the form. Choose the susceptibility
      units in that section (equivalent to ``--susc_units``), then click
      **▶ Run**.

Among the outputs is an ``isoaxrh``-type fit file summarising the fitted tensor
as a function of temperature — used as the input to the next workflow.

2. Extracting a spin Hamiltonian
--------------------------------

The fitted susceptibility can be converted into spin-Hamiltonian parameters.
Take the ``iso``/``ax``/``rh`` fit produced above (an example is shown below;
save it as ``isoaxrh_fit.csv``):

.. code-block:: text
   :caption: isoaxrh_fit.csv

   # Data reported in Curie-normalised chiT (dimensionless) Units
   # spin 2.0
   type,intercept,slope,intercept_err,slope_err,adj_r2
   iso,4.314820,383.105186,0.498071,138.244569,0.970155
   ax,1.743901,323.957098,0.150237,41.699677,0.976824
   rh,0.000000,0.000000,0.000000,0.000000,

Then run:

.. code-block:: bash

   simpnmr get_sh --spin 2.0 isoaxrh_fit.csv

.. note::

   ``get_sh`` is a command-line utility and has no equivalent in the desktop
   GUI, which covers the **Predict** and **Fit susceptibility** workflows.

Next steps
----------

- See :doc:`dy_point_dipole` for the lanthanide workflow, where hyperfine
  tensors are approximated from geometry rather than DFT.
- Read :doc:`../user_guide/workflows` for an overview of all workflows.
- Consult :ref:`input_files` when adapting ``run.yml`` to your own system.
