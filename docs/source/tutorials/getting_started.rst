.. _getting_started:

Getting started: your first prediction
======================================

This tutorial walks through a complete first run of ``simpnmr``, from
installation to inspecting the generated outputs. It predicts the paramagnetic
:sup:`1`\ H NMR shifts of a dysprosium complex from its molecular structure and
a magnetic susceptibility tensor.

Everything you need is created by hand from a few short text files shown on this
page — there is nothing to download. By the end you will know how to:

- install ``simpnmr`` and verify the installation,
- assemble the input files for a prediction,
- read a ``run.yml`` configuration file and understand its blocks,
- run a prediction workflow from the command line,
- locate and interpret the generated output files.

Prerequisites
-------------

``simpnmr`` requires Python 3.10 or later. Install the package from PyPI:

.. code-block:: bash

   pip install simpnmr

Verify the installation by printing the version:

.. code-block:: bash

   simpnmr --version

For alternative installation routes (including the double-click Windows app and
a macOS Automator app), see :doc:`../user_guide/installation`.

Set up the working directory
----------------------------

Create a folder for this tutorial and move into it, then create the ``data/``
sub-folders that will hold the input files:

.. code-block:: bash

   mkdir simpnmr-getting-started
   cd simpnmr-getting-started
   mkdir -p data/chi data/hfc data/labels data/dia

By the end of the next section you will have built this layout:

.. code-block:: text

   simpnmr-getting-started/
   ├── run.yml
   └── data/
       ├── chi/susceptibility.csv       # magnetic susceptibility tensor
       ├── hfc/structure.xyz            # molecular structure
       ├── labels/chemical_labels.csv   # atom → chemical group mapping
       └── dia/diamagnetic_shifts.csv   # diamagnetic reference shifts

Create the input files
----------------------

Create each file below with a plain-text editor (VS Code, Notepad, ``nano``, …),
copying the contents exactly. Use the copy button in the top-right corner of
each block.

**1. The susceptibility tensor** — the magnetic susceptibility of the complex at
the temperature of interest, in Å\ :sup:`3`:

.. code-block:: text
   :caption: data/chi/susceptibility.csv

   chi_xx (Å^3),chi_xy (Å^3),chi_xz (Å^3),chi_yy (Å^3),chi_yz (Å^3),chi_zz (Å^3),Temperature (K)
   0.074458543,0,0,0.074458543,0,-0.148917086,302.15

**2. The diamagnetic reference shifts** — measured on an isostructural
diamagnetic analogue, one value per chemical group (ppm):

.. code-block:: text
   :caption: data/dia/diamagnetic_shifts.csv

   chem_label,shift,
   ceq,2.87,
   cax,3.67,
   ceqp,2.66,
   caxp,2.27,
   aax,4.15,
   aeq,4.08,
   py3,8.09,
   py4,8.19,
   py5,7.75,

**3. The chemical-label map** — assigns each hydrogen atom in the structure to a
chemical group, so that symmetry-equivalent nuclei are averaged together:

.. dropdown:: data/labels/chemical_labels.csv  (27 atoms)

   .. code-block:: text
      :caption: data/labels/chemical_labels.csv

      atom_label,chem_label,
      H1,ceq,
      H2,ceq,
      H3,ceq,
      H4,cax,
      H5,cax,
      H6,cax,
      H7,ceqp,
      H8,ceqp,
      H9,ceqp,
      H10,caxp,
      H11,caxp,
      H12,caxp,
      H13,aeq,
      H14,aax,
      H15,aeq,
      H16,aax,
      H17,aeq,
      H18,aax,
      H19,py3,
      H20,py4,
      H21,py5,
      H22,py3,
      H23,py4,
      H24,py5,
      H25,py3,
      H26,py4,
      H27,py5,

**4. The molecular structure** — an XYZ file. The first line is the atom count,
the second is a comment (blank here), and each remaining line is an element
symbol followed by its Cartesian coordinates in Å:

.. dropdown:: data/hfc/structure.xyz  (67 atoms)

   .. code-block:: text
      :caption: data/hfc/structure.xyz

      67

      Dy1  0.006857  -0.008695  0.057882
      O1  -0.302027  -1.830532  -1.419955
      O2  -1.421572  1.175692  -1.42229
      O3  1.749501  0.631576  -1.405317
      O4  0.37101  -3.709201  -2.441562
      O5  -3.395665  1.573815  -2.408402
      O6  3.066362  2.152547  -2.393889
      N1  1.677607  -0.008754  2.131025
      N2  -0.843693  -1.449175  2.132149
      N3  -0.838461  1.453493  2.129045
      N4  1.889226  -1.691301  0.016201
      N5  -2.403829  -0.782665  0.017119
      N6  0.512716  2.467123  0.016552
      C1  1.388674  -1.051509  3.155166
      C2  -1.602271  -0.679189  3.15723
      C3  0.210929  1.728028  3.150359
      C4  0.349952  -2.080411  2.724242
      C5  -1.978465  0.732834  2.724694
      C6  1.622671  1.340697  2.723084
      C7  3.0129  -0.237982  1.554704
      C8  -1.711124  -2.487906  1.551347
      C9  -1.311353  2.719107  1.546985
      C10  3.011522  -1.487697  0.712959
      C11  -2.792311  -1.855265  0.713582
      C12  -0.228617  3.340474  0.704336
      C13  4.100166  -2.353457  0.609131
      C14  -4.090623  -2.355398  0.61663
      C15  -0.022604  4.715666  0.596268
      C16  4.010931  -3.439383  -0.259915
      C17  -4.988202  -1.724653  -0.243025
      C18  0.96996  5.17746  -0.265821
      C19  2.850395  -3.62226  -1.00971
      C20  -4.564452  -0.626242  -0.988737
      C21  1.717669  4.260169  -1.001965
      C22  1.811887  -2.712726  -0.84068
      C23  -3.252412  -0.193021  -0.829114
      C24  1.446994  2.906539  -0.83121
      C25  0.522848  -2.779727  -1.642952
      C26  -2.662272  0.95673  -1.629424
      C27  2.164454  1.821142  -1.618161
      H1  2.31528  -1.587822  3.415912
      H2  -2.527794  -1.216233  3.419855
      H3  0.214285  2.80021  3.404525
      H4  1.063935  -0.55658  4.07935
      H5  -1.01028  -0.642445  4.080704
      H6  -0.055828  1.205465  4.078062
      H7  0.077557  -2.691814  3.604836
      H8  -2.371831  1.275332  3.604759
      H9  2.284755  1.410156  3.606479
      H10  0.769145  -2.762295  1.972916
      H11  -2.778561  0.706362  1.973668
      H12  2.007054  2.044642  1.973765
      H13  3.789806  -0.294535  2.336465
      H14  3.253224  0.612921  0.896256
      H15  -2.149835  -3.135409  2.329992
      H16  -1.094164  -3.119474  0.891154
      H17  -1.65281  3.423591  2.32501
      H18  -2.169392  2.495622  0.892255
      H19  4.99888  -2.172676  1.199166
      H20  4.845914  -4.134774  -0.355802
      H21  2.73878  -4.443619  -1.716171
      H22  -4.385361  -3.224578  1.204937
      H23  -6.011081  -2.092533  -0.334059
      H24  -5.222768  -0.110891  -1.686717
      H25  -0.633023  5.40561  1.179275
      H26  1.155636  6.247781  -0.365238
      H27  2.493193  4.572044  -1.700048

The configuration file
----------------------

Every ``simpnmr`` workflow is driven by a single YAML input file. Create
``run.yml`` in the ``simpnmr-getting-started`` directory (next to ``data/``):

.. code-block:: yaml
   :caption: run.yml

   project:
     name: output

   hyperfine:
     method: pdip
     file: data/hfc/structure.xyz
     paramagnetic_centre: [0.006857, -0.008695, 0.057882]
     spin: 2.5
     orbit: 5
     total_momentum_J: 7.5

   nuclei:
     include: H

   diamagnetic:
     method: csv
     file: data/dia/diamagnetic_shifts.csv

   chem_labels:
     file: data/labels/chemical_labels.csv

   susceptibility:
     file: data/chi/susceptibility.csv
     format: csv
     temperatures: 302.15

The blocks have the following roles:

``project``
   Names the run. All outputs are written to a directory with this name
   (here, ``output/``).

``hyperfine``
   Defines how hyperfine coupling tensors are obtained. Here
   ``method: pdip`` computes point-dipole hyperfine tensors directly from the
   structure in ``structure.xyz``, using the given paramagnetic centre
   coordinates and the Dy(III) quantum numbers (``spin``, ``orbit``,
   ``total_momentum_J``). Alternatively, ``method: dft`` reads hyperfine
   tensors from a quantum chemistry output file.

``nuclei``
   Selects which nuclei to predict — here all :sup:`1`\ H nuclei.

``diamagnetic``
   Supplies diamagnetic reference shifts (measured on an isostructural
   diamagnetic analogue) that are added to the paramagnetic contributions.

``chem_labels``
   Maps individual atom labels (``H1``, ``H2``, …) to chemical groups
   (``ceq``, ``cax``, …), so that shifts of symmetry-equivalent nuclei are
   averaged.

``susceptibility``
   Provides the magnetic susceptibility tensor and the temperature(s) at
   which shifts are evaluated.

.. note::

   The YAML input is strict: unknown keys are not allowed and will result in
   a configuration error. See :ref:`input_files` for the full reference of
   configuration blocks.

Run the prediction
------------------

From inside ``simpnmr-getting-started/``, run:

.. code-block:: bash

   simpnmr predict run.yml

``simpnmr`` logs each step to the terminal and finishes with
``Job finished successfully``.

.. dropdown:: Useful command-line options

   .. code-block:: bash

      simpnmr --hide predict run.yml      # save figures without displaying them
      simpnmr --dry-run predict run.yml   # validate the input file, then exit
      simpnmr --verbose predict run.yml   # enable debug logging

Inspect the outputs
-------------------

All results are written to the directory named in the ``project`` block:

.. code-block:: text

   output/
   ├── peak_data_302.15_K.csv               # per-group shifts, linewidths, R1/R2
   ├── hyperfines_and_shifts_302.15_K.csv   # per-atom hyperfines and shift terms
   ├── susceptibility_tensor.csv            # susceptibility tensor used
   ├── shift_vs_intensity_302.15_K.csv      # simulated spectrum data
   ├── pcs_isosurf_302.15_K.cube            # PCS isosurface (cube file)
   ├── pred_spectrum_302.15_K.pdf           # predicted spectrum
   ├── pred_shift_spread_302.15_K.pdf       # shift spread per chemical group
   ├── pred_mean_components_302.15_K.pdf    # shift decomposition per group
   ├── pred_r1_decomposition_302.15_K.pdf   # R1 relaxation decomposition
   ├── pred_linewidth_decomposition_302.15_K.pdf
   ├── structure.xyz                        # structure with atom labels
   └── chemcraft_structure.xyz              # structure for ChemCraft

The two most important files are:

``peak_data_302.15_K.csv``
   One row per chemical group, with the averaged total shift
   (``δ_total_avg``) and its decomposition into diamagnetic (``δ_dia_avg``),
   pseudocontact (``δ_pc_avg``), and Fermi-contact (``δ_fc_spin_only_avg``)
   contributions, along with predicted linewidths and relaxation rates.

``hyperfines_and_shifts_302.15_K.csv``
   One row per nucleus, with coordinates, hyperfine tensor components, and
   the individual shift contributions before averaging.

The predicted spectrum in ``pred_spectrum_302.15_K.pdf`` gives a quick visual
check of the result, and ``pcs_isosurf_302.15_K.cube`` can be rendered in
molecular visualisation software to display the pseudocontact shift field
around the complex.

For a full description of every output file, see
:doc:`../user_guide/output_files`.

Next steps
----------

- Work through the other examples — see :doc:`using_examples`, where each
  workflow's input files are given inline just like this one.
- Read :doc:`../user_guide/workflows` for an overview of all available
  workflows, including susceptibility fitting with ``simpnmr fit_susc``.
- Consult :ref:`input_files` when adapting ``run.yml`` to your own system.
