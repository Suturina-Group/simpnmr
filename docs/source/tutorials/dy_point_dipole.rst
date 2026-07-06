.. _tutorial_dy:

Dy(III) complex: the point-dipole approximation
===============================================

This tutorial works through the main SimpNMR workflows on a **dysprosium(III)
complex**, using the **point-dipole approximation** for the hyperfine coupling.

For a lanthanide such as Dy(III), the unpaired electrons occupy core-like 4f
orbitals that are well shielded from the ligands. The electron–nucleus
hyperfine interaction is then dominated by the through-space *dipolar* coupling,
which depends only on the position of each nucleus relative to the paramagnetic
centre. SimpNMR computes this directly from the molecular geometry
(``hyperfine: method: pdip``) — no quantum-chemistry hyperfine calculation is
required. (For transition-metal complexes, where covalent delocalisation makes
the Fermi-contact term important, hyperfine tensors are instead read from a DFT
calculation; see :doc:`fe_dft_hfc`.)

The tutorial covers three workflows on the same complex:

.. contents:: On this page
   :local:
   :depth: 1
   :backlinks: none

Prerequisites
-------------

Install ``simpnmr`` by following the :doc:`../user_guide/installation` guide
(which covers pip as well as the double-click Windows and macOS apps), then
check it is available:

.. code-block:: bash

   simpnmr --version

Command line or the desktop app
-------------------------------

Every workflow below can be run either from the terminal (the ``simpnmr``
command) or from the **desktop GUI**. Each step shows both — pick whichever tab
you prefer.

The GUI presents the same options as a form and can open and save the very
``run.yml`` files shown in this tutorial. To use it, install the optional GUI
dependencies and launch it:

.. code-block:: bash

   pip install "simpnmr[gui]"
   simpnmr-gui

The window has a toolbar (**New**, **Open YAML…**, **Save YAML**, **Save As…**,
**▶ Run**, **■ Stop**); a scrollable form on the left whose sections mirror the
``run.yml`` blocks (Hyperfine, Nuclei, Diamagnetic, Susceptibility, …) with a
**Workflow** selector at the top (**Predict** / **Fit susceptibility**); and a
3D structure viewer with a live log panel on the right. The general pattern is
always the same: **Open YAML…** → choose your ``run.yml`` (the form fills in) →
set the **Workflow** mode → click **▶ Run**; output streams in the log panel and
files are written to the project folder.

The complex and its input files
-------------------------------

Create a working directory with the ``data/`` sub-folders that will hold the
input files:

.. code-block:: bash

   mkdir dy-tutorial
   cd dy-tutorial
   mkdir -p data/hfc data/labels data/dia

The workflows below share three input files: the molecular structure, the
chemical-label map, and the diamagnetic reference shifts. The magnetic
susceptibility is **not** supplied as a file here — in the first workflow it is
generated from a crystal-field parameter, and in the last it is fitted. Create
each file with a plain-text editor, copying the contents exactly (use the copy
button in the top-right of each block).

**The molecular structure** — an XYZ file: the first line is the atom count, the
second is a comment (blank here), and each remaining line is an element symbol
followed by its Cartesian coordinates in Å:

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

**The chemical-label map** — assigns each hydrogen atom to a chemical group so
that symmetry-equivalent nuclei are averaged together:

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

**The diamagnetic reference shifts** — measured on an isostructural diamagnetic
analogue, one value per chemical group (ppm):

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

1. Predicting the paramagnetic shifts
-------------------------------------

The standard prediction workflow computes paramagnetic :sup:`1`\ H shifts from
the structure and a susceptibility tensor. Here the susceptibility is not read
from a file but **generated from a single crystal-field parameter**, the
second-rank axial Stevens parameter B²₀ (set to −100 cm⁻¹), using Bleaney theory.
Create the configuration file ``run.yml`` next to the ``data/`` folder:

.. code-block:: yaml
   :caption: run.yml

   project:
     name: output

   hyperfine:
     method: pdip
     file: data/hfc/structure.xyz
     paramagnetic_centre: Dy1
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
     method: bleaney
     bleaney:
       B20: -100.0
       B22: 0.0
       alpha: 0.0
       beta: 0.0
       gamma: 0.0
     temperatures: 302.15

The blocks have the following roles:

``project``
   Names the run. All outputs are written to a directory with this name
   (here, ``output/``).

``hyperfine``
   ``method: pdip`` computes point-dipole hyperfine tensors directly from the
   structure. The paramagnetic centre is given as an atom label, ``Dy1``, which
   is resolved to its coordinates from ``structure.xyz`` (you could instead give
   explicit ``[x, y, z]`` coordinates). The Dy(III) quantum numbers are
   ``spin`` = 5/2 for the effective treatment, ``orbit`` = 5, and
   ``total_momentum_J`` = 15/2.

``nuclei``
   Selects which nuclei to predict — here all :sup:`1`\ H nuclei.

``diamagnetic``
   Supplies the diamagnetic reference shifts added to the paramagnetic
   contributions.

``chem_labels``
   Maps individual atom labels to chemical groups so equivalent nuclei are
   averaged.

``susceptibility``
   ``method: bleaney`` builds the susceptibility tensor from the second-rank
   crystal-field parameters using Bleaney theory, instead of reading it from a
   file. B²₀ (here −100 cm⁻¹) is the crystal-field parameter in front of the
   axial second-rank Stevens operator, B²₂ the rhombic parameter, and α/β/γ the
   orientation of the crystal-field frame; the isotropic Landé g\ :sub:`J` is
   derived from the Dy(III) quantum numbers above. These crystal-field
   parameters can be estimated, for example, from the emission spectrum of an
   isostructural Eu(III) complex. ``temperatures`` gives the temperature(s) at
   which shifts are evaluated.

.. note::

   The YAML input is strict: unknown keys are rejected. See :ref:`input_files`
   for the full reference of configuration blocks.

Run the prediction:

.. tab-set::

   .. tab-item:: Command line
      :sync: cli

      From inside ``dy-tutorial/``:

      .. code-block:: bash

         simpnmr predict run.yml

      ``simpnmr`` logs each step and finishes with ``Job finished
      successfully``.

      .. dropdown:: Useful command-line options

         .. code-block:: bash

            simpnmr --hide predict run.yml      # save figures without displaying them
            simpnmr --dry-run predict run.yml   # validate the input file, then exit
            simpnmr --verbose predict run.yml   # enable debug logging

   .. tab-item:: Desktop app
      :sync: gui

      1. Launch the app: ``simpnmr-gui``.
      2. Click **Open YAML…** and select the ``run.yml`` you created — the form
         fills in from it.
      3. In the **Workflow** box at the top, make sure **Predict** is selected.
      4. Click **▶ Run**. Each step is logged in the panel on the right, ending
         with ``Job finished successfully``; the loaded structure is shown in
         the 3D viewer.

All results are written to the ``output/`` directory:

.. code-block:: text

   output/
   ├── peak_data_302.15_K.csv               # per-group averaged shift components
   ├── hyperfines_and_shifts_302.15_K.csv   # per-atom hyperfines and shift terms
   ├── susceptibility_tensor.csv            # susceptibility tensor used
   ├── shift_vs_intensity_302.15_K.csv      # simulated spectrum data
   ├── pcs_isosurf_302.15_K.cube            # PCS isosurface (cube file)
   ├── pred_spectrum_302.15_K.pdf           # predicted spectrum
   ├── pred_shift_spread_302.15_K.pdf       # shift spread per chemical group
   ├── pred_mean_components_302.15_K.pdf    # shift decomposition per group
   ├── structure.xyz                        # structure with atom labels
   └── chemcraft_structure.xyz              # structure for ChemCraft

Because there is no ``relaxation`` block, no relaxation model is applied: there
are no R\ :sub:`1`/linewidth relaxation-decomposition plots, and the predicted
peaks are given a small cosmetic display width (a fixed fraction of the shift
range) purely so the spectrum is drawable. Add a ``relaxation`` block
(workflow 2) to compute physical linewidths and relaxation rates.

The two most useful files are ``peak_data_302.15_K.csv`` (one row per chemical
group, with the total shift and its diamagnetic / pseudocontact / Fermi-contact
decomposition, plus the cosmetic ``linewidth_avg_auto`` display width; physical
linewidths and R\ :sub:`1`/R\ :sub:`2` rates appear only with a relaxation
block, as in workflow 2) and
``hyperfines_and_shifts_302.15_K.csv`` (one row per nucleus, before averaging).
The run also writes ``pcs_isosurf_302.15_K.cube`` — an isosurface of the
pseudocontact-shift field that can be opened in molecular-visualisation software
to display the field around the complex. For a full description of every output
file, see :doc:`../user_guide/output_files`.

The predicted spectrum gives a quick visual check of the result:

.. figure:: /_static/dy_pred_spectrum.png
   :alt: Predicted 1H NMR spectrum of the Dy(III) complex
   :width: 95%
   :align: center

   ``pred_spectrum_302.15_K.pdf`` — the predicted :sup:`1`\ H spectrum. Each
   chemical group appears at its averaged paramagnetic shift, spanning roughly
   +130 to −140 ppm. Because no relaxation model is set, the peaks carry the
   cosmetic display width described above.

2. Adding a relaxation model
----------------------------

Prediction can be extended with a relaxation model to produce predicted
linewidths and R\ :sub:`1`/R\ :sub:`2` rates alongside the shifts — computed
entirely from theory, with **no experimental data required**. It uses the same
shared input files as workflow 1; you add only a ``relaxation`` block, giving the
electronic and rotational correlation times together with the field and
temperature the rates apply to:

.. code-block:: yaml
   :caption: run.yml

   project:
     name: output

   hyperfine:
     method: pdip
     file: data/hfc/structure.xyz
     paramagnetic_centre: Dy1
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
     method: bleaney
     bleaney:
       B20: -100.0
       B22: 0.0
       alpha: 0.0
       beta: 0.0
       gamma: 0.0
     temperatures: 302.15

   relaxation:
     model: sbm curie
     temperature: 302.15
     magnetic_field_tesla: 4.7
     T1e: 0.2e-12
     T2e: 0.2e-12
     tR: 140e-12

The ``relaxation`` block adds the transverse (linewidth) and longitudinal
(R\ :sub:`1`) rates and their SBM-dipolar / contact / Curie decomposition,
including the ``pred_r1_decomposition`` and ``pred_linewidth_decomposition``
plots that were absent in workflow 1.

.. dropdown:: Alternative: estimate τ_R from the solvent instead of fixing it

   The rotational correlation time ``tR`` is often not known in advance. Rather
   than fixing it, you can have it estimated from the molecular shape and the
   solvent viscosity at the given temperature. Replace the ``tR`` line with a
   ``tau_r_method`` and a ``tau_r_solvent``:

   .. code-block:: yaml
      :caption: run.yml (relaxation block)

      relaxation:
        model: sbm curie
        temperature: 302.15
        magnetic_field_tesla: 4.7
        tau_r_method: ellipsoid     # or 'beadshell'
        tau_r_solvent: methanol     # from the built-in solvent database
        T1e: 0.2e-12
        T2e: 0.2e-12

   ``tau_r_method`` selects the hydrodynamic model (``ellipsoid`` — a Perrin
   ellipsoid fit to the molecular shape — or the finer ``beadshell`` model), and
   ``tau_r_solvent`` looks up the solvent viscosity (``methanol``, ``CDCl3``,
   ``D2O``, ``DMSO``, ``CD2Cl2``, …). To use a viscosity that is not in the
   database,
   give ``tau_r_eta`` (Pa·s) instead of ``tau_r_solvent``. The estimated τ_R is
   printed to the log and recorded in the ``peak_data`` header.

Then run:

.. tab-set::

   .. tab-item:: Command line
      :sync: cli

      .. code-block:: bash

         simpnmr predict run.yml

   .. tab-item:: Desktop app
      :sync: gui

      Launch ``simpnmr-gui``, **Open YAML…** this ``run.yml``, keep the
      **Workflow** on **Predict**, and click **▶ Run**. The ``relaxation`` block
      is picked up automatically, so the run also produces the linewidth and
      relaxation-rate outputs.

With the relaxation model the predicted peaks now carry physical linewidths:

.. figure:: /_static/dy_pred_spectrum_relax.png
   :alt: Predicted 1H NMR spectrum of the Dy(III) complex with relaxation
   :width: 95%
   :align: center

   ``pred_spectrum_302.15_K.pdf`` with the relaxation model. Compare with
   workflow 1: the peaks farthest from the diamagnetic region (``aax`` at
   +130 ppm, ``cax`` at −140 ppm) are strongly broadened and shortened by
   Curie/R\ :sub:`2` relaxation, while the peaks near the centre stay sharp —
   the position-dependent linewidths the cosmetic display width could not show.

3. Assignment of experimental peaks and fitting of the magnetic susceptibility
------------------------------------------------------------------------------

The inverse of a prediction: given a measured peak list, fit the susceptibility
tensor. This replaces the ``susceptibility`` block with an ``assignment`` block
(which resolves ambiguous peak assignments) and a ``susc_fit`` block (which
defines the fitted tensor model), and adds an ``experiment`` block pointing at
the measured data. It reuses the same ``structure.xyz``,
``chemical_labels.csv`` and ``diamagnetic_shifts.csv`` as workflows 1 and 2.

.. code-block:: yaml
   :caption: run.yml

   project:
     name: output

   hyperfine:
     method: pdip
     file: data/hfc/structure.xyz
     paramagnetic_centre: Dy1
     spin: 2.5
     orbit: 5
     total_momentum_J: 7.5

   nuclei:
     include: H

   assignment:
     method: permute
     groups:
       - [aax, aeq]

   diamagnetic:
     method: csv
     file: data/dia/diamagnetic_shifts.csv

   experiment:
     files: data/para/experiment.csv
     spectrum_files: data/para/raw_spectrum.csv
     exp_reference: 0.56

   chem_labels:
     file: data/labels/chemical_labels.csv

   susc_fit:
     type: isoaxrh
     variables:
       iso: [fix, 0.00]
       ax: [fit, 0.001]
       rh_over_ax: [fix, 0.00]
     average_shifts: 'all'

Create the measured peak list in a new ``data/para/`` folder
(``mkdir -p data/para``):

.. code-block:: text
   :caption: data/para/experiment.csv

   #temperature 302.15
   #magnetic_field 4.7
   #isotope 1H
   assignment,shift (ppm),width (Hz),area (),r1 (Hz)
   aax,82.89,587.31,4108.48,3013
   py5,24.14,97.24,5139.3,126
   py3,23.91,102.86,3914.16,126
   py4,21.6,74.05,4609,59
   aeq,6.73,258.23,6320.07,425
   caxp,0.56,308.51,4984.95,0
   ceq,-42.36,182.53,4452.48,400
   ceqp,-49.12,224.78,4826.13,430
   cax,-97.33,252.06,4772.37,528

The raw spectrum (``spectrum_files``, used to overlay the predicted and measured
spectra) is a large two-column trace, too big to paste — download it and save it
as ``data/para/raw_spectrum.csv``:

* :download:`raw_spectrum.csv <../_downloads/examples/03_Shift_Prediction_With_Relaxation/data/para/raw_spectrum.csv>`

Run the fit:

.. tab-set::

   .. tab-item:: Command line
      :sync: cli

      .. code-block:: bash

         simpnmr fit_susc run.yml

   .. tab-item:: Desktop app
      :sync: gui

      Launch ``simpnmr-gui``, **Open YAML…** this ``run.yml``, then in the
      **Workflow** box switch to **Fit susceptibility** (this reveals the
      **Assignment** and fit-option sections). Click **▶ Run** to fit the
      tensor; the fitted shifts and susceptibility appear in the output folder.

Next steps
----------

- See :doc:`fe_dft_hfc` for the transition-metal workflow, where hyperfine
  tensors come from a DFT calculation instead of the point-dipole model.
- Read :doc:`../user_guide/workflows` for an overview of all workflows.
- Consult :ref:`input_files` when adapting ``run.yml`` to your own system.
