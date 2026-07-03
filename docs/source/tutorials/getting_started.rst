.. _getting_started:

Getting started: your first prediction
======================================

This tutorial walks through a complete first run of ``simpnmr``, from
installation to inspecting the generated outputs. It uses the
``01_Shift_Prediction`` example from the tutorial bundle, which predicts the
paramagnetic :sup:`1`\ H NMR shifts of a dysprosium complex from its molecular
structure and a magnetic susceptibility tensor.

By the end of this tutorial you will know how to:

- install ``simpnmr`` and verify the installation,
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

For alternative installation routes, see :doc:`../user_guide/installation`.

Get the example files
---------------------

Download the tutorial example bundle
(:download:`examples.zip <../_downloads/examples.zip>`), unpack it, and move
into the first example directory:

.. code-block:: bash

   unzip examples.zip
   cd examples/01_Shift_Prediction

The example directory contains a configuration file and a data directory:

.. code-block:: text

   01_Shift_Prediction/
   ├── run.yml
   └── data/
       ├── chi/susceptibility.csv       # magnetic susceptibility tensor
       ├── dia/diamagnetic_shifts.csv   # diamagnetic reference shifts
       ├── hfc/structure.xyz            # molecular structure
       └── labels/chemical_labels.csv   # atom → chemical group mapping

The configuration file
----------------------

Every ``simpnmr`` workflow is driven by a single YAML input file. The
``run.yml`` for this example reads:

.. code-block:: yaml

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

From inside ``01_Shift_Prediction/``, run:

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

- Try the remaining examples in the bundle — see :doc:`using_examples` for
  the commands used to run each workflow.
- Read :doc:`../user_guide/workflows` for an overview of all available
  workflows, including susceptibility fitting with ``simpnmr fit_susc``.
- Consult :ref:`input_files` when adapting ``run.yml`` to your own system.
