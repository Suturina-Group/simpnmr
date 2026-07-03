Workflows
=========

This section describes the main command-line workflows provided by ``simpnmr``.
Each workflow is configured using a strict YAML input file: unknown keys are not
allowed and will result in a configuration error.

For a complete reference of available configuration blocks and their semantics,
see :ref:`input_files`.

Conventions
-----------

- Commands are shown as they should be typed in a terminal.
- ``<input.yml>`` denotes a user-supplied YAML configuration file.
- Paths in YAML may be absolute or relative to the working directory.

pNMR Prediction
---------------

Runs a pNMR prediction workflow using supplied tensors and molecular structure
information.

Run
^^^

.. code-block:: console

   simpnmr predict <input.yml>

Minimal Input Example
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   project:
     name: run_name

   nuclei:
     include: H

   hyperfine:
     method: dft
     file: hfc/file.out

   susceptibility:
     file: chi/file.out
     format: orca_nev
     temperatures: [100, 200, 300]

.. note::

   In prediction workflows, list-like inputs (e.g. susceptibility tensors) are processed positionally.

Optional Additions
^^^^^^^^^^^^^^^^^^

You may additionally supply optional blocks to refine the prediction workflow.
Refer to :ref:`input_files` for the full contract and conditional requirements.

- ``chem_labels``: provide chemical grouping labels (enables averaging).
- ``diamagnetic`` / ``diamagnetic_ref``: include diamagnetic corrections.
- ``relaxation``: apply relaxation-based line broadening or weighting.

Magnetic susceptibility fitting
-------------------------------

Fits a magnetic susceptibility tensor model to experimental data.

Run
^^^

.. code-block:: console

   simpnmr fit_susc <input.yml>

Minimal input example
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   project:
     name: fit_run

   hyperfine:
     method: dft
     file: hfc/file.out

   chem_labels:
     file: chem_labels.csv

   nuclei:
     include: H

   experiment:
     files: exp/peaks_*.csv

   assignment:
     method: fixed

   susc_fit:
     type: isoaxrh
     input_units: A3
     variables:
       iso: [fit, 0.0]
       ax: [fit, 0.0]
       rh_over_ax: [fix, 0.0]

Optional additions
^^^^^^^^^^^^^^^^^^

- ``susc_vt``: temperature-dependent susceptibility fitting (optional and model-dependent).
- ``diamagnetic`` / ``diamagnetic_ref``: include diamagnetic corrections.
- ``assignment`` with ``permute``: explore assignments within user-defined groups.
- ``susc_fit:input_units``: optionally specify YAML fit-variable units as ``A3``,
  ``cm3 mol-1``, or Curie-normalised ``reduced`` values.

Notes
^^^^^

- Some optional blocks are model- or method-dependent (e.g. quantum numbers for
  certain hyperfine sources, or TIP handling in ``susc_vt``).
- Ensure that configuration inputs are ordered consistently
  where positional processing is used.
