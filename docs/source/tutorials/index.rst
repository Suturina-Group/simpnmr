Tutorials
=========


This section provides hands-on tutorials for the SimpNMR workflows and
workshop. Each tutorial is self-contained: the input files are shown inline on
the page, so you can copy them into a working directory and run the workflow
immediately — there is no bundle to download.

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   using_examples

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: getting_started
      :link-type: doc

      Step-by-step first run: install ``simpnmr``, assemble the inputs, run a
      shift prediction on an example system, and inspect the outputs.

   .. grid-item-card:: Worked examples
      :link: using_examples
      :link-type: doc

      Six worked examples covering prediction, PCS isosurfaces, susceptibility
      fitting, and spin-Hamiltonian extraction, with all input files inline and
      the command to run each one.

Notes on the examples
^^^^^^^^^^^^^^^^^^^^^^

- Most examples are workflow directories containing a ``data/`` folder and a
  ``run.yml`` configuration file, both reproduced inline in the tutorials.
- A few examples use large ab-initio or spectrometer output files that are too
  big to paste; those are offered as individual file downloads on the relevant
  example.
- Output directories are generated locally when you run each workflow.

Related documentation
^^^^^^^^^^^^^^^^^^^^^

- :doc:`../user_guide/installation`
- :doc:`../user_guide/workflows`
- :doc:`../user_guide/input_files`
- :doc:`../user_guide/faq`
