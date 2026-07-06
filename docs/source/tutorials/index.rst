Tutorials
=========


This section provides two hands-on tutorials, each following one complex through
the SimpNMR workflows. They differ in how the hyperfine coupling is obtained —
the point-dipole approximation for a lanthanide, or DFT-calculated tensors for a
transition metal. Each tutorial is self-contained: the input files are shown
inline, so you can copy them into a working directory and run the workflows
immediately — there is no bundle to download.

.. toctree::
   :maxdepth: 1
   :hidden:

   dy_point_dipole
   fe_dft_hfc

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Dy(III) — point-dipole approximation
      :link: dy_point_dipole
      :link-type: doc

      A dysprosium(III) complex: shift prediction, PCS isosurface, prediction
      with relaxation, and susceptibility fitting, with hyperfine tensors
      computed from geometry (``method: pdip``).

   .. grid-item-card:: Fe(II) — DFT hyperfine tensors
      :link: fe_dft_hfc
      :link-type: doc

      An iron(II) complex: variable-temperature susceptibility fitting and
      spin-Hamiltonian extraction, with hyperfine tensors read from a DFT
      calculation (``method: dft``).

Notes on the examples
^^^^^^^^^^^^^^^^^^^^^^

- Input files are reproduced inline in each tutorial and created by copy-paste.
- A few large ab-initio or spectrometer output files are too big to paste;
  those are offered as individual file downloads where they are used.
- Output directories are generated locally when you run each workflow.

Related documentation
^^^^^^^^^^^^^^^^^^^^^

- :doc:`../user_guide/installation`
- :doc:`../user_guide/workflows`
- :doc:`../user_guide/input_files`
- :doc:`../user_guide/faq`
