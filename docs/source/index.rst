Home
====

``simpnmr`` is a Python package for analysing the solution NMR of paramagnetic
metal complexes with computational chemistry.

Use it to:

- **predict pNMR spectra** — both chemical shifts and paramagnetic relaxation
  enhancement (PRE) — from quantum-chemistry output (ORCA, Gaussian)
- **assist the assignment** of experimental paramagnetic peaks
- **extract the full magnetic susceptibility tensor** from experimental shifts
- **derive spin-Hamiltonian parameters** for mononuclear complexes, from
  variable-temperature (VT) pNMR data


.. grid:: 4
   :gutter: 3

   .. grid-item-card:: Get started
      :link: user_guide/index
      :link-type: doc

      Installation instructions, workflows, and practical guidance for running
      prediction and fitting calculations with ``simpnmr``.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Downloadable example materials for the SimpNMR tutorials and workshop,
      provided as ready-to-use files from the repository.

   .. grid-item-card:: Developer Guide
      :link: developer_guide/index
      :link-type: doc

      Internal architecture, contribution guidelines, and maintenance notes
      for developers and advanced users.

   .. grid-item-card:: Templates
      :link: templates/index
      :link-type: doc

      Downloadable YAML templates for common prediction and fitting workflows.
      Modify the file to run your own calculations with minimal setup.

.. admonition:: License and disclaimer
   :class: caution

   SimpNMR is free software released under the GNU General Public License
   v3.0 (see the ``LICENSE`` file in the repository). It is provided **"as is",
   without warranty of any kind**, express or implied, including but not limited
   to the warranties of merchantability and fitness for a particular purpose. To
   the maximum extent permitted by law, the authors and the Suturina Group
   accept **no liability** for any claim, damage, or other loss arising from the
   use of this software or of any results it produces. Users are responsible for
   independently validating all computed results before relying on them.

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   user_guide/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide
   :hidden:

   developer_guide/index

.. toctree::
   :maxdepth: 1
   :caption: Templates
   :hidden:

   templates/index