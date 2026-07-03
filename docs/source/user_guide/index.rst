User Guide
==========


.. toctree::
   :maxdepth: 1
   :hidden:

   whats_new
   installation
   workflows
   input_files
   standalone_cli
   output_files
   theory
   faq

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: What's New
      :link: whats_new
      :link-type: doc

      Major features and improvements in the current development branch.

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Install ``simpnmr`` and verify that the command-line interface is available.

   .. grid-item-card:: Workflows
      :link: workflows
      :link-type: doc

      Overview of prediction and susceptibility fitting workflows supported by ``simpnmr``.

   .. grid-item-card:: Input Files
      :link: input_files
      :link-type: doc

      Detailed description of the YAML configuration format and available input blocks.

   .. grid-item-card:: Standalone CLI Utilities
      :link: standalone_cli
      :link-type: doc

      Documentation of stable command-line tools that operate directly on files
      and do not require YAML pipeline configuration.

   .. grid-item-card:: Output Files
      :link: output_files
      :link-type: doc

      Description of generated output files and their structure.

   .. grid-item-card:: Theory
      :link: theory
      :link-type: doc

      Theoretical background underlying pNMR prediction and susceptibility fitting.

   .. grid-item-card:: FAQ
      :link: faq
      :link-type: doc

      Common questions, caveats, and troubleshooting tips.


Quick start
^^^^^^^^^^^

After installing ``simpnmr`` (see :ref:`installation`), you can explore the command line interface:

.. code-block:: bash

    simpnmr -h
    simpnmr predict -h
    simpnmr fit_susc -h

As a minimal example using a YAML configuration file:

.. code-block:: bash

    simpnmr predict your_config.yml

For details of the available options and required input fields, see the :ref:`input_files` page.

Getting help
^^^^^^^^^^^^

If something does not work as expected, please:

- Check the :ref:`faq` for answers to common questions and error messages.
- See :doc:`Bugs <../developer_guide/bugs>` for how to report issues or unexpected behaviour.
- If you would like to contribute fixes or new features, read :doc:`Contributing <../developer_guide/contributing>`.
