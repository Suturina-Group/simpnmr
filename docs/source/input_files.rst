.. _input_files:

Input files
===========

This page outlines expected formats of the input files and how ``simpnmr`` parses them.


Input YAML files
----------------

In general, the following rules apply to input files

1. All input files use the ``YAML`` format - see `here <https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started>`_ for a quick tutorial.
2. Any file names specified in an input file can contain absolute or relative path information.
3. Any file names specified in an input file can contain the ``*`` wildcard.
4. Additional keyword entries are ignored.
5. Comment lines (``#``) are ignored by the ``pyaml`` parser.

There is a list of kewords and associated options that are common for both ``predict`` and ``fit`` functions.

.. table:: Table 1: general input file keywords and subkeywords.

    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | Keyword                   | Subkeyword        | Value                    | Description                                                             | Needed for ``predict``| Needed for ``fit``    |
    +===========================+===================+==========================+=========================================================================+=======================+=======================+
    | ``project``               | ``name``          | Directory Name           | Name of directory to which all output files are written                 | |:white_check_mark:|  | |:white_check_mark:|  |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | If directory does not exist then it will be created                     |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``hyperfine``             | ``method``        |                          | Type of HFC input                                                       | |:white_check_mark:|  | |:white_check_mark:|  |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | Options:                                                                |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   | ``dft``                  | - ``method:dft`` specifies DFT calculated hyperfine values will be      |                       |                       |
    |                           |                   |                          |   used                                                                  |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   | ``pdip``                 | - ``method:pdip`` specifies hyperfines will be calculated with the      |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          |   point dipole approximation                                            |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   | ``csv``                  | - ``method:csv`` specifies ``.csv`` data entry - see Hyperfine Format   |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``hyperfine``             | ``file``          | File Name                | File containing hyperfine data and/or structure - Format depends on     | |:white_check_mark:|  | |:white_check_mark:|  |
    |                           |                   |                          | ``method``                                                              |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | Options:                                                                |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | - ``method:dft`` supports Gaussian ``.log`` or Orca ``.out``            |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | - ``method:pdip`` supports .xyz file                                    |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | - ``method:raw`` supports ``.csv`` file - see Hyperfine Format          |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``hyperfine``             | ``average``       | List of chemical labels  | Specifies atoms for which hyperfines are averaged.                      | |:x:|                 | |:x:|                 |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | e.g. ``[Me1, Me2]`` replaces hyperfines for atoms with chemical         |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | labels of ``Me1`` and ``Me2`` with the average of all occurrences of    |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | each label.                                                             |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``hyperfine``             | ``pdip_centre``   | Atomic Label(s)          | Atomic label (including index) of paramagnetic centre                   | |:x:|                 | |:x:|                 |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``hyperfine``             | ``spin``          | Float values             | Total spin S e.g. 2.5 for Dy(III)                                       | |:x:|                 | |:x:|                 |
    |                           |                   |                          |                                                                         |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``hyperfine``             | ``orbit``         | Float values             | Total orbital momentum L e.g. 5                                         | |:x:|                 | |:x:|                 |
    |                           |                   |                          |                                                                         |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``hyperfine``             | ``total_J``       | Float values             | Total momentum J e.g. 5                                                 | |:x:|                 | |:x:|                 |
    |                           |                   |                          |                                                                         |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``chem_labels``           | ``file``          | File Name                | File containing chemical labels, atom labels, and                       | |:white_check_mark:|  | |:white_check_mark:|  |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | optionally chemical math labels used for plotting.                      |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | See :ref:`here <chemlabels_csv>` Format for more information            |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``nuclei``                | ``include``       | List of atom labels      | Specifies which nuclei for which shifts will be calculated              | |:white_check_mark:|  | |:white_check_mark:|  |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   | or an atomic symbol      |                                                                         |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   | e.g. ``[C, H1, H2]``     |                                                                         |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``nuclei``                |``include_groups`` | List of chemical labels  | Specifies which nuclei are considered based on their chemical labels.   |  |:x:|                |  |:x:|                |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | It is an alternative to ``include`` where atom labels are requested     |                       |                       |
    |                           |                   | e.g. ``['C_a', 'Me3']``  |                                                                         |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``diamagnetic``           | ``method``        |                          | Method for calculation of diamagnetic shifts                            | |:x:|                 | |:x:|                 |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | Options:                                                                |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   | ``dft``                  |  - ``method:dft`` specifies DFT calculated shifts                       |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   | ``csv``                  |  - ``method:csv`` specifies ``.csv`` data entry                         |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``diamagnetic``           | ``file``          | File Name                | File containing hyperfine data - Format depends on ``method``           | |:x:|                 | |:x:|                 |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | Options:                                                                |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          |  - ``method:dft`` supports Gaussian ``.log`` or Orca ``.out``           |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          |  - ``method:raw`` supports ``.csv`` file - see :ref:`here <dia_csv>`.   |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``diamagnetic``           | ``average``       | List of chemical labels  | Specifies atoms for which diamagnetic shifts are averaged.              | |:x:|                 | |:x:|                 |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | e.g. ``[Me1, Me2]`` replaces diamagnetic shifts for atoms with chemical |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | labels of ``Me1`` and ``Me2`` with the average of all occurrences of    |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | each label.                                                             |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``diamagnetic_reference`` | ``method``        |                          | Method for calculation of reference diamagnetic shifts                  | |:x:|                 | |:x:|                 |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | Options:                                                                |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |  ``dft``                 |  - ``method:dft`` specifies DFT calculated shifts                       |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |  ``csv``                 |  - ``method:raw`` specifies ``.csv`` data entry                         |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``diamagnetic_reference`` | ``file``          | File Name                | File containing reference diamagnetic data - Format depends on          | |:x:|                 | |:x:|                 |
    |                           |                   |                          | ``method``                                                              |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | Options:                                                                |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          |  - ``method:dft`` supports Gaussian ``.log`` or ORCA ``.out``           |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          |  - ``method:raw`` supports ``.csv`` file - see :ref:`here <dia_csv>`.   |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``experiment``            | ``files``         | File Name(s)             | File(s) containing experimental NMR peak data as ``.csv``.              | |:x:|                 | |:white_check_mark:|  |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | See :ref:`here <exp_csv>` for format information.                       |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``experiment``            | ``spectrum_files``| File Name                | File(s) containing experimental NMR spectrum ``.csv``.                  | |:x:|                 | |:x:|                 |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   |                          | See :ref:`here <exp_csv>` for format information.                       |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``relaxation``            | ``model``         |                          | Specifies a model to be used for PRE calulations.                       | |:x:|                 | |:x:|                 |
    |                           |                   |                          |                                                                         |                       |                       |
    |                           |                   | ``curie``                | Curie and                                    vvvvvv                     |                       |                       |
    |                           |                   | ``sbm``                  | Solomon-Bloembergen-Morgan mechanism                                    |                       |                       |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+
    | ``relaxation``            |``B``              |  Float values            | Magnetic field in Tesla for PRE.                                        | |:white_check_mark:|  | |:white_check_mark:|  |
    |                           |                   |                          |                                                                         | for relaxation        | for relaxation        |
    +---------------------------+-------------------+--------------------------+-------------------------------------------------------------------------+-----------------------+-----------------------+


In addition to the keywords above, ``predict`` subprogram has the following dedicated keywords

.. table:: Table 2: ``predict`` specific input file keywords and subkeywords.

    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | Keyword                     | Subkeyword              | Value                               | Description                                                             |Needed for ``predict``|
    +=============================+=========================+=====================================+=========================================================================+======================+
    | ``susceptibility``          | ``file``                | File Name                           | File(s) containing susceptibility data                                  | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See :ref:`here <general_csv>` for format information.                   |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``susceptibility``          | ``format``              | ``orca_nev``/``orca_cas``/``molcas``| Format of file(s) containing susceptibility data                        | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | /``txt``/``csv``                    |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``format:orca_nev`` uses data from NEVPT2 in an Orca ``.out`` file    |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``format:orca_cas`` uses data from CASSCF in an Orca ``.out`` file    |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``format:molcas`` uses data from an OpenMOLCAS ``.out`` file          |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``format:csv`` uses data from a ``simpnmr`` ``susceptibility.csv``    |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``format:txt`` uses a tensor a ``.txt`` file                          |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See :ref:`here <general_csv>` for format information.                   |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``susceptibility``          | ``temperature``         | Float values                        | Temperature(s) to use from susceptibility file                          | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+

And the ``fit_susc`` has the follwing specific kewords.

.. table:: Table 1: ``fit_susc`` subprogram input file keywords and subkeywords.

    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | Keyword                     | Subkeyword(s)           | Value                               | Description                                                             | Mandatory            |
    +=============================+=========================+=====================================+=========================================================================+======================+
    | ``project``                 | ``name``                | Directory Name                      | Name of directory to which all output files are written                 | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | If the named directory does not exist then it will be created           |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``hyperfine``               | ``method``              | ``dft``/``pdip``/``raw``            | Method for calculation of hyperfines                                    | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``method:dft`` specifies DFT calculated hyperfine values will be      |                      |
    |                             |                         |                                     |   used                                                                  |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``method:pdip`` specifies hyperfines will be calculated with the      |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |   point dipole approximation                                            |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``method:raw`` specifies ``.csv`` data entry - see Hyperfine Format   |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``hyperfine``               | ``file``                | File Name                           | File containing hyperfine data and/or structure - Format depends on     | |:white_check_mark:| |
    |                             |                         |                                     | ``method``                                                              |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``method:dft`` supports Gaussian ``.log`` or Orca ``.out``            |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``method:pdip`` supports .xyz file                                    |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``method:raw`` supports ``.csv`` file - see Hyperfine Format          |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``hyperfine``               | ``average``             | List of Chemical labels             | Specifies atoms for which hyperfines are averaged.                      | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | or list of lists of chemical labels | e.g. ``[Me1, tBu2]`` replaces hyperfine tensors for atoms with chemical |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | labels of ``Me1`` with the average of all atoms bearing that label, and |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | that the same will happen, separately, for ``tBu2``.                    |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | If a list of lists is provided, then a more complex averaging scheme is |                      |
    |                             |                         |                                     | performed.                                                              |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | e.g ``[[Me1, Me2], [tBu1, tBu2]]``                                      |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Implies that the hyperfine tensors of ``Me1`` and ``Me2`` will be       |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | averaged together, as will those of ``tBu1`` and ``tBu2``.              |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``hyperfine``               | ``pdip_centre``         | Atomic Label(s)                     | Atomic label(s) (including index) of electron position in point dipole  | |:x:|                |
    |                             |                         |                                     | approximation.                                                          |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Used only when ``method:pdip``                                          |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``experiment``              | ``files``               | File Name(s)                        | File(s) containing experimental NMR peak data for a given temperature   | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | as ``.csv``                                                             |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See :ref:`here <exp_csv>` for format information.                       |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``experiment``              | ``spectrum_files``      | File Name                           | File(s) containing experimental NMR spectrum as ``.csv``.               | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See :ref:`here <exp_spectrum>` for format information.                  |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``chem_labels``             | ``file``                | File Name                           | File containing chemical labels, atom labels, and optionally            | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | chemical math labels in Mathtext format which are used in plots.        |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See :ref:`here <chemlabels_csv>` Format for more information            |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``nuclei``                  | ``include``             | Atomic symbols                      | Specifies nuclei for which shifts will be calculated/fitted             | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | e.g. ``H, C``                       |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``nuclei``                  | ``include_groups``      | List of atom labels with indexing,  | Specifies nuclei for which shifts will be calculated/fitted             | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | or just X where X is an atomic      |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | symbol and signifies all occurrences|                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | e.g. ``[H25, C12]``                 |                                                                         |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``diamagnetic``             | ``method``              | ``dft``/``pdip``/``raw``            | Method for calculation of diamagnetic shifts                            | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:dft`` specifies DFT calculated shifts                       |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:raw`` specifies ``.csv`` data entry                         |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``diamagnetic``             | ``file``                | File Name                           | File containing hyperfine data - Format depends on ``method``           | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:dft`` supports Gaussian ``.log`` or Orca ``.out``           |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:raw`` supports ``.csv`` file - see :ref:`here <dia_csv>`.   |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``diamagnetic_ref``         | ``method``              | ``dft``/``pdip``/``raw``            | Method for calculation of reference diamagnetic shifts                  | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:dft`` specifies DFT calculated shifts                       |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:raw`` specifies ``.csv`` data entry                         |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``diamagnetic_ref``         | ``file``                | File Name                           | File containing reference diamagnetic data - Format depends on          | |:x:|                |
    |                             |                         |                                     | ``method``                                                              |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:dft`` supports Gaussian ``.log`` or Orca ``.out``           |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:raw`` supports ``.csv`` file - see :ref:`here <dia_csv>`.   |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``susc_fit``                | ``type``                | ``split``/``isoaxrho``/``full``     | Form of susceptibility tensor to fit.                                   | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | ``isoeigen``/``eigen``              | See :ref:`Susceptibility Models <susc_models>`. for more information    |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``susc_fit``                | ``variables``           | List of                             | Specifies variables of model, whether each is fitted or fixed, and      | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | ``variable_name: [fit/fix, value]`` | the initial or fixed value of that variable in Å :sup:`3`               |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | e.g. ``iso [fit, 0.02]``            | See :ref:`Susceptibility Models <susc_models>`. for more information    |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``assignment``              | ``method``              | ``fixed``/``permute``               | Specifies how to carry out assignment of experimental signals to        | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | nuclei of molecule.                                                     |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``fixed`` uses assignment given in experiment file                   |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``permute`` permutes labels in assignment file according to          |                      |
    |                             |                         |                                     |    ``assignment: groups``                                               |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``assignment``              | ``groups``              | ``- [chemlabel1, chemlabel2, ...]`` | Specifies a group of atoms whose assignments will be permuted           | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | One group per line                  | using chemlabels - Only required if assignment method is ``permute``.   |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | with no repeated labels             |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See :ref:`Permutation <permutation>` for more information               |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+

Optional arguments
^^^^^^^^^^^^^^^^^^

Additionally, certain aspects of the program can be controlled on the command line, these are listed in Table 2.


.. table:: Table 2: Optional command line arguments to ``fit_susc``

    +------------------------------+--------------------------------------------------------------------------+
    | Optional Argument(s)         | Description                                                              |
    +==============================+==========================================================================+
    | ``-h``                       | Print help documentation                                                 |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--dry_run``                | Parses all files but exits prior to fitting                              |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--susc_units (string)``    | Controls units of susceptibility values in plots and output files        |
    |                              |                                                                          |
    |                              | Options:                                                                 |
    |                              |                                                                          |
    |                              | - ``A3`` :math:`\mathrm{Å}^3`                                            |
    |                              | - ``cm3 mol-1`` :math:`\mathrm{cm}^3 \ \mathrm{mol}^{-1}`                |
    |                              |                                                                          |
    |                              | Default ``A3``                                                           |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--contrib_plots (string)`` | Plots theoretical shift broken down into Fermi contact, pseudocontact,   |
    |                              | and diamagnetic contributions                                            |
    |                              |                                                                          |
    |                              | Options:                                                                 |
    |                              |                                                                          |
    |                              | - ``on`` shows and saves the plots                                       |
    |                              | - ``show`` shows the plots                                               |
    |                              | - ``save`` saves the plots                                               |
    |                              | - ``off`` neither shows nor saves                                        |
    |                              |                                                                          |
    |                              | Default ``off``                                                          |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--shift_plots (string)``   | Creates plots of experimental vs theoretical shift at each temperature   |
    |                              |                                                                          |
    |                              | Options:                                                                 |
    |                              |                                                                          |
    |                              | - ``on`` shows and saves the plots                                       |
    |                              | - ``show`` shows the plots                                               |
    |                              | - ``save`` saves the plots                                               |
    |                              | - ``off`` neither shows nor saves                                        |
    |                              |                                                                          |
    |                              | Default ``on``                                                           |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--spread_plots (string)``  | Plots theoretical shift broken down into Fermi contact, pseudocontact,   |
    |                              | and diamagnetic contributions as violin plots to illustrate spreads      |
    |                              |                                                                          |
    |                              | Options:                                                                 |
    |                              |                                                                          |
    |                              | - ``on`` shows and saves the plots                                       |
    |                              | - ``show`` shows the plots                                               |
    |                              | - ``save`` saves the plots                                               |
    |                              | - ``off`` neither shows nor saves                                        |
    |                              |                                                                          |
    |                              | Default ``on``                                                           |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--isoaxrho_plots (string)``| Creates plots isotropic, axial, and rhombic susceptibilities versus      |
    |                              | temperature                                                              |
    |                              |                                                                          |
    |                              | Options:                                                                 |
    |                              |                                                                          |
    |                              | - ``on`` shows and saves the plots                                       |
    |                              | - ``show`` shows the plots                                               |
    |                              | - ``save`` saves the plots                                               |
    |                              | - ``off`` neither shows nor saves                                        |
    |                              |                                                                          |
    |                              | Default ``on``                                                           |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--show_single``            | Show plots for first temperature and hide for the rest                   |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--pcs_isosurface``         | Saves PCS isosurface for each temperature to separate cube files         |
    +------------------------------+--------------------------------------------------------------------------+


Input files for HFC
-------------------

xyz file with molecular coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Gaussian output file
^^^^^^^^^^^^^^^^^^^^
DFT calulated hyperfine tensors are automatically printed in the Gaussian output in several units including MHz (which is being parsed). 
For molecules with less than 100 atoms Gaussian prints HFC tensors without a need for additional keywords, and for larger moleucles it should be requested with ``prop=epr`` keyword.

.. warning::
 Gaussian reports HFC tensor in internal coordinate frame by default, which may be different from the input coordinates

.. note::
 ``simpnmr`` will rotate the HFC tensors to match the frame of the magnetic susceptibility tensor if it is provided via ORCA file.

To prevent Gaussian from reporting HFC in the internal coordinates frame instead of the input frame, one could specify ``Symmetry=None`` keyword in Gaussian input file

.. warning::
 Gaussian reports dipolar HFC, whcih is NOT normalised per unpaired electron

.. note::
 ``simpnmr`` normalises dipolar HFC from Gaussian to be per unpaired electron.

ORCA output file
^^^^^^^^^^^^^^^^



.. _general_csv:

Parsing ``.csv`` files
----------------------

In general, any ``.csv`` file read by ``simpnmr`` follows the following set of rules

1. Column names are case-insensitive
2. Comment lines (``#``) are ignored (apart from metadata lines e.g. Experiment files)
3. No assumptions are made about delimiter
4. Additional columns are ignored


.. _exp_csv:

Experimental data ``.csv`` files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This file contains the positions, tentative assignment (chem_labels), and shape information for the paramagnetic signals
of a single NMR spectrum.

This file must be formatted as ``.csv``, and must contain the columns specified in Table 2, along
with all of the following metadata comment lines.

.. code-block ::

    # temperature VALUE
    # isotope VALUE
    # magnetic_field VALUE

where ``VALUE`` contains the temperature in Kelvin, the isotope measured in this experiment (e.g. ``1H``),
and the magnetic field in Tesla, respectively.

Table 2: Experimental data file ``.csv`` headers

+-------------------+----------------------+------------------------------------------------+
| Property          | Headers              | Headers                                        |
+===================+======================+================================================+
| Assignment        | assignment           | chem_label (tentative) assignment of signal    |
+-------------------+----------------------+------------------------------------------------+
| Shift             | shift (ppm)          | Experimental chemcial shift of signal          |
+-------------------+----------------------+------------------------------------------------+
| Area              | area                 | Area of signal                                 |
+-------------------+----------------------+------------------------------------------------+
| Linewidth         | width (Hz)           | Width of signal                                |
+-------------------+----------------------+------------------------------------------------+
| L/G               | L/G                  | Ratio of Lorentzian to Gaussian functions used |
|                   |                      | to model signal                                |
+-------------------+----------------------+------------------------------------------------+

.. _dia_csv:

Diamagnetic correction ``.csv`` files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This contains diamagnetic shifts for a given system.

+-------------------+----------------------+
| Property          | Headers              |
+===================+======================+
| Chemical label    | chem_label           |
+-------------------+----------------------+
| Shift             | shift (ppm)          |
+-------------------+----------------------+

It is assumed that diamagnetic corrections are independent of temperature.

.. _chemlabels_csv:

Chemical label files
^^^^^^^^^^^^^^^^^^^^

This file contains the additional labels which allow the user to refer to the atoms of a molecule using a set of
chemically meaningful labels e.g. ``Me1`` rather than the atomic labels ``H1``, ``H2``, ``H3``. Chemical labels
can be repeated and serve to group atoms together for plotting and averaging purposes.

Optionally, the user may also specify `mathtext <https://matplotlib.org/stable/users/explain/text/mathtext.html>`_ labels to be used when plotting data in ``matplotlib`` - these must
be wrapped in ``$$``.

This must be a ``.csv`` file containing a series of columns, each denoting a specific property (see below).

+-------------------+----------------------+------------------------------------------------------------------------------+--------------------------+
| Property          | Headers              | Description                                                                  | Mandatory                |
+===================+======================+==============================================================================+==========================+
| Atom label        | atom_label           | Labels of each NMR active nucleus including indexing number                  | |:white_check_mark:|     |
+-------------------+----------------------+------------------------------------------------------------------------------+--------------------------+
| Chemical label    | chem_label           | Chemical label of each NMR active nucleus                                    | |:white_check_mark:|     |
+-------------------+----------------------+------------------------------------------------------------------------------+--------------------------+
| Math label        | chem_math_label      | Mathtext label of each NMR active nucleus e.g. ``$\mathregular{Me}_1$``      | |:x:|                    |
+-------------------+----------------------+------------------------------------------------------------------------------+--------------------------+

At least of each property must be present in the file (in any order), and a comma delimited header line
containing the headers in Table 4 must be present above the data columns.

Users can convert annotated chemcraft ``.xyz`` files into a Chemical label file using ``simpnmr``'s included ``xyz_to_chemlabel`` script. Run ``xyz_to_chemlabel -h`` for more information.
