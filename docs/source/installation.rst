.. _installation:

Installation
==============================

Setting up python
-----------------

If you are experienced with python and installing packages, go to the :ref:`next section <pip_install>`.

If you are new to python, we recommend using `Anaconda <https://www.anaconda.com/download>`_ - see the
`Anaconda guide <https://docs.anaconda.com/free/anaconda/install/>`_ for more information. Anaconda may be available
at your institution through your IT provider - **note we do not offer support for installing Anaconda/python**.

If you're using Anaconda, run the following command in your Anaconda enabled terminal with a `conda env` activated to install
the ``pip`` package manager.

.. code-block:: bash

    conda install pip

.. _pip_install:

Installation
------------

The ``simpnmr`` python package and its command line interface can be installed using the ``pip`` package manager

.. code-block:: bash

    pip install simpnmr

To test your installation was successful, run the following command

.. code-block:: bash

    simpnmr -h

You should see the help text for ``simpnmr``. You are now ready to start using ``SimpNMR``, head to the :ref:`usage` pages for more information, 
details.

.. _updating:

Updating
--------

To update ``SimpNMR``, run

.. code-block:: bash

    pip install simpnmr --upgrade
