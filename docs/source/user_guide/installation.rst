.. _installation:

Installation
============

.. admonition:: Windows users — the easy way
   :class: tip

   If you are on Windows and just want the graphical application, you do **not**
   need Python or the command line. Download the installer, double-click it, and
   launch SimpNMR from the Start menu like any other program:

   * `Download the latest Windows installer <https://gitlab.com/suturina-group/simpnmr/-/releases>`_
     (``SimpNMR-Setup-<version>.exe`` under the newest release).

   The rest of this page describes the command-line installation, which works on
   Windows, macOS and Linux and is required for scripted or headless use.

``simpnmr`` is a Python program that you run from a terminal (also called a
"command line" or "command prompt"). If you have never used a terminal before,
don't worry — this page walks through every step, starting from a computer with
nothing installed.

There are three steps:

1. :ref:`Install Python <install-python>` (once per computer).
2. :ref:`Open a terminal <open-terminal>`.
3. :ref:`Install simpnmr <install-simpnmr>` with a single command.

If you already have Python 3.10 or newer and know how to use a terminal, you can
skip straight to the short version:

.. code-block:: bash

    pip install simpnmr

.. _install-python:

Step 1 — Install Python
-----------------------

``simpnmr`` needs **Python 3.10 or newer**. You only have to do this once.

.. tab-set::

    .. tab-item:: Windows
        :sync: windows

        1. Go to the official Python website:
           `python.org/downloads <https://www.python.org/downloads/>`_.
        2. Click the big yellow **Download Python** button. This downloads an
           installer file (for example ``python-3.12.x-amd64.exe``).
        3. Open the downloaded file to start the installer.
        4. **Very important:** on the first screen, tick the checkbox at the
           bottom that says **"Add python.exe to PATH"** *before* clicking
           anything else.

           .. warning::

              If you skip the **"Add python.exe to PATH"** checkbox, Windows will
              not be able to find Python later and you will see errors like
              ``'python' is not recognized``. This is the single most common
              installation problem on Windows.

        5. Click **Install Now** and wait for it to finish. When it says
           "Setup was successful", click **Close**.

    .. tab-item:: macOS
        :sync: macos

        macOS often ships with an old Python. Install a current version:

        1. Go to `python.org/downloads <https://www.python.org/downloads/>`_.
        2. Click **Download Python** — it will offer the correct macOS
           installer (a ``.pkg`` file).
        3. Open the downloaded ``.pkg`` file and follow the prompts
           (click **Continue** / **Install**), entering your password when
           asked.

        .. tip::

           If you use `Homebrew <https://brew.sh/>`_, you can instead run
           ``brew install python`` in the Terminal.

    .. tab-item:: Linux

        Most Linux distributions already include a suitable Python 3. Check
        with ``python3 --version``. If it is older than 3.10, install a newer
        version with your package manager, for example on Debian/Ubuntu:

        .. code-block:: bash

            sudo apt update
            sudo apt install python3 python3-pip python3-venv

.. _open-terminal:

Step 2 — Open a terminal
------------------------

A terminal is a window where you type commands. Here is how to open one.

.. tab-set::

    .. tab-item:: Windows
        :sync: windows

        1. Click the **Start** menu (or press the Windows key).
        2. Type ``PowerShell``.
        3. Click **Windows PowerShell** in the results.

        A dark blue or black window opens with a blinking cursor. This is where
        you type the commands in the next step.

    .. tab-item:: macOS
        :sync: macos

        1. Press ``Cmd`` + ``Space`` to open Spotlight search.
        2. Type ``Terminal`` and press ``Return``.

        A window opens with a blinking cursor. This is where you type the
        commands in the next step.

    .. tab-item:: Linux

        Open your **Terminal** application from the applications menu, or press
        ``Ctrl`` + ``Alt`` + ``T`` on many distributions.

.. _install-simpnmr:

Step 3 — Install simpnmr
------------------------

In the terminal you just opened, type the following command and press
``Enter``:

.. tab-set::

    .. tab-item:: Windows
        :sync: windows

        .. code-block:: powershell

            py -m pip install simpnmr

    .. tab-item:: macOS / Linux
        :sync: macos

        .. code-block:: bash

            python3 -m pip install simpnmr

``pip`` is Python's package installer. It downloads ``simpnmr`` and everything
it depends on from the internet, so this step needs an active connection and may
take a minute or two. You will see a stream of "Collecting…" and "Installing…"
messages, ending with ``Successfully installed simpnmr-…``.

Step 4 — Check that it worked
-----------------------------

Confirm the installation by asking ``simpnmr`` to print its version:

.. code-block:: bash

    simpnmr --version

You should see a version number (for example ``simpnmr 2.0.0``). You can also
run:

.. code-block:: bash

    simpnmr -h

to print the list of available commands and options.

If both commands print output instead of an error, the installation is
complete. **You are ready to use SimpNMR** — head to the
:doc:`../tutorials/dy_point_dipole` tutorial for your first run.

.. _install-troubleshooting:

Troubleshooting
---------------

.. dropdown:: "python" or "py" is not recognized (Windows)

    Windows cannot find Python. This almost always means the **"Add python.exe
    to PATH"** checkbox was not ticked during installation
    (:ref:`Step 1 <install-python>`).

    The simplest fix is to reinstall Python: run the installer again, and this
    time make sure to tick **"Add python.exe to PATH"** on the first screen
    before clicking *Install*. You can also choose **Modify → Repair** if the
    installer offers it. **Close and reopen** the terminal afterwards so it
    picks up the change.

.. dropdown:: "simpnmr" is not recognized / command not found

    The package installed, but the terminal cannot find the ``simpnmr``
    command. First, **close and reopen the terminal** — the command becomes
    available only in terminals opened *after* installation.

    The quickest workaround is to run ``simpnmr`` through Python directly, which
    always works even when the command itself is not found:

    .. tab-set::

        .. tab-item:: Windows
            :sync: windows

            .. code-block:: powershell

                py -m simpnmr --version

        .. tab-item:: macOS / Linux
            :sync: macos

            .. code-block:: bash

                python3 -m simpnmr --version

    **Permanent fix on Windows — add the Scripts folder to PATH.** When ``pip``
    installs ``simpnmr``, it places the ``simpnmr.exe`` command in a folder
    called ``Scripts``. If that folder is not on your PATH, Windows cannot find
    the command even though the install succeeded. In fact ``pip`` usually warns
    you about this during installation, with a message like::

        WARNING: The script simpnmr.exe is installed in
        'C:\Users\<you>\AppData\...\Scripts' which is not on PATH.

    That quoted path is exactly the folder you need to add. If you did not see
    the warning, find the folder by running:

    .. code-block:: powershell

        py -c "import sysconfig; print(sysconfig.get_path('scripts'))"

    Then add it to your PATH permanently:

    1. Copy the ``Scripts`` folder path from the command above.
    2. Press the **Start** menu and type ``environment variables``, then click
       **Edit the system environment variables**.
    3. In the window that opens, click the **Environment Variables…** button.
    4. Under **User variables for <your name>**, select the variable named
       **Path** and click **Edit…**.
    5. Click **New**, paste the ``Scripts`` folder path, and click **OK** on
       each window to close them.
    6. **Close and reopen** the terminal, then run ``simpnmr --version`` again.

    .. tip::

       You can avoid this problem entirely by using a
       :ref:`virtual environment <virtual-environments>`. Activating one puts
       its ``Scripts`` folder on PATH automatically, so ``simpnmr`` is always
       found while the environment is active.

.. dropdown:: "pip is not recognized" or "No module named pip"

    Use the longer form that goes through Python, which always works:

    .. tab-set::

        .. tab-item:: Windows
            :sync: windows

            .. code-block:: powershell

                py -m pip install simpnmr

        .. tab-item:: macOS / Linux
            :sync: macos

            .. code-block:: bash

                python3 -m pip install simpnmr

.. dropdown:: "Permission denied" or a request to use "sudo"

    Do **not** install with ``sudo``. Instead, install ``simpnmr`` just for
    your user account by adding ``--user``:

    .. code-block:: bash

        python3 -m pip install --user simpnmr

    A :ref:`virtual environment <virtual-environments>` (below) avoids this
    problem entirely and is the recommended approach.

.. dropdown:: I have several versions of Python

    If you have more than one Python installed, make sure ``pip`` and
    ``simpnmr`` use the same one. The reliable way is to always call pip
    *through* the Python you intend to use:

    .. code-block:: bash

        python3 -m pip install simpnmr    # macOS / Linux
        py -m pip install simpnmr         # Windows

.. _virtual-environments:

Optional — using a virtual environment
--------------------------------------

A *virtual environment* is a private, self-contained Python setup for a single
project. It keeps ``simpnmr`` and its dependencies separate from the rest of
your system, so different projects can't interfere with one another. This is
optional and **not required** to use ``simpnmr``, but it is good practice.

Create and activate one, then install ``simpnmr`` inside it:

.. tab-set::

    .. tab-item:: Windows
        :sync: windows

        .. code-block:: powershell

            py -m venv simpnmr-env
            simpnmr-env\Scripts\activate
            pip install simpnmr

    .. tab-item:: macOS / Linux
        :sync: macos

        .. code-block:: bash

            python3 -m venv simpnmr-env
            source simpnmr-env/bin/activate
            pip install simpnmr

While the environment is active, its name appears in front of your terminal
prompt (``(simpnmr-env)``). Run ``deactivate`` to leave it. The next time you
want to use ``simpnmr``, reopen the terminal and run the activate command again.

Users who prefer ``conda`` can create an environment the same way
(``conda create -n simpnmr-env python=3.12`` followed by
``conda activate simpnmr-env``) and then ``pip install simpnmr`` inside it.

.. _macos-automator-app:

Optional — a double-click app on macOS (Automator)
--------------------------------------------------

``simpnmr`` ships a graphical interface. On macOS you can wrap it in a small
application so you can launch it from the Dock or Launchpad by double-clicking,
instead of typing a command each time. This uses **Automator**, which is built
into macOS — nothing extra to install.

**1. Install the GUI.** The graphical interface needs some extra packages
(Qt). Install them with the ``gui`` option:

.. code-block:: bash

    python3 -m pip install "simpnmr[gui]"

Check it launches from the terminal first:

.. code-block:: bash

    simpnmr-gui

**2. Find the launcher's full path.** Automator runs commands with a minimal
environment that does not know where ``simpnmr-gui`` lives, so you must give it
the complete path. Print it with:

.. code-block:: bash

    which simpnmr-gui

Copy the line it prints — for example
``/Users/you/Library/Python/3.12/bin/simpnmr-gui`` (or a path inside your
virtual environment). You will paste this in step 5.

.. note::

   If ``which simpnmr-gui`` prints nothing, the launcher is not on your PATH.
   Re-run the install command in step 1, and see
   :ref:`the troubleshooting section <install-troubleshooting>` above.

**3. Create an Automator application.**

1. Open **Automator** (press ``Cmd`` + ``Space``, type ``Automator``, press
   ``Return``).
2. Choose **New Document**, select **Application**, and click **Choose**.
3. In the search box on the left, type ``Run Shell Script``. Drag the
   **Run Shell Script** action into the empty workflow area on the right.
4. Leave **Shell** set to ``/bin/zsh``.
5. Replace the default text in the box with the full path from step 2, for
   example:

   .. code-block:: bash

       /Users/you/Library/Python/3.12/bin/simpnmr-gui

**4. Save it as an app.** Choose **File → Save**, name it ``SimpNMR``, set
**Where** to **Applications**, and save. ``SimpNMR`` now appears in your
Applications folder and Launchpad; double-click it to start the interface.

.. tip::

   **Give it an icon.** In Finder, select your ``SimpNMR`` app and press
   ``Cmd`` + ``I`` to open *Get Info*. Drag an image file (``.png`` or
   ``.icns``) onto the small icon in the top-left corner of that window.

   **Keep it working after updates.** The app just runs ``simpnmr-gui``, so it
   automatically picks up new versions when you upgrade with ``pip``. The only
   time you need to recreate it is if the launcher's path changes (for example,
   if you move to a different virtual environment).

.. _updating:

Updating
--------

To update ``simpnmr`` to the latest release, run the install command again with
``--upgrade``:

.. code-block:: bash

    pip install simpnmr --upgrade

Check the installed version at any time with ``simpnmr --version``.
