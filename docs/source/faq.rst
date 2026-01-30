.. _faq:

FAQ
---

This page contains some frequently asked questions. Please consult this page before raising a GitLab issue!

Errors:
^^^^^^^^^^

1. ``'simpnmr' is not recognized as an internal or external command, operable program or batch file.``

    You are trying to run ``simpnmr`` in a terminal where Python (and SimpNMR) is not available.
    First check that Python is installed and works in this terminal (for example, by running ``python --version``),
    then reinstall or activate SimpNMR in that environment before running ``simpnmr`` again.

2. ``ModuleNotFoundError: No module named 'simpnmr'.``

    Python cannot find the SimpNMR package in the environment you are using.
    Activate the environment where you want to use SimpNMR (for example with ``conda activate ...`` or ``source venv/bin/activate``)
    and (re)install it:
    
    .. code-block:: bash

        pip install simpnmr

3. ``no matches found: *yml``

    Your shell does not see any files matching ``*yml`` in the current working directory.
    Check that you are in the folder that contains your YAML configuration file(s) before running the command.

4. ``Missing file error: [Errno 2] No such file or directory``

    SimpNMR cannot find one or more of the files you asked it to use.
    Double-check the paths to all required input files (see ``simpnmr -h`` for a summary of the expected inputs and options).

5. ``yaml.scanner.ScannerError`` or ``yaml.parser.ParserError``

    SimpNMR could not read your YAML file because it is not valid YAML.
    Check the indentation, colons, and quotation marks in your configuration file
    (a YAML linter or editor with YAML support can also help you spot the issue).

6. ``PermissionError: [Errno 13] Permission denied``

    SimpNMR does not have permission to read or write one of the files or folders you selected.
    Make sure you have write access to the output directory and read access to all input files,
    or choose a different location for the simulation outputs.

