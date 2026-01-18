import re

from ..text import find_first_group


def read_exp_metadata(file_name: str) -> tuple[float, float, str]:
    """Reads metadata from an experiment CSV file.

    Metadata is stored as single comment lines beginning with ``#`` and formatted as
    ``name value``. Supported keys are ``temperature``, ``magnetic_field``, and
    ``isotope``.

    Args:
        file_name: Path to the experiment file.`

    Returns:
        A tuple ``(temperature, magnetic_field, isotope)`` where temperature is in K,
        magnetic field is in T, and isotope is formatted like ``"1H"`` or ``"13C"``.

    Raises:
        IndexError: If a required metadata line is missing.
        ValueError: If a numeric metadata value cannot be parsed.
    """

    temperature, magnetic_field, isotope = None, None, None

    temperature = float(
        find_first_group(file_name, r"# *temperature (\d*\.*\d*)", re.IGNORECASE)
    )

    magnetic_field = float(
        find_first_group(file_name, r"# *magnetic_field (\d*\.*\d*)", re.IGNORECASE)
    )

    isotope = str(
        find_first_group(file_name, r"# *isotope (\d{0,3}[A-Za-z]{0,2})", re.IGNORECASE)
    )

    return temperature, magnetic_field, isotope
