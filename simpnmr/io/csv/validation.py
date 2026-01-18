def can_float(s: str) -> bool:
    """Returns whether a string can be parsed as a float."""
    out = True
    try:
        s = float(s.strip())
    except ValueError:
        out = False

    return out
