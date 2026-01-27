import os

from simpnmr.app.setup.options import CalcPcsIsoRunOptions
from simpnmr.core.domain.tensors import Susceptibility
from simpnmr.io.qc import qc_readers as rdrs
from simpnmr.tools.coords_tools import xyz_format as xyzf


def run_calc_pcs_iso(
    *,
    susc_file: str,
    susc_format: str,
    temperatures: list[float],
    structure_file: str,
    central_atom: str,
    options: CalcPcsIsoRunOptions,
) -> int:
    """Run PCS isosurface calculation using an isotropic susceptibility tensor.

    This pipeline is intentionally config-free (no YAML). It is designed to be
    callable both from the CLI and from Python code.

    Args:
        susc_file: Path to the susceptibility tensor file.
        susc_format: Susceptibility file format identifier (e.g. "orca_*", "csv_*").
        temperatures: Temperatures (K) to compute PCS isosurfaces for.
        structure_file: Path to the structure file (.xyz, .log, .out).
        central_atom: Indexed label of the paramagnetic centre (e.g. "Ni1").
        options: Runtime-related options.

    Returns:
        Exit code.
    """

    # Load structure
    ext = os.path.splitext(structure_file)[1]
    if ext == ".xyz":
        labels, coords = xyzf.load_xyz(structure_file)
    elif ext in {".log", ".out"}:
        qcs = rdrs.QCStructure.guess_from_file(structure_file)
        labels = qcs.labels
        coords = qcs.coords
    else:
        raise ValueError(f"Unsupported structure file format: {ext}")

    if central_atom not in labels:
        raise ValueError(
            "Specified central atom not present in structure file. "
            "Try indexed labels, e.g. Ni1."
        )

    # Load susceptibility tensors
    if "orca" in susc_format:
        suscs = Susceptibility.from_orca(
            susc_file,
            section=susc_format.split("orca_")[1],
        )
    elif "csv" in susc_format:
        suscs = Susceptibility.from_csv(susc_file)
    elif "molcas" in susc_format:
        raise ValueError("Molcas files are not currently supported")
    else:
        raise ValueError(f"Unknown susceptibility format: {susc_format}")

    matched = [
        s
        for s in suscs
        if round(s.temperature, 2) in {round(t, 2) for t in temperatures}
    ]

    if not matched:
        raise ValueError(
            f"No susceptibility tensors matched temperatures {temperatures}. "
            f"Available: {[s.temperature for s in suscs]}"
        )

    for s in matched:
        s.calc_irred()
        s.save_pcs_isosurface(
            labels,
            coords,
            central_atom,
            comment=f"PCS Isosurface from {susc_file} at {s.temperature:.2f} K",
            file_name=f"pcs_isosurface_{s.temperature:.2f}_K.cube",
        )

    return 0
