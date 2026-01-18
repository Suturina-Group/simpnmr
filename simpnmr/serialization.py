import pandas as pd

from . import main


def build_experiment_signals_df(experiment: "main.Experiment") -> pd.DataFrame:
    columns = ["assignment ()", "shift (ppm)", "width (Hz)", "area ()", "L/G ()"]

    data = {
        "assignment ()": [s.assignment for s in experiment.signals],
        "shift (ppm)": [s.shift for s in experiment.signals],
        "width (Hz)": [s.width for s in experiment.signals],
        "area ()": [s.area for s in experiment.signals],
        "L/G ()": [s.l_to_g for s in experiment.signals],
    }

    df = pd.DataFrame(data, columns=columns)

    if df.empty:
        return df

    return df.sort_values("shift (ppm)").reset_index(drop=True)


def build_molecule_df(molecule):
    """Build a full molecule table for CSV export."""

    columns = [
        "atom_label ()",
        "chem_label ()",
        "x (Å)",
        "y (Å)",
        "z (Å)",
        "Aiso (ppm Å^-3)",
        "Adip_xx (ppm Å^-3)",
        "Adip_xy (ppm Å^-3)",
        "Adip_xz (ppm Å^-3)",
        "Adip_yy (ppm Å^-3)",
        "Adip_yz (ppm Å^-3)",
        "Adip_zz (ppm Å^-3)",
        "δ_total_avg (ppm)",
        "δ_total (ppm)",
        "δ_dia (ppm)",
        "δ_fc (ppm)",
        "δ_pc (ppm)",
        "linewidth (Hz)",
    ]

    nuclei = molecule.nuclei

    data = {
        "atom_label ()": [nuc.label for nuc in nuclei],
        "chem_label ()": [nuc.chem_label for nuc in nuclei],
        "x (Å)": [nuc.coord[0] for nuc in nuclei],
        "y (Å)": [nuc.coord[1] for nuc in nuclei],
        "z (Å)": [nuc.coord[2] for nuc in nuclei],
        "Aiso (ppm Å^-3)": [nuc.A.iso for nuc in nuclei],
        "Adip_xx (ppm Å^-3)": [nuc.A.dip[0, 0] for nuc in nuclei],
        "Adip_xy (ppm Å^-3)": [nuc.A.dip[0, 1] for nuc in nuclei],
        "Adip_xz (ppm Å^-3)": [nuc.A.dip[0, 2] for nuc in nuclei],
        "Adip_yy (ppm Å^-3)": [nuc.A.dip[1, 1] for nuc in nuclei],
        "Adip_yz (ppm Å^-3)": [nuc.A.dip[1, 2] for nuc in nuclei],
        "Adip_zz (ppm Å^-3)": [nuc.A.dip[2, 2] for nuc in nuclei],
        "δ_total_avg (ppm)": [nuc.shift.avg for nuc in nuclei],
        "δ_total (ppm)": [nuc.shift.total for nuc in nuclei],
        "δ_dia (ppm)": [nuc.shift.dia for nuc in nuclei],
        "δ_fc (ppm)": [nuc.shift.fc for nuc in nuclei],
        "δ_pc (ppm)": [nuc.shift.pc for nuc in nuclei],
        "linewidth (Hz)": [1 for _ in nuclei],
    }

    df = pd.DataFrame(data, columns=columns)

    if df.empty:
        return df

    return df.reset_index(drop=True)


def build_hyperfines_df(molecule):
    """Build a hyperfine-couplings table for CSV export."""

    columns = [
        "atom_label ()",
        "chem_label ()",
        "Aiso (ppm Å^-3)",
        "Adip_xx (ppm Å^-3)",
        "Adip_xy (ppm Å^-3)",
        "Adip_xz (ppm Å^-3)",
        "Adip_yy (ppm Å^-3)",
        "Adip_yz (ppm Å^-3)",
        "Adip_zz (ppm Å^-3)",
    ]

    nuclei = molecule.nuclei

    data = {
        "atom_label ()": [nuc.label for nuc in nuclei],
        "chem_label ()": [nuc.chem_label for nuc in nuclei],
        "Aiso (ppm Å^-3)": [nuc.A.iso for nuc in nuclei],
        "Adip_xx (ppm Å^-3)": [nuc.A.dip[0, 0] for nuc in nuclei],
        "Adip_xy (ppm Å^-3)": [nuc.A.dip[0, 1] for nuc in nuclei],
        "Adip_xz (ppm Å^-3)": [nuc.A.dip[0, 2] for nuc in nuclei],
        "Adip_yy (ppm Å^-3)": [nuc.A.dip[1, 1] for nuc in nuclei],
        "Adip_yz (ppm Å^-3)": [nuc.A.dip[1, 2] for nuc in nuclei],
        "Adip_zz (ppm Å^-3)": [nuc.A.dip[2, 2] for nuc in nuclei],
    }

    df = pd.DataFrame(data, columns=columns)

    if df.empty:
        return df

    return df.reset_index(drop=True)
