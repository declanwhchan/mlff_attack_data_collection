from pathlib import Path
import os

import pytest
import numpy as np
import pandas as pd


def rows():
    root = Path(
        f"/scratch/{os.environ['USER']}/mlff_attack_data_collection"
    )

    paths = list(
        root.glob("trial*_seed*/array_summaries/*_summary.csv")
    )

    if not paths:
        pytest.skip("No calculation summaries found")

    data = pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )

    return data.query("status == 'success'").iloc[::20]


def forces(path):
    return pd.read_csv(path)[["fx", "fy", "fz"]].to_numpy()


def positions(path):
    return pd.read_csv(path)[["x", "y", "z"]].to_numpy()


def test_perturbation_moves_atoms():
    for _, r in rows().iterrows():
        d = np.linalg.norm(positions(r["perturbed_force_csv"]) - positions(r["before_force_csv"]), axis=1)
        assert d.max() > 0


def test_perturbation_changes_forces():
    for _, r in rows().iterrows():
        df = np.linalg.norm(forces(r["perturbed_force_csv"]) - forces(r["before_force_csv"]), axis=1)
        assert df.max() > 0


def test_larger_epsilon_tends_to_give_larger_displacement():
    data = rows().dropna(subset=["epsilon", "mean_displacement"])
    grouped = data.groupby("epsilon")["mean_displacement"].median().sort_index()
    assert grouped.iloc[-1] >= grouped.iloc[0]


def test_stronger_perturbation_tends_to_change_forces_more():
    changes_from_perturbations = []
    for _, r in rows().iterrows():
        changes_from_perturbations.append((float(r["epsilon"]), np.linalg.norm(forces(r["perturbed_force_csv"]) - forces(r["before_force_csv"]), axis=1).mean()))

    grouped = pd.DataFrame(
        data=changes_from_perturbations,
        columns=["epsilon", "delta_force"],
    ).groupby("epsilon")["delta_force"].median().sort_index()

    assert grouped.iloc[-1] >= grouped.iloc[0]