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


def energy_column(data):
    for column in data.columns:
        if "energy" in column.lower():
            return column


def test_relaxation_reduces_energy():
    for _, r in rows().iterrows():
        for name in ["before_attack_relaxation_data.csv", "after_attack_relaxation_data.csv"]:
            data = pd.read_csv(Path(r["actual_output_dir"]) / name)
            energy = pd.to_numeric(data[energy_column(data)], errors="coerce").dropna()
            assert energy.iloc[-1] <= energy.iloc[0] + 1e-8


def test_relaxation_reduces_after_attack_max_force():
    for _, r in rows().iterrows():
        before = np.linalg.norm(forces(r["perturbed_force_csv"]), axis=1).max()
        after = np.linalg.norm(forces(r["after_force_csv"]), axis=1).max()
        assert after <= before


def test_final_displacement_is_nonnegative():
    for _, r in rows().iterrows():
        d = np.linalg.norm(positions(r["after_force_csv"]) - positions(r["before_force_csv"]), axis=1)
        assert (d >= 0).all()


def test_final_force_meets_relaxation_threshold():
    for _, r in rows().iterrows():
        max_force = np.linalg.norm(forces(r["after_force_csv"]), axis=1).max()
        assert max_force <= float(r["relax_fmax"]) + 1e-8


def test_relaxation_reduces_force_after_attack():
    for _, r in rows().iterrows():
        before = np.linalg.norm(forces(r["perturbed_force_csv"]), axis=1).max()
        after = np.linalg.norm(forces(r["after_force_csv"]), axis=1).max()
        assert after <= before