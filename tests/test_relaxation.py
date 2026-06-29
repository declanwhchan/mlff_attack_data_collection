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


def energy_column(data):
    for column in data.columns:
        if "energy" in column.lower():
            return column


def test_relaxation_energies_are_finite():
    for _, r in rows().iterrows():
        names = [
            "before_attack_relaxation_data.csv",
            "after_attack_relaxation_data.csv",
        ]

        for name in names:
            data = pd.read_csv(Path(r["actual_output_dir"]) / name)
            energy = pd.to_numeric(
                data[energy_column(data)],
                errors="coerce",
            ).dropna()

            assert not energy.empty
            assert np.isfinite(energy).all()


def test_relaxation_forces_are_finite():
    for _, r in rows().iterrows():
        perturbed = forces(r["perturbed_force_csv"])
        final = forces(r["after_force_csv"])

        assert perturbed.shape == final.shape
        assert np.isfinite(perturbed).all()
        assert np.isfinite(final).all()
