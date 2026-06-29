from pathlib import Path
import sys

import numpy as np
import pandas as pd
from ase import Atoms


scripts_dir = Path(__file__).resolve().parents[1] / "scripts_python"
sys.path.insert(0, str(scripts_dir))

import run_tests as topology

def carbon_pair(distance):
    return Atoms(
        "C2",
        positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )


def test_identical_structures_have_zero_topology_change(tmp_path):
    atoms = carbon_pair(distance=1.2)

    metrics = topology.topology_change_metrics(atoms, atoms.copy(), tmp_path)

    assert metrics["neighbor_jaccard_distance"] == 0.0
    assert metrics["coordination_change_mean"] == 0.0
    assert metrics["coordination_change_max"] == 0.0
    assert metrics["rdf_l1_distance"] == 0.0


def test_removing_the_only_bond_has_jaccard_distance_one(tmp_path):
    bonded = carbon_pair(distance=1.2)
    separated = carbon_pair(distance=3.0)

    metrics = topology.topology_change_metrics(bonded, separated, tmp_path)

    assert metrics["neighbor_edges_before"] == 1
    assert metrics["neighbor_edges_after"] == 0
    assert metrics["neighbor_edges_removed"] == 1
    assert metrics["neighbor_jaccard_distance"] == 1.0
    assert metrics["coordination_change_max"] == 1.0


def test_rdf_histogram_is_normalized():
    atoms = Atoms(
        "C3",
        positions=[[0, 0, 0], [1.2, 0, 0], [0, 1.2, 0]],
        cell=[10, 10, 10],
        pbc=False,
    )

    histogram = topology.rdf_histogram(atoms)

    assert len(histogram) == 60
    assert np.isclose(histogram.sum(), 1.0)


def test_edge_change_csv_lists_removed_bond(tmp_path):
    topology.topology_change_metrics(
        carbon_pair(distance=1.2),
        carbon_pair(distance=3.0),
        tmp_path,
    )

    changes = pd.read_csv(tmp_path / "topology_edge_changes.csv")

    assert changes.to_dict("records") == [
        {"change": "removed", "edge": "C0-C1"}
    ]
