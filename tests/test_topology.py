from pathlib import Path
import sys

import numpy as np
from ase import Atoms


scripts_dir = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(scripts_dir))

import run_tests as topology

def carbon_pair(distance):
    return Atoms(
        "C2",
        positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )


def test_identical_structures_have_zero_topology_change():
    atoms = carbon_pair(distance=1.2)

    metrics = topology.topology_change_metrics(
        atoms,
        atoms.copy(),
    )

    assert metrics["neighbor_jaccard_distance"] == 0.0
    assert metrics["coordination_change_mean"] == 0.0
    assert metrics["coordination_change_max"] == 0.0
    assert metrics["rdf_l1_distance"] == 0.0


def test_removing_the_only_bond_has_jaccard_distance_one():
    bonded = carbon_pair(distance=1.2)
    separated = carbon_pair(distance=3.0)

    metrics = topology.topology_change_metrics(
        bonded,
        separated,
    )

    assert metrics["neighbor_edges_before"] == 1
    assert metrics["neighbor_edges_after"] == 0
    assert metrics["neighbor_edges_removed"] == 1
    assert metrics["neighbor_jaccard_distance"] == 1.0
    assert metrics["coordination_change_max"] == 1.0


def test_standard_rdf_is_finite_and_nonnegative():
    atoms = Atoms(
        "C4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, 0.0, 1.2],
        ],
        cell=[15.0, 15.0, 15.0],
        pbc=True,
    )

    rdf, radii = topology.rdf_values(atoms)

    assert len(rdf) == 60
    assert len(radii) == 60
    assert np.all(np.isfinite(rdf))
    assert np.all(rdf >= 0.0)
    assert np.all(np.diff(radii) > 0.0)


def test_moving_an_atom_changes_standard_rdf():
    before = Atoms(
        "C4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, 0.0, 1.2],
        ],
        cell=[15.0, 15.0, 15.0],
        pbc=True,
    )
    after = before.copy()
    after.positions[1, 0] = 1.6

    distance = topology.rdf_l1_distance(before, after)

    assert np.isfinite(distance)
    assert distance > 0.0


def test_topology_summary_records_removed_bond():
    summary = topology.topology_change_metrics(
        carbon_pair(distance=1.2),
        carbon_pair(distance=3.0),
    )

    assert summary["neighbor_edges_before"] == 1
    assert summary["neighbor_edges_after"] == 0
    assert summary["neighbor_edges_added"] == 0
    assert summary["neighbor_edges_removed"] == 1
    assert summary["neighbor_edge_change_count"] == 1
