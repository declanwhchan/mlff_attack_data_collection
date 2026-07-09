#!/usr/bin/env python3
from pathlib import Path
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import jaccard as scipy_jaccard

from mlff_attack.attacks import make_attack
from mlff_attack.relaxation import load_structure, run_relaxation, setup_calculator
from mlff_attack.visualization import (
    extract_trajectory_data,
    load_trajectory,
)
from ase.neighborlist import neighbor_list, natural_cutoffs

import spglib

try:
    from ase.geometry.rdf import get_rdf
except ImportError:
    from ase.geometry.analysis import get_rdf


BASE_DIR = Path(__file__).resolve().parent.parent
TEST_FILE = BASE_DIR / "tests_sample.csv"
outputs_mace_DIR = BASE_DIR / "outputs_mace"
outputs_uma_DIR = BASE_DIR / "outputs_uma"
outputs_chgnet_DIR = BASE_DIR / "outputs_chgnet"


def active_environment():
    exe = str(sys.executable).lower()
    if ".venv-mace" in exe:
        return "mace"
    elif ".venv-uma" in exe:
        return "uma"
    elif ".venv-chgnet" in exe:
        return "chgnet"
    return None


def dtype_for_row(row):
    value = summary_text(row, "dtype_str")
    if value is None:
        value = os.environ.get("MLFF_DTYPE", "float64")
    value = str(value).strip().lower()
    if value not in {"float32", "float64"}:
        raise RuntimeError(f"dtype_str must be float32 or float64, got {value!r}")
    return value


def infer_calculator(model_path):
    model_name = Path(str(model_path)).name.lower()

    if model_name.startswith("mace"):
        return "mace"

    elif model_name.startswith("uma"):
        return "uma"

    elif model_name.startswith("chgnet"):
        return "chgnet"

    raise RuntimeError(
        "Could not infer calculator. model_path basename must start with 'mace', 'uma', or 'chgnet'."
    )


def output_root():
    root = os.environ.get("MLFF_OUTPUT_ROOT")
    if root:
        return BASE_DIR / root
    return BASE_DIR


def scratch_output():
    return output_root().resolve().is_relative_to(Path("/scratch"))


def run_relaxation_and_save_data(
    atoms,
    traj_path,
    output_dir,
    fmax,
    max_steps,
    optimizer,
):
    success = run_relaxation(
        atoms,
        traj_path,
        fmax=fmax,
        max_steps=max_steps,
        optimizer=optimizer,
        verbose=True,
    )

    if not success:
        raise RuntimeError(f"Relaxation failed for {traj_path.name}")

    trajectory = load_trajectory(traj_path)
    if trajectory is None:
        raise RuntimeError(f"Could not load {traj_path}")

    steps, energies, max_forces, volumes = (
        extract_trajectory_data(trajectory)
    )

    data_path = output_dir / f"{traj_path.stem}_data.csv"

    pd.DataFrame({
        "Step": steps,
        "Energy (eV)": energies,
        "Max Force (eV/Å)": max_forces,
        "Volume (Å³)": volumes,
    }).to_csv(data_path, index=False)

    return data_path


def run_seed_for(row):
    if "seed" in row:
        seed = as_int_or_none(row["seed"])
        if seed is not None:
            return seed
    return int(os.environ.get("MLFF_SEED", "42"))


def output_base_for(calculator, dtype_str):
    calculator = str(calculator).strip().lower()

    if calculator == "mace":
        return output_root() / f"outputs_{dtype_str}" / "mace"

    elif calculator == "uma":
        return output_root() / f"outputs_{dtype_str}" / "uma"

    elif calculator == "chgnet":
        return output_root() / f"outputs_{dtype_str}" / "chgnet"

    raise RuntimeError(f"Unknown calculator for output folder: {calculator}")


def as_none(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return value


def as_bool(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    value = str(value).strip().lower()
    return value in ["true", "1", "yes", "y"]


def as_float_or_none(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def as_int_or_none(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return int(value)


def summary_text(row, column):
    if column not in row:
        return None
    value = as_none(row[column])
    if value is None:
        return None
    return value


def value_or_default(value, default):
    if pd.isna(value) or str(value).strip() == "":
        return default
    return value


def attack_name(row):
    attack_type = str(row["attack_type"]).strip().lower()
    n_steps = int(row["n_steps"])

    if attack_type == "fgsm" and n_steps > 1:
        return "I-FGSM"

    return attack_type.upper()


def attack_parameters(row):
    parts = [
        f"epsilon={float(row['epsilon']):g}",
        f"steps={int(row['n_steps'])}",
    ]

    alpha = as_float_or_none(row["alpha"])
    if alpha is not None:
        parts.append(f"alpha={alpha:g}")

    return ", ".join(parts)


def save_force_data(atoms, output_dir, label):
    forces = atoms.get_forces()
    positions = atoms.get_positions()
    force_magnitudes = np.linalg.norm(forces, axis=1)

    force_df = pd.DataFrame({
        "atom_index": range(len(atoms)),
        "symbol": atoms.get_chemical_symbols(),
        "x": positions[:, 0],
        "y": positions[:, 1],
        "z": positions[:, 2],
        "fx": forces[:, 0],
        "fy": forces[:, 1],
        "fz": forces[:, 2],
        "force_magnitude": force_magnitudes,
    })

    force_csv = output_dir / f"{label}_forces.csv"
    force_df.to_csv(force_csv, index=False)

    return force_csv


def atom_signature(symbols, index):
    return f"{symbols[index]}{index}"


def neighbor_edge_set(atoms):
    symbols = atoms.get_chemical_symbols()
    cutoffs = natural_cutoffs(atoms)

    try:
        i_list, j_list = neighbor_list("ij", atoms, cutoffs)
    except Exception:
        i_list, j_list = [], []

    edges = set()
    for i, j in zip(i_list, j_list):
        i = int(i)
        j = int(j)
        if i == j:
            continue

        a, b = sorted([i, j])
        edges.add((atom_signature(symbols, a), atom_signature(symbols, b)))

    return edges


def edge_jaccard_distance(before_edges, after_edges):
    """Return SciPy Jaccard dissimilarity for two neighbor-edge sets."""
    edge_universe = sorted(before_edges | after_edges)

    if not edge_universe:
        return 0.0

    before_vector = np.fromiter(
        (edge in before_edges for edge in edge_universe),
        dtype=bool,
    )
    after_vector = np.fromiter(
        (edge in after_edges for edge in edge_universe),
        dtype=bool,
    )

    return float(scipy_jaccard(before_vector, after_vector))


def coordination_by_atom(edges, atoms):
    symbols = atoms.get_chemical_symbols()
    counts = {atom_signature(symbols, index): 0 for index in range(len(atoms))}

    for a, b in edges:
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1

    return counts


RDF_METHOD = "ase_total_rdf_l1_v1"


def rdf_analysis_cell(atoms, r_max=6.0):
    """Repeat periodic cells until ASE can safely calculate RDF to r_max."""
    expanded = atoms.copy()

    if not np.any(expanded.pbc):
        return expanded

    cell = np.asarray(expanded.cell, dtype=float)
    volume = abs(float(np.linalg.det(cell)))

    if volume <= 1e-12:
        raise ValueError("RDF requires a cell with nonzero volume")

    repeats = [1, 1, 1]

    for axis in range(3):
        if not expanded.pbc[axis]:
            continue

        other_axes = [index for index in range(3) if index != axis]
        face_area = np.linalg.norm(
            np.cross(cell[other_axes[0]], cell[other_axes[1]])
        )

        if face_area <= 1e-12:
            raise ValueError("RDF requires valid periodic cell vectors")

        cell_height = volume / face_area
        repeats[axis] = max(
            1,
            int(np.ceil((2.0 * r_max + 1e-8) / cell_height)),
        )

    return expanded.repeat(tuple(repeats))


def rdf_values(atoms, r_max=6.0, bins=60):
    """Return ASE's standard solid-state radial distribution function g(r)."""
    analysis_atoms = rdf_analysis_cell(atoms, r_max=r_max)

    rdf, radii = get_rdf(
        analysis_atoms,
        rmax=r_max,
        nbins=bins,
        no_dists=False,
    )

    rdf = np.asarray(rdf, dtype=float)
    radii = np.asarray(radii, dtype=float)

    if rdf.shape != (bins,) or radii.shape != (bins,):
        raise ValueError("ASE returned an unexpected RDF shape")

    if not np.all(np.isfinite(rdf)):
        raise ValueError("ASE returned non-finite RDF values")

    return rdf, radii


def rdf_l1_distance(before_atoms, after_atoms, r_max=6.0, bins=60):
    """Integrated absolute difference between two standard ASE RDF curves."""
    before_rdf, before_radii = rdf_values(
        before_atoms,
        r_max=r_max,
        bins=bins,
    )
    after_rdf, after_radii = rdf_values(
        after_atoms,
        r_max=r_max,
        bins=bins,
    )

    if not np.allclose(before_radii, after_radii):
        raise ValueError("Before and after RDF grids do not match")

    dr = r_max / bins
    return float(np.sum(np.abs(before_rdf - after_rdf)) * dr)


def topology_change_metrics(
    before_atoms,
    after_atoms,
    output_dir,
    edge_changes_filename="topology_edge_changes.csv",
):
    before_edges = neighbor_edge_set(before_atoms)
    after_edges = neighbor_edge_set(after_atoms)

    added_edges = after_edges - before_edges
    removed_edges = before_edges - after_edges

    before_coord = coordination_by_atom(before_edges, before_atoms)
    after_coord = coordination_by_atom(after_edges, after_atoms)

    coordination_changes = [
        abs(after_coord.get(atom, 0) - before_coord.get(atom, 0))
        for atom in sorted(set(before_coord) | set(after_coord))
    ]

    edge_changes_csv = Path(output_dir) / edge_changes_filename
    edge_rows = [
        {"change": "added", "edge": "-".join(edge)}
        for edge in sorted(added_edges)
    ]
    edge_rows.extend(
        {"change": "removed", "edge": "-".join(edge)}
        for edge in sorted(removed_edges)
    )

    pd.DataFrame(
        edge_rows,
        columns=["change", "edge"],
    ).to_csv(edge_changes_csv, index=False)

    return {
        "topology_edge_changes_csv": str(edge_changes_csv),
        "neighbor_edges_before": len(before_edges),
        "neighbor_edges_after": len(after_edges),
        "neighbor_edges_added": len(added_edges),
        "neighbor_edges_removed": len(removed_edges),
        "neighbor_edge_change_count": len(added_edges) + len(removed_edges),
        "neighbor_jaccard_distance": edge_jaccard_distance(
            before_edges,
            after_edges,
        ),
        "coordination_change_mean": (
            float(np.mean(coordination_changes))
            if coordination_changes
            else 0.0
        ),
        "coordination_change_max": (
            float(np.max(coordination_changes))
            if coordination_changes
            else 0.0
        ),
        "rdf_l1_distance": rdf_l1_distance(before_atoms, after_atoms),
        "rdf_method": RDF_METHOD,
    }


SYMPRECS = [1e-2, 1e-3, 1e-4]


def symmetry_signature(atoms, symprec):
    cell = (
        atoms.cell.array,
        atoms.get_scaled_positions(wrap=True),
        atoms.get_atomic_numbers(),
    )
    dataset = spglib.get_symmetry_dataset(
        cell,
        symprec=symprec,
        angle_tolerance=-1.0,
    )
    if dataset is None:
        return None

    return {
        "number": int(dataset.number),
        "operations": len(dataset.rotations),
        "unique_sites": len(np.unique(dataset.equivalent_atoms)),
    }


def symmetry_change_metrics(initial, current):
    initial_data = [symmetry_signature(initial, value) for value in SYMPRECS]
    current_data = [symmetry_signature(current, value) for value in SYMPRECS]
    pairs = [
        (before, after)
        for before, after in zip(initial_data, current_data)
        if before is not None and after is not None
    ]
    if not pairs:
        return {
            "space_group_change_fraction": np.nan,
            "symmetry_operation_retention": np.nan,
            "unique_site_change": np.nan,
        }

    return {
        "space_group_change_fraction": float(np.mean([
            before["number"] != after["number"] for before, after in pairs
        ])),
        "symmetry_operation_retention": float(np.median([
            min(after["operations"], before["operations"])
            / before["operations"]
            for before, after in pairs
        ])),
        "unique_site_change": float(np.median([
            after["unique_sites"] - before["unique_sites"] for before, after in pairs
        ])),
    }


def validate_row(row):
    if "C:\\path\\to\\" in str(row["model_path"]):
        raise RuntimeError(
            "Replace the placeholder model_path in tests.csv with a real model path."
        )

    calculator = infer_calculator(row["model_path"])

    input_path = BASE_DIR / str(row["input_path"])
    if not input_path.exists():
        raise RuntimeError(f"Input structure does not exist: {input_path}")

    if calculator == "mace":
        model_path = BASE_DIR / str(row["model_path"])
        if not model_path.exists():
            raise RuntimeError(f"MACE model does not exist: {model_path}")

    if int(row["n_steps"]) <= 0:
        raise RuntimeError("n_steps must be greater than 0")

    attack_type = str(row["attack_type"]).lower()
    if attack_type not in ["fgsm", "pgd"]:
        raise RuntimeError("attack_type must be fgsm or pgd")

    target_energy = as_none(row["target_energy"])
    if target_energy is not None:
        try:
            float(target_energy)
        except ValueError as exc:
            raise RuntimeError(
                "target_energy must be blank or a number. "
                f"Got {target_energy!r}. Check comma alignment in tests_sample.csv."
            ) from exc

    relax_fmax = as_float_or_none(row["relax_fmax"])
    if relax_fmax is not None and relax_fmax <= 0:
        raise RuntimeError("relax_fmax must be greater than 0")

    relax_max_steps = as_int_or_none(row["relax_max_steps"])
    if relax_max_steps is not None and relax_max_steps <= 0:
        raise RuntimeError("relax_max_steps must be greater than 0")

    relax_optimizer = str(value_or_default(row["relax_optimizer"], "LBFGS")).upper()
    if relax_optimizer not in ["BFGS", "LBFGS"]:
        raise RuntimeError("relax_optimizer must be BFGS or LBFGS")


def run_one(row):
    validate_row(row)

    run_id = str(row["run_id"])
    dtype_str = dtype_for_row(row)
    run_seed = run_seed_for(row)
    calculator = infer_calculator(row["model_path"])
    material_slug = summary_text(row, "material_slug")
    if material_slug is None:
        material_slug = Path(str(row["input_path"])).stem.lower()

    run_folder = summary_text(row, "run_folder")
    if run_folder is None:
        run_folder = run_id

    output_dir = output_base_for(calculator, dtype_str) / material_slug / run_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=output_dir / "run.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
        encoding="utf-8",
    )

    atoms = load_structure(BASE_DIR / str(row["input_path"]))
    if atoms is None:
        raise RuntimeError(f"Could not load structure: {row['input_path']}")

    if calculator == "mace":
        model_path = BASE_DIR / str(row["model_path"])
    elif calculator == "uma":
        model_path = Path(str(row["model_path"])).stem
    else:
        model_path = str(row["model_path"])

    relax_fmax = as_float_or_none(row["relax_fmax"])
    if relax_fmax is None:
        relax_fmax = 0.01

    relax_max_steps = as_int_or_none(row["relax_max_steps"])
    if relax_max_steps is None:
        relax_max_steps = 300

    relax_optimizer = str(value_or_default(row["relax_optimizer"], "LBFGS")).upper()

    relaxed_atoms = atoms.copy()
    relaxed_atoms = setup_calculator(
        relaxed_atoms,
        model_path,
        device=row["device"],
        dtype_str=dtype_str,
        seed=run_seed,
        calculator=calculator,
        mace_head=as_none(row["mace_head"]),
        uma_task=as_none(row["uma_task"]),
        uma_charge=as_int_or_none(row["uma_charge"]),
        uma_spin=as_int_or_none(row["uma_spin"]),
    )

    if relaxed_atoms is None:
        raise RuntimeError("Could not set up calculator for pre-attack relaxation")

    before_relax_traj = output_dir / "before_attack_relaxation.traj"

    run_relaxation_and_save_data(
        relaxed_atoms,
        before_relax_traj,
        output_dir,
        relax_fmax,
        relax_max_steps,
        relax_optimizer,
    )

    output_cif = output_dir / "perturbed.cif"

    output_file, perturbed_atoms, _ = make_attack(
        atoms=relaxed_atoms,
        model_path=model_path,
        device=row["device"],
        dtype_str=dtype_str,
        seed=run_seed,
        output_cif=output_cif,
        attack_type=str(row["attack_type"]).lower(),
        epsilon=float(row["epsilon"]),
        alpha=as_float_or_none(row["alpha"]),
        n_steps=int(row["n_steps"]),
        target_energy=as_float_or_none(row["target_energy"]),
        clip=as_bool(row["clip"]),
        verbose=True,
        calculator=calculator,
        mace_head=as_none(row["mace_head"]),
        uma_task=as_none(row["uma_task"]),
        uma_charge=as_int_or_none(row["uma_charge"]),
        uma_spin=as_int_or_none(row["uma_spin"]),
    )

    perturbed_force_csv = save_force_data(perturbed_atoms, output_dir, "perturbed")


    attack_relaxed_atoms = perturbed_atoms.copy()
    attack_relaxed_atoms.calc = perturbed_atoms.calc

    after_relax_traj = output_dir / "after_attack_relaxation.traj"

    run_relaxation_and_save_data(
        attack_relaxed_atoms,
        after_relax_traj,
        output_dir,
        relax_fmax,
        relax_max_steps,
        relax_optimizer,
    )

    before_force_csv = save_force_data(relaxed_atoms, output_dir, "before")

    after_force_csv = save_force_data(attack_relaxed_atoms, output_dir, "after")

    perturbed_topology = topology_change_metrics(
        relaxed_atoms,
        perturbed_atoms,
        output_dir,
        edge_changes_filename="topology_edge_changes_perturbed.csv",
    )

    perturbed_topology = {
        f"perturbed_{name}": value
        for name, value in perturbed_topology.items()
    }

    final_relaxed_cif = output_dir / "final_relaxed.cif"
    attack_relaxed_atoms.write(final_relaxed_cif)

    final_topology = topology_change_metrics(
        relaxed_atoms,
        attack_relaxed_atoms,
        output_dir,
        edge_changes_filename="topology_edge_changes.csv",
    )

    original_positions = relaxed_atoms.get_positions()
    perturbed_positions = attack_relaxed_atoms.get_positions()
    displacement = perturbed_positions - original_positions
    displacement_magnitudes = np.linalg.norm(displacement, axis=1)

    immediate_symmetry = symmetry_change_metrics(relaxed_atoms, perturbed_atoms)
    final_symmetry = symmetry_change_metrics(relaxed_atoms, attack_relaxed_atoms)

    symmetry_metrics = {
        **{f"perturbed_{key}": value for key, value in immediate_symmetry.items()},
        **final_symmetry,
    }

    summary = {
        "run_id": run_id,
        "material_label": summary_text(row, "material_label"),
        "material_slug": material_slug,
        "run_folder": run_folder,
        "status": "success",
        "input_path": row["input_path"],
        "model_path": row["model_path"],
        "calculator": calculator,
        "dtype_str": dtype_str,
        "seed": run_seed,
        "attack_type": row["attack_type"],
        "epsilon": float(row["epsilon"]),
        "n_steps": int(row["n_steps"]),
        "alpha": as_float_or_none(row["alpha"]),
        "clip": as_bool(row["clip"]),
        "device": row["device"],
        "output_dir": row["output_dir"],
        "actual_output_dir": str(output_dir),
        "mace_head": summary_text(row, "mace_head"),
        "uma_task": summary_text(row, "uma_task"),
        "uma_charge": as_int_or_none(row["uma_charge"]),
        "uma_spin": as_int_or_none(row["uma_spin"]),
        "target_energy": as_float_or_none(row["target_energy"]),
        "relax_fmax": relax_fmax,
        "relax_max_steps": relax_max_steps,
        "relax_optimizer": relax_optimizer,
        "contour_steps": as_int_or_none(row["contour_steps"]),
        "contour_maxstep": as_float_or_none(row["contour_maxstep"]),
        "contour_parallel_drift": as_bool(row["contour_parallel_drift"]),
        "contour_angle_limit": as_float_or_none(row["contour_angle_limit"]),
        "contour_seed": as_int_or_none(row["contour_seed"]),
        "contour_energy_target": as_float_or_none(row["contour_energy_target"]),
        "output_cif": str(output_file),
        "final_relaxed_cif": str(final_relaxed_cif),
        "before_relax_traj": str(before_relax_traj),
        "after_attack_relax_traj": str(after_relax_traj),
        "before_force_csv": str(before_force_csv),
        "perturbed_force_csv": str(perturbed_force_csv),
        "after_force_csv": str(after_force_csv),
        "mean_displacement": float(displacement_magnitudes.mean()),
        "max_displacement": float(displacement_magnitudes.max()),
        "final_energy": float(attack_relaxed_atoms.get_potential_energy()),
        **symmetry_metrics,
    }

    summary.update(perturbed_topology)
    summary.update(final_topology)

    return summary


def main(test_file=TEST_FILE):
    current_env = active_environment()
    test_file = Path(test_file)
    experiments = pd.read_csv(test_file, keep_default_na=False)
    summaries = []

    summary_override = os.environ.get("SUMMARY_FILE")
    output_root_dir = output_root()

    if summary_override:
        summary_file = Path(summary_override)
    elif current_env == "mace":
        summary_file = output_root_dir / "outputs_mace" / "summary.csv"
    elif current_env == "uma":
        summary_file = output_root_dir / "outputs_uma" / "summary.csv"
    elif current_env == "chgnet":
        summary_file = output_root_dir / "outputs_chgnet" / "summary.csv"
    else:
        summary_file = output_root_dir / "summary.csv"

    summary_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading tests from {test_file}")
    print(f"Active environment: {current_env}")

    for index, row in experiments.iterrows():
        run_id = row["run_id"]
        calculator = infer_calculator(row["model_path"])

        print(f"Running row {index + 1}: {run_id}")

        if current_env in ["mace", "uma", "chgnet"] and calculator != current_env:
            summary = {
                "run_id": run_id,
                "status": "skipped",
                "reason": f"Active environment is {current_env}, row calculator is {calculator}",
            }
            print(f"Skipped {run_id}: wrong environment")

        else:
            try:
                summary = run_one(row)
                print(f"Finished {run_id}")

            except Exception as error:
                summary = {
                    "run_id": run_id,
                    "status": "failed",
                    "error": str(error),
                }
                print(f"Failed {run_id}: {error}")

        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(summary_file, index=False)

    print("Done.")
    print(f"Results saved to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", default=TEST_FILE)
    args = parser.parse_args()
    main(args.tests)
