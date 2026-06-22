from pathlib import Path
import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlff_attack.attacks import make_attack, visualize_perturbation
from mlff_attack.relaxation import load_structure, run_relaxation, setup_calculator
from mlff_attack.visualization import load_trajectory, create_visualization
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii


BASE_DIR = Path(__file__).resolve().parent.parent
TEST_FILE = BASE_DIR / "tests_sample.csv"
outputs_mace_DIR = BASE_DIR / "outputs_mace"
outputs_uma_DIR = BASE_DIR / "outputs_uma"


def active_environment():
    exe = str(sys.executable).lower()
    if ".venv-mace" in exe:
        return "mace"
    if ".venv-uma" in exe:
        return "uma"
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

    if model_name.startswith("uma"):
        return "uma"

    if model_name.startswith("mace"):
        return "mace"

    raise RuntimeError(
        "Could not infer calculator. model_path basename must start with 'mace' or 'uma'."
    )


def output_root():
    root = os.environ.get("MLFF_OUTPUT_ROOT")
    if root:
        return BASE_DIR / root
    return BASE_DIR


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

    if calculator == "uma":
        return output_root() / f"outputs_{dtype_str}" / "uma"

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


def plot_title(row, calculator, subject):
    return (
        f"{calculator.upper()} {attack_name(row)} {subject} "
        f"({attack_parameters(row)})"
    )

def save_attack_history(history, path):
    rows = []
    max_len = 0

    for values in history.values():
        if len(values) > max_len:
            max_len = len(values)

    for step in range(max_len):
        row = {"step": step}

        for key, values in history.items():
            if step < len(values):
                value = values[step]
                if isinstance(value, np.ndarray):
                    row[key] = value.tolist()
                else:
                    row[key] = value

        rows.append(row)

    pd.DataFrame(rows).to_json(path, orient="records", indent=2)


def save_relaxation_plot(atoms, traj_path, output_dir, fmax, max_steps, optimizer):
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

    traj = load_trajectory(traj_path)
    if traj is None:
        raise RuntimeError(f"Could not load relaxation trajectory: {traj_path}")

    create_visualization(
        traj,
        traj_path,
        output_dir,
        output_format="png",
        show=False,
        save_to_csv=True,
        fmax=fmax,
    )

    default_plot = output_dir / "relaxation_analysis.png"
    named_plot = output_dir / f"{traj_path.stem}_analysis.png"
    if default_plot.exists():
        default_plot.replace(named_plot)

    default_data = output_dir / "relaxation_data.csv"
    named_data = output_dir / f"{traj_path.stem}_data.csv"
    if default_data.exists():
        default_data.replace(named_data)

    default_noise = output_dir / "noise_spectrum.csv"
    named_noise = output_dir / f"{traj_path.stem}_noise_spectrum.csv"
    if default_noise.exists():
        default_noise.replace(named_noise)

    return named_plot


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


def save_force_plot(atoms, output_dir, label, title):
    positions = atoms.get_positions()
    forces = atoms.get_forces()
    force_magnitudes = np.linalg.norm(forces, axis=1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=force_magnitudes,
        s=90,
    )

    ax.quiver(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        forces[:, 0],
        forces[:, 1],
        forces[:, 2],
        length=0.2,
        normalize=True,
    )

    ax.set_title(title)
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")

    fig.colorbar(scatter, ax=ax, label="Force Magnitude")

    force_png = output_dir / f"{label}_forces.png"
    fig.savefig(force_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return force_png


def atom_signature(symbols, index):
    return f"{symbols[index]}{index}"


def neighbor_edge_set(atoms, scale=1.25, min_cutoff=1.2, max_cutoff=3.2):
    symbols = atoms.get_chemical_symbols()

    cutoffs = []
    for atom in atoms:
        radius = float(covalent_radii[atom.number])
        if not np.isfinite(radius) or radius <= 0:
            radius = 0.8
        cutoffs.append(max(min_cutoff, min(max_cutoff, radius * scale)))

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


def coordination_by_atom(edges, atoms):
    symbols = atoms.get_chemical_symbols()
    counts = {atom_signature(symbols, index): 0 for index in range(len(atoms))}

    for a, b in edges:
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1

    return counts


def rdf_histogram(atoms, r_max=6.0, bins=60):
    distances = atoms.get_all_distances(mic=True)
    values = []

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            distance = float(distances[i, j])
            if 0 < distance <= r_max:
                values.append(distance)

    hist, edges = np.histogram(values, bins=bins, range=(0.0, r_max), density=False)
    hist = hist.astype(float)

    total = hist.sum()
    if total > 0:
        hist /= total

    return hist


def topology_change_metrics(before_atoms, after_atoms, output_dir):
    before_edges = neighbor_edge_set(before_atoms)
    after_edges = neighbor_edge_set(after_atoms)

    added_edges = after_edges - before_edges
    removed_edges = before_edges - after_edges
    union_edges = before_edges | after_edges

    if union_edges:
        neighbor_jaccard_distance = 1.0 - (len(before_edges & after_edges) / len(union_edges))
    else:
        neighbor_jaccard_distance = 0.0

    before_coord = coordination_by_atom(before_edges, before_atoms)
    after_coord = coordination_by_atom(after_edges, after_atoms)

    coord_delta = []
    for atom in sorted(set(before_coord) | set(after_coord)):
        coord_delta.append(abs(after_coord.get(atom, 0) - before_coord.get(atom, 0)))

    before_rdf = rdf_histogram(before_atoms)
    after_rdf = rdf_histogram(after_atoms)
    rdf_l1_distance = float(np.sum(np.abs(before_rdf - after_rdf)))

    edge_rows = []
    for edge in sorted(added_edges):
        edge_rows.append({"change": "added", "edge": "-".join(edge)})
    for edge in sorted(removed_edges):
        edge_rows.append({"change": "removed", "edge": "-".join(edge)})

    edge_changes_csv = output_dir / "topology_edge_changes.csv"
    pd.DataFrame(edge_rows, columns=["change", "edge"]).to_csv(edge_changes_csv, index=False)

    return {
        "topology_edge_changes_csv": str(edge_changes_csv),
        "neighbor_edges_before": len(before_edges),
        "neighbor_edges_after": len(after_edges),
        "neighbor_edges_added": len(added_edges),
        "neighbor_edges_removed": len(removed_edges),
        "neighbor_edge_change_count": len(added_edges) + len(removed_edges),
        "neighbor_jaccard_distance": float(neighbor_jaccard_distance),
        "coordination_change_mean": float(np.mean(coord_delta)) if coord_delta else 0.0,
        "coordination_change_max": float(np.max(coord_delta)) if coord_delta else 0.0,
        "rdf_l1_distance": rdf_l1_distance,
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
    else:
        model_path = Path(str(row["model_path"])).stem

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
    before_relax_plot = save_relaxation_plot(
        relaxed_atoms,
        before_relax_traj,
        output_dir,
        relax_fmax,
        relax_max_steps,
        relax_optimizer,
    )

    output_cif = output_dir / "perturbed.cif"

    output_file, perturbed_atoms, attack_history = make_attack(
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
    perturbed_force_png = save_force_plot(
        perturbed_atoms,
        output_dir,
        "perturbed",
        plot_title(row, calculator, "Forces After Perturbation Before Relaxation"),
    )

    attack_relaxed_atoms = perturbed_atoms.copy()
    attack_relaxed_atoms.calc = perturbed_atoms.calc

    after_relax_traj = output_dir / "after_attack_relaxation.traj"
    after_relax_plot = save_relaxation_plot(
        attack_relaxed_atoms,
        after_relax_traj,
        output_dir,
        relax_fmax,
        relax_max_steps,
        relax_optimizer,
    )

    history_file = output_dir / "history.json"
    save_attack_history(attack_history, history_file)

    before_force_csv = save_force_data(relaxed_atoms, output_dir, "before")
    before_force_png = save_force_plot(
        relaxed_atoms,
        output_dir,
        "before",
        plot_title(row, calculator, "Forces Before Attack Relaxation"),
    )

    after_force_csv = save_force_data(attack_relaxed_atoms, output_dir, "after")
    after_force_png = save_force_plot(
        attack_relaxed_atoms,
        output_dir,
        "after",
        plot_title(row, calculator, "Forces After Attack Relaxation"),
    )

    final_relaxed_cif = output_dir / "final_relaxed.cif"
    attack_relaxed_atoms.write(final_relaxed_cif)

    fig = visualize_perturbation(
        relaxed_atoms,
        attack_relaxed_atoms,
        epsilon=float(row["epsilon"]),
        outdir=None,
    )

    fig.suptitle(
        plot_title(row, calculator, "Perturbed Structure"),
        fontsize=14,
        fontweight="bold",
    )

    perturbation_png = output_dir / "perturbation.png"
    fig.savefig(perturbation_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    duplicate_png = output_dir / "perturbation_analysis.png"
    if duplicate_png.exists():
        duplicate_png.unlink()

    original_positions = relaxed_atoms.get_positions()
    perturbed_positions = attack_relaxed_atoms.get_positions()
    displacement = perturbed_positions - original_positions
    displacement_magnitudes = np.linalg.norm(displacement, axis=1)
    topology_metrics = topology_change_metrics(
        relaxed_atoms,
        attack_relaxed_atoms,
        output_dir,
    )

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
        "history_file": str(history_file),
        "before_relax_traj": str(before_relax_traj),
        "before_relax_plot": str(before_relax_plot),
        "after_attack_relax_traj": str(after_relax_traj),
        "after_attack_relax_plot": str(after_relax_plot),
        "before_force_csv": str(before_force_csv),
        "before_force_png": str(before_force_png),
        "perturbed_force_csv": str(perturbed_force_csv),
        "perturbed_force_png": str(perturbed_force_png),
        "after_force_csv": str(after_force_csv),
        "after_force_png": str(after_force_png),
        "perturbation_png": str(perturbation_png),
        "mean_displacement": float(displacement_magnitudes.mean()),
        "max_displacement": float(displacement_magnitudes.max()),
        "final_energy": float(attack_relaxed_atoms.get_potential_energy()),
        **topology_metrics,
    }

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
    else:
        summary_file = output_root_dir / "summary.csv"

    summary_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading tests from {test_file}")
    print(f"Active environment: {current_env}")

    for index, row in experiments.iterrows():
        run_id = row["run_id"]
        calculator = infer_calculator(row["model_path"])

        print(f"Running row {index + 1}: {run_id}")

        if current_env in ["mace", "uma"] and calculator != current_env:
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
