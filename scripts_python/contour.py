#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ase.io import read
from ase.md.contour_exploration import ContourExploration
from ase.neighborlist import neighbor_list
from ase.optimize import LBFGS

from mlff_attack.relaxation import setup_calculator


BASE_DIR = Path(__file__).resolve().parent.parent


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def as_float(value, default=None):
    value = clean(value)
    if value == "":
        return default
    return float(value)


def as_int(value, default=None):
    value = clean(value)
    if value == "":
        return default
    return int(float(value))


def slug_from_input(path):
    return Path(str(path)).stem.lower()


def infer_calculator(model_path):
    name = Path(str(model_path)).name.lower()
    if name.startswith("mace"):
        return "mace"
    if name.startswith("uma"):
        return "uma"
    raise RuntimeError(f"Cannot infer calculator from model_path={model_path!r}")


def parse_betas(text, default):
    if text:
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    return [float(x) for x in default]


def beta_tag(beta):
    return f"beta_{int(round(float(beta) * 1000)):03d}"


def read_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jobs(tests_path):
    with Path(tests_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    jobs = {}
    for row in rows:
        calculator = infer_calculator(row["model_path"])
        material_slug = clean(row.get("material_slug")) or slug_from_input(row["input_path"])
        material_label = clean(row.get("material_label")) or material_slug
        key = (calculator, material_slug)
        if key not in jobs:
            jobs[key] = {
                "calculator": calculator,
                "material_slug": material_slug,
                "material_label": material_label,
                "input_path": row["input_path"],
                "model_path": row["model_path"],
                "device": clean(row.get("device")) or "cpu",
                "mace_head": clean(row.get("mace_head")),
                "uma_task": clean(row.get("uma_task")),
                "uma_charge": clean(row.get("uma_charge")),
                "uma_spin": clean(row.get("uma_spin")),
            }

    return [jobs[key] for key in sorted(jobs)]


def select_jobs(jobs, calculator=None, material_slug=None):
    selected = jobs
    if calculator:
        selected = [job for job in selected if job["calculator"] == calculator]
    if material_slug:
        selected = [job for job in selected if job["material_slug"] == material_slug]

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id and not calculator and not material_slug:
        index = int(task_id) - 1
        if index < 0 or index >= len(selected):
            raise SystemExit(f"ERROR: SLURM_ARRAY_TASK_ID must be 1..{len(selected)}, got {task_id}")
        selected = [selected[index]]

    return selected


def setup_job_calculator(atoms, job, dtype_str):
    calculator = job["calculator"]
    model_path = BASE_DIR / job["model_path"] if calculator == "mace" else Path(job["model_path"]).stem

    atoms = setup_calculator(
        atoms,
        model_path,
        device=job["device"],
        dtype_str=dtype_str,
        calculator=calculator,
        mace_head=job["mace_head"] or None,
        uma_task=job["uma_task"] or None,
        uma_charge=as_int(job["uma_charge"]),
        uma_spin=as_int(job["uma_spin"]),
    )
    if atoms is None:
        raise RuntimeError("setup_calculator returned None")
    return atoms


def initial_velocities(atoms, seed):
    rng = np.random.default_rng(seed)
    velocities = rng.normal(size=(len(atoms), 3))
    velocities -= velocities.mean(axis=0)
    atoms.set_velocities(velocities)


def neighbor_pairs(atoms):
    cutoffs = []
    for _ in atoms:
        cutoffs.append(3.0)
    try:
        i, j, d = neighbor_list("ijd", atoms, cutoffs)
        mask = i < j
        return list(zip(i[mask], j[mask]))
    except Exception:
        positions = atoms.get_positions()
        pairs = []
        for i in range(len(positions)):
            distances = np.linalg.norm(positions - positions[i], axis=1)
            distances[i] = np.inf
            j = int(np.argmin(distances))
            pairs.append(tuple(sorted((i, j))))
        return sorted(set(pairs))


def separation_distance(atoms, pairs):
    if not pairs:
        return np.nan
    positions = atoms.get_positions()
    distances = [atoms.get_distance(i, j, mic=True) for i, j in pairs]
    return float(np.mean(distances))


def frame_displacement(atoms, initial_positions):
    displacement = atoms.get_positions() - initial_positions
    return float(np.mean(np.linalg.norm(displacement, axis=1)))


def max_frame_displacement(atoms, initial_positions):
    displacement = atoms.get_positions() - initial_positions
    return float(np.max(np.linalg.norm(displacement, axis=1)))


def force_delta(atoms, initial_forces):
    forces = atoms.get_forces()
    return float(np.mean(np.linalg.norm(forces - initial_forces, axis=1)))


def out_of_plane_angle(prev_prev_positions, prev_positions, positions):
    if prev_prev_positions is None or prev_positions is None:
        return np.nan

    a = prev_positions - prev_prev_positions
    b = positions - prev_positions
    a = a.reshape(-1)
    b = b.reshape(-1)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return np.nan

    a = a / norm_a
    b = b / norm_b
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def relax_if_requested(atoms, fmax, max_steps):
    if fmax is None or max_steps <= 0:
        return atoms

    relaxed = atoms.copy()
    relaxed.calc = atoms.calc
    opt = LBFGS(relaxed, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    return relaxed


def run_contour(job, beta, config, args):
    calculator = job["calculator"]
    material_slug = job["material_slug"]
    dtype_str = args.dtype_str
    outdir = BASE_DIR / f"outputs_{dtype_str}" / calculator / "contour" / material_slug / beta_tag(beta)
    outdir.mkdir(parents=True, exist_ok=True)

    atoms = read(BASE_DIR / job["input_path"])
    atoms = setup_job_calculator(atoms, job, dtype_str)

    pre_relax_fmax = as_float(config.get("contour_pre_relax_fmax"), None)
    pre_relax_steps = as_int(config.get("contour_pre_relax_max_steps"), 0)
    atoms = relax_if_requested(atoms, pre_relax_fmax, pre_relax_steps)

    seed = as_int(args.seed, as_int(config.get("contour_seed"), 12345))
    initial_velocities(atoms, seed + int(round(beta * 1000)))

    energy_target = as_float(args.energy_target, as_float(config.get("contour_energy_target"), None))
    initial_energy = float(atoms.get_potential_energy())
    if energy_target is None:
        energy_target = initial_energy

    initial_positions = atoms.get_positions().copy()
    initial_forces = atoms.get_forces().copy()
    pairs = neighbor_pairs(atoms)

    traj_path = outdir / "contour.traj"
    log_path = outdir / "contour.log"

    dyn = ContourExploration(
        atoms,
        maxstep=as_float(args.maxstep, as_float(config.get("contour_maxstep"), 0.01)),
        parallel_drift=float(beta),
        energy_target=energy_target,
        angle_limit=as_float(args.angle_limit, as_float(config.get("contour_angle_limit"), 20.0)),
        rng=np.random.default_rng(seed + int(round(beta * 1000))),
        trajectory=str(traj_path),
        logfile=str(log_path),
        loginterval=1,
    )

    steps = as_int(args.steps, as_int(config.get("contour_steps"), 500))
    rows = []

    prev_prev_positions = None
    prev_positions = None

    for step in range(steps + 1):
        energy = float(atoms.get_potential_energy())
        row = {
            "step": step,
            "material_label": job["material_label"],
            "material_slug": material_slug,
            "calculator": calculator,
            "beta": float(beta),
            "energy_target_ev": energy_target,
            "energy_ev": energy,
            "energy_deviation_ev": energy - energy_target,
            "energy_deviation_mev_per_atom": (energy - energy_target) * 1000.0 / len(atoms),
            "step_size_a": float(getattr(dyn, "step_size", np.nan)),
            "curvature_1_per_a": float(getattr(dyn, "curvature", np.nan)),
            "separation_distance_a": separation_distance(atoms, pairs),
            "mean_displacement_from_initial_a": frame_displacement(atoms, initial_positions),
            "max_displacement_from_initial_a": max_frame_displacement(atoms, initial_positions),
            "mean_force_delta_from_initial_ev_a": force_delta(atoms, initial_forces),
            "out_of_plane_angle_deg": out_of_plane_angle(prev_prev_positions, prev_positions, atoms.get_positions()),
        }
        rows.append(row)

        if step == steps:
            break

        prev_prev_positions = None if prev_positions is None else prev_positions.copy()
        prev_positions = atoms.get_positions().copy()
        dyn.run(1)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "contour_metrics.csv", index=False)
    atoms.write(outdir / "final_contour.cif")

    return {
        "status": "success",
        "calculator": calculator,
        "dtype_str": dtype_str,
        "material_label": job["material_label"],
        "material_slug": material_slug,
        "beta": float(beta),
        "steps": steps,
        "output_dir": str(outdir),
        "metrics_csv": str(outdir / "contour_metrics.csv"),
        "traj": str(traj_path),
        "log": str(log_path),
        "final_contour_cif": str(outdir / "final_contour.cif"),
        "energy_target_ev": energy_target,
        "initial_energy_ev": initial_energy,
        "mean_abs_energy_deviation_mev_per_atom": float(metrics["energy_deviation_mev_per_atom"].abs().mean()),
        "max_abs_energy_deviation_mev_per_atom": float(metrics["energy_deviation_mev_per_atom"].abs().max()),
        "mean_step_size_a": float(metrics["step_size_a"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_curvature_1_per_a": float(metrics["curvature_1_per_a"].replace([np.inf, -np.inf], np.nan).mean()),
        "max_displacement_from_initial_a": float(metrics["max_displacement_from_initial_a"].max()),
        "mean_force_delta_from_initial_ev_a": float(metrics["mean_force_delta_from_initial_ev_a"].mean()),
    }


def append_summary(dtype_str, calculator, rows):
    summary_path = BASE_DIR / f"outputs_{dtype_str}" / calculator / "contour" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(["dtype_str", "calculator", "material_slug", "beta"], keep="last")
    combined.to_csv(summary_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", default="generated_material_tests.csv")
    parser.add_argument("--config", default="tests_comprehensive.json")
    parser.add_argument("--calculator", choices=["mace", "uma"])
    parser.add_argument(
        "--dtype-str",
        choices=["float32", "float64"],
        default=os.environ.get("MLFF_DTYPE", "float64"),
    )
    parser.add_argument("--material-slug")
    parser.add_argument("--betas")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--maxstep", type=float)
    parser.add_argument("--angle-limit", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--energy-target", type=float)
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = read_config(BASE_DIR / args.config)
    jobs = select_jobs(read_jobs(BASE_DIR / args.tests), args.calculator, args.material_slug)
    betas = parse_betas(args.betas, config.get("contour_betas", [0.10, 0.05, 0.00]))

    if args.list_jobs:
        for i, job in enumerate(jobs, start=1):
            print(f"{i},{job['calculator']},{job['material_slug']},{job['input_path']}")
        return

    if args.dry_run:
        for job in jobs:
            for beta in betas:
                print(f"DRY RUN: {job['calculator']} {job['material_slug']} beta={beta:g}")
        return

    summaries_by_calculator = {"mace": [], "uma": []}

    for job in jobs:
        for beta in betas:
            print(f"Running contour: {job['calculator']} {job['material_slug']} beta={beta:g}", flush=True)
            try:
                summary = run_contour(job, beta, config, args)
            except Exception as exc:
                summary = {
                    "status": "failed",
                    "calculator": job["calculator"],
                    "material_label": job["material_label"],
                    "material_slug": job["material_slug"],
                    "beta": float(beta),
                    "error": str(exc),
                }
                print(f"FAILED: {job['calculator']} {job['material_slug']} beta={beta:g}: {exc}", flush=True)

            summaries_by_calculator[job["calculator"]].append(summary)

    for calculator, rows in summaries_by_calculator.items():
        if rows:
            append_summary(args.dtype_str, calculator, rows)


if __name__ == "__main__":
    main()
