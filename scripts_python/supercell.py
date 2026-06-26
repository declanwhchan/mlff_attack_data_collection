#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import json
import re

import pandas as pd
from ase.io import read, write


BASE_DIR = Path(__file__).resolve().parent.parent

MATERIALS = ["graphite", "reo3", "bazro3", "cspbi3"]
REPEAT_VALUES = [1, 2, 3]
CALCULATORS = ["mace", "uma"]
DTYPE = "float64"
SEED = 42

BASE_COLUMNS = [
    "run_id", "material_label", "material_slug", "run_folder",
    "input_path", "model_path", "attack_type", "epsilon", "n_steps",
    "alpha", "clip", "device", "output_dir", "mace_head", "uma_task",
    "uma_charge", "uma_spin", "target_energy", "relax_fmax",
    "relax_max_steps", "relax_optimizer", "contour_steps", "contour_maxstep",
    "contour_parallel_drift", "contour_angle_limit", "contour_seed",
    "contour_energy_target",
]

EXTRA_COLUMNS = [
    "dtype_str", "seed", "base_material_slug", "base_material_label",
    "base_input_path", "supercell_repeat_x", "supercell_repeat_y",
    "supercell_repeat_z", "supercell_repeat_tuple", "unit_cell_atoms",
    "supercell_atoms",
]

COLUMNS = BASE_COLUMNS + EXTRA_COLUMNS


def slug(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def epsilon_tag(epsilon):
    text = f"{float(epsilon):g}"
    if "e" in text.lower():
        text = f"{float(epsilon):.8f}".rstrip("0").rstrip(".")
    return "eps" + text.replace(".", "")


def repeat_tuples():
    return [(x, y, z) for x in REPEAT_VALUES for y in REPEAT_VALUES for z in REPEAT_VALUES]


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def structure_path_for_material(row, structures_dir):
    mpid = str(row["mpid"]).strip()
    label = slug(row["material_label"])
    return Path(structures_dir) / f"{mpid}_{label}.cif"


def blank_row():
    return {column: "" for column in COLUMNS}


def make_run_row(context, supercell, model, attack, epsilon, n_steps=None, sweep=False):
    calculator = model["calculator"].lower()
    attack_name = attack["name"].lower()
    n_steps = int(n_steps if n_steps is not None else attack["n_steps"])

    alpha = attack.get("alpha")
    if alpha is None and attack.get("alpha_ratio") is not None:
        alpha = float(epsilon) * float(attack["alpha_ratio"])

    run_folder = f"{attack_name}_{epsilon_tag(epsilon)}"
    if sweep:
        run_folder = f"{run_folder}_steps{n_steps:03d}"

    row = blank_row()
    row["run_id"] = f'{supercell["material_slug"]}_{calculator}_{run_folder}'
    row["material_label"] = supercell["material_label"]
    row["material_slug"] = supercell["material_slug"]
    row["run_folder"] = run_folder
    row["input_path"] = supercell["input_path"]
    row["model_path"] = model["model_path"]
    row["attack_type"] = attack["attack_type"]
    row["epsilon"] = f"{float(epsilon):g}"
    row["n_steps"] = n_steps
    row["alpha"] = "" if alpha is None else f"{float(alpha):g}"
    row["clip"] = "" if attack.get("clip") is None else attack["clip"]
    row["device"] = context["device"]
    row["output_dir"] = "outputs"
    row["relax_fmax"] = context["relax_fmax"]
    row["relax_max_steps"] = context["relax_max_steps"]
    row["relax_optimizer"] = context["relax_optimizer"]

    row["dtype_str"] = DTYPE
    row["seed"] = SEED
    row["base_material_slug"] = context["base_material_slug"]
    row["base_material_label"] = context["base_material_label"]
    row["base_input_path"] = context["base_input_path"]
    row["supercell_repeat_x"] = supercell["repeat_x"]
    row["supercell_repeat_y"] = supercell["repeat_y"]
    row["supercell_repeat_z"] = supercell["repeat_z"]
    row["supercell_repeat_tuple"] = supercell["repeat_tuple"]
    row["unit_cell_atoms"] = supercell["unit_cell_atoms"]
    row["supercell_atoms"] = supercell["supercell_atoms"]

    if calculator == "mace":
        row["mace_head"] = model.get("mace_head", "")
    if calculator == "uma":
        row["uma_task"] = model.get("uma_task", "")
        row["uma_charge"] = model.get("uma_charge", "")
        row["uma_spin"] = model.get("uma_spin", "")

    return row


def generate(args):
    output_root = Path(args.output_root).resolve()
    structures_dir = (BASE_DIR / args.structures_dir).resolve()
    config = load_config(BASE_DIR / args.config)
    material_rows = read_csv_rows(BASE_DIR / args.materials)

    selected = [row for row in material_rows if slug(row["material_label"]) in MATERIALS]
    found = {slug(row["material_label"]) for row in selected}
    missing = sorted(set(MATERIALS) - found)
    if missing:
        raise SystemExit(f"Missing requested materials in {args.materials}: {missing}")

    tests = []
    metadata = []
    structures_root = output_root / "supercell_structures"
    structures_root.mkdir(parents=True, exist_ok=True)

    for material in selected:
        base_slug = slug(material["material_label"])
        base_path = structure_path_for_material(material, structures_dir).resolve()
        if not base_path.exists():
            raise SystemExit(f"Missing base CIF: {base_path}")

        atoms = read(base_path)
        unit_cell_atoms = len(atoms)

        context = {
            "base_material_slug": base_slug,
            "base_material_label": material["material_label"],
            "base_input_path": str(base_path),
            "device": config.get("device", "cpu"),
            "relax_fmax": config.get("relax_fmax", 0.01),
            "relax_max_steps": config.get("relax_max_steps", 300),
            "relax_optimizer": config.get("relax_optimizer", "LBFGS"),
        }

        for rx, ry, rz in repeat_tuples():
            repeat_label = f"{rx}x{ry}x{rz}"
            super_slug = f"{base_slug}_r{repeat_label}"
            cif_path = structures_root / base_slug / f"{super_slug}.cif"
            cif_path.parent.mkdir(parents=True, exist_ok=True)

            if args.force or not cif_path.exists():
                write(cif_path, atoms.repeat((rx, ry, rz)))

            supercell_atoms = unit_cell_atoms * rx * ry * rz
            supercell = {
                "material_label": f'{material["material_label"]}_r{repeat_label}',
                "material_slug": super_slug,
                "input_path": str(cif_path.resolve()),
                "repeat_x": rx,
                "repeat_y": ry,
                "repeat_z": rz,
                "repeat_tuple": repeat_label,
                "unit_cell_atoms": unit_cell_atoms,
                "supercell_atoms": supercell_atoms,
            }

            metadata.append({
                "base_material_slug": base_slug,
                "base_material_label": material["material_label"],
                "supercell_material_slug": super_slug,
                "base_input_path": str(base_path),
                "supercell_input_path": str(cif_path.resolve()),
                "repeat_x": rx,
                "repeat_y": ry,
                "repeat_z": rz,
                "repeat_tuple": repeat_label,
                "unit_cell_atoms": unit_cell_atoms,
                "supercell_atoms": supercell_atoms,
            })

            for model in config["models"]:
                for attack in config["attacks"]:
                    for epsilon in config["epsilons"]:
                        tests.append(make_run_row(context, supercell, model, attack, epsilon))

                for sweep in config.get("n_step_sweeps", []):
                    for attack in sweep["attacks"]:
                        for n_steps in sweep["n_steps"]:
                            tests.append(make_run_row(
                                context, supercell, model, sweep["epsilon"], n_steps=n_steps, sweep=True
                            ))

    output_root.mkdir(parents=True, exist_ok=True)
    tests_out = output_root / "generated_supercell_tests.csv"
    metadata_out = output_root / "supercell_metadata.csv"

    with tests_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(tests)

    pd.DataFrame(metadata).to_csv(metadata_out, index=False)
    print(f"Wrote {len(tests)} rows to {tests_out}")
    print(f"Wrote {len(metadata)} metadata rows to {metadata_out}")


def task_info(args):
    task_id = int(args.task_id)
    tasks = []
    for material in MATERIALS:
        for rx, ry, rz in repeat_tuples():
            repeat_label = f"{rx}x{ry}x{rz}"
            for calculator in CALCULATORS:
                tasks.append((material, repeat_label, calculator))

    if task_id < 1 or task_id > len(tasks):
        raise SystemExit(f"task-id must be 1..{len(tasks)}, got {task_id}")

    material, repeat_label, calculator = tasks[task_id - 1]
    output_root = Path(args.output_root).resolve()
    tests_path = output_root / "generated_supercell_tests.csv"
    rows = read_csv_rows(tests_path)

    selected = [
        row for row in rows
        if row["base_material_slug"] == material
        and row["supercell_repeat_tuple"] == repeat_label
        and row["model_path"].lower().startswith(calculator)
    ]

    if not selected:
        raise SystemExit(f"No rows for {material} {repeat_label} {calculator}")

    selected_dir = output_root / "material_tests" / DTYPE
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_csv = selected_dir / f"{calculator}_{material}_r{repeat_label}.csv"

    with selected_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    summary = output_root / "array_summaries" / f"{DTYPE}_{calculator}_{material}_r{repeat_label}_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)

    print(f"MATERIAL={material}")
    print(f"REPEAT={repeat_label}")
    print(f"CALCULATOR={calculator}")
    print(f"TEST_CSV={selected_csv}")
    print(f"SUMMARY_FILE={summary}")


def combine(args):
    output_root = Path(args.output_root).resolve()
    summary_dir = output_root / "array_summaries"

    for calculator in CALCULATORS:
        files = sorted(summary_dir.glob(f"{DTYPE}_{calculator}_*_summary.csv"))
        if not files:
            raise SystemExit(f"No {calculator} summary files found in {summary_dir}")

        combined = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        out_dir = output_root / f"outputs_{DTYPE}" / calculator
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "summary.csv"
        combined.to_csv(out_path, index=False)
        print(f"Wrote {len(combined)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--output-root", required=True)
    generate_parser.add_argument("--materials", default="tests_materials.csv")
    generate_parser.add_argument("--config", default="tests_comprehensive.json")
    generate_parser.add_argument("--structures-dir", default="mp_structures")
    generate_parser.add_argument("--force", action="store_true")

    task_parser = subparsers.add_parser("task-info")
    task_parser.add_argument("--output-root", required=True)
    task_parser.add_argument("--task-id", required=True)

    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--output-root", required=True)

    args = parser.parse_args()
    if args.command == "generate":
        generate(args)
    elif args.command == "task-info":
        task_info(args)
    elif args.command == "combine":
        combine(args)


if __name__ == "__main__":
    main()