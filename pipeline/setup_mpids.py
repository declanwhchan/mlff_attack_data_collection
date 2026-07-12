#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import os
import re

BASE_DIR = Path(__file__).resolve().parent.parent

COLUMNS = [
    "run_id", "material_label", "material_slug", "run_folder",
    "input_path", "model_path", "attack_type", "epsilon", "n_steps",
    "alpha", "clip", "device", "output_dir", "mace_head", "uma_task",
    "uma_charge", "uma_spin", "target_energy", "relax_fmax",
    "relax_max_steps", "relax_optimizer", "contour_steps", "contour_maxstep",
    "contour_parallel_drift", "contour_angle_limit", "contour_seed",
    "contour_energy_target",
]


def require(condition, message):
    if not condition:
        raise SystemExit(f"ERROR: {message}")


def require_package(module_name, pip_name=None):
    if importlib.util.find_spec(module_name) is None:
        pip_name = pip_name or module_name
        raise SystemExit(
            f"ERROR: Missing Python package '{pip_name}'. "
            f"Install it with: python -m pip install {pip_name}"
        )


def slug(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def epsilon_tag(epsilon):
    text = f"{float(epsilon):g}"
    if "e" in text.lower():
        text = f"{float(epsilon):.8f}".rstrip("0").rstrip(".")
    return "eps" + text.replace(".", "")


def blank_row():
    return {column: "" for column in COLUMNS}


def read_materials(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    require(rows, f"No materials found in {path}")

    for row in rows:
        require(row.get("material_label"), "material_label is required")
        require(row.get("mpid"), f"mpid is required for {row}")

    return rows


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    for key in ["models", "attacks", "epsilons"]:
        require(key in config and config[key], f"{path} must define non-empty {key}")

    return config


def download_structures(materials, structures_dir, force=False):
    require(
        os.environ.get("MP_API_KEY"),
        "Set MP_API_KEY first. In PowerShell: $env:MP_API_KEY='your_key_here'",
    )
    require_package("mp_api", "mp-api")
    require_package("pymatgen", "pymatgen")

    from mp_api.client import MPRester

    structures_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}

    with MPRester() as mpr:
        for material in materials:
            mpid = material["mpid"].strip()
            label = slug(material["material_label"])
            cif_path = structures_dir / f"{mpid}_{label}.cif"

            if cif_path.exists() and not force:
                print(f"Using existing {cif_path}")
                output_paths[mpid] = cif_path
                continue

            print(f"Downloading {mpid} -> {cif_path}")
            structure = mpr.get_structure_by_material_id(mpid)
            structure.to(filename=str(cif_path))
            output_paths[mpid] = cif_path

    return output_paths


def validate_config_files(config):
    for model in config["models"]:
        calculator = str(model.get("calculator", "")).lower()
        require(calculator in {"mace", "uma", "chgnet"}, f"Unknown calculator: {calculator}")
        require(model.get("model_path"), f"Model missing model_path: {model}")

        if calculator == "mace":
            model_path = BASE_DIR / model["model_path"]
            require(
                model_path.exists(),
                f"Missing MACE model file: {model_path}",
            )

        elif calculator == "uma":
            model_path = Path(str(model["model_path"]))
            require(
                (BASE_DIR / model_path).exists()
                or (BASE_DIR / f"{model_path.stem}.pt").exists(),
                f"Missing UMA model file: {model['model_path']} "
                f"or {model_path.stem}.pt",
            )

        elif calculator == "chgnet":
            require(
                model["model_path"] in {
                    "chgnet-0.2.0",
                    "chgnet-0.3.0",
                    "chgnet-r2scan",
                },
                f"Unsupported CHGNet model: {model['model_path']}",
            )


def add_row(rows, material_label, material_slug, input_path, model, attack, epsilon, n_steps=None, sweep=False):
    calculator = model["calculator"].lower()
    attack_name = attack["name"].lower()
    n_steps = int(n_steps if n_steps is not None else attack["n_steps"])

    alpha = attack.get("alpha")
    if alpha is None and attack.get("alpha_ratio") is not None:
        alpha = float(epsilon) * float(attack["alpha_ratio"])

    if sweep:
        run_folder = f"{attack_name}_{epsilon_tag(epsilon)}_steps{n_steps:03d}"
    else:
        run_folder = f"{attack_name}_{epsilon_tag(epsilon)}"

    row = blank_row()
    row["material_label"] = material_label
    row["material_slug"] = material_slug
    row["run_folder"] = run_folder
    row["run_id"] = f"{material_slug}_{calculator}_{run_folder}"
    row["input_path"] = str(input_path).replace("\\", "/")
    row["model_path"] = model["model_path"]
    row["attack_type"] = attack["attack_type"]
    row["epsilon"] = f"{float(epsilon):g}"
    row["n_steps"] = n_steps
    row["alpha"] = "" if alpha is None else f"{float(alpha):g}"
    row["clip"] = "" if attack.get("clip") is None else attack["clip"]
    row["device"] = add_row.device
    row["output_dir"] = "outputs"
    row["relax_fmax"] = add_row.relax_fmax
    row["relax_max_steps"] = add_row.relax_max_steps
    row["relax_optimizer"] = add_row.relax_optimizer

    if calculator == "mace":
        row["mace_head"] = model.get("mace_head", "")

    elif calculator == "uma":
        row["uma_task"] = model.get("uma_task", "")
        row["uma_charge"] = model.get("uma_charge", "")
        row["uma_spin"] = model.get("uma_spin", "")

    rows.append(row)


def build_rows(materials, structure_paths, config):
    validate_config_files(config)
    add_row.device = config.get("device", "cpu")
    add_row.relax_fmax = config.get("relax_fmax", 0.01)
    add_row.relax_max_steps = config.get("relax_max_steps", 300)
    add_row.relax_optimizer = config.get("relax_optimizer", "LBFGS")

    rows = []
    for material in materials:
        material_label = material["material_label"]
        material_slug = slug(material_label)
        input_path = structure_paths[material["mpid"]].relative_to(BASE_DIR)

        for model in config["models"]:
            for attack in config["attacks"]:
                for epsilon in config["epsilons"]:
                    add_row(
                        rows,
                        material_label,
                        material_slug,
                        input_path,
                        model,
                        attack,
                        epsilon,
                        sweep=False,
                    )

            for sweep in config.get("n_step_sweeps", []):
                epsilon = sweep["epsilon"]
                for attack in sweep["attacks"]:
                    for n_steps in sweep["n_steps"]:
                        add_row(
                            rows,
                            material_label,
                            material_slug,
                            input_path,
                            model,
                            attack,
                            epsilon,
                            n_steps=n_steps,
                            sweep=True,
                        )

    return rows


def write_tests(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--materials", default="datasets/2d_structures/tests_materials.csv")
    parser.add_argument("--config", default="datasets/2d_structures/tests_comprehensive.json")
    parser.add_argument("--tests-out", default="generated_material_tests.csv")
    parser.add_argument("--structures-dir", default="mp_structures")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    materials_path = BASE_DIR / args.materials
    config_path = BASE_DIR / args.config
    tests_out = BASE_DIR / args.tests_out
    structures_dir = BASE_DIR / args.structures_dir

    require(materials_path.exists(), f"Missing materials file: {materials_path}")
    require(config_path.exists(), f"Missing config file: {config_path}")

    materials = read_materials(materials_path)
    config = load_config(config_path)

    structure_paths = download_structures(
        materials,
        structures_dir,
        force=args.force_download,
    )

    if args.download_only:
        print("Download-only mode complete.")
        return

    rows = build_rows(materials, structure_paths, config)
    write_tests(rows, tests_out)

    if args.run:
        import run_tests
        run_tests.main(tests_out)
    else:
        print(f"Next: python pipeline/run_tests.py --tests {tests_out.name}")


if __name__ == "__main__":
    main()
