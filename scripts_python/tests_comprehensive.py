#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

COLUMNS = [
    "run_id",
    "input_path",
    "model_path",
    "attack_type",
    "epsilon",
    "n_steps",
    "alpha",
    "clip",
    "device",
    "output_dir",
    "mace_head",
    "uma_task",
    "uma_charge",
    "uma_spin",
    "target_energy",
    "relax_fmax",
    "relax_max_steps",
    "relax_optimizer",
    "contour_steps",
    "contour_maxstep",
    "contour_parallel_drift",
    "contour_angle_limit",
    "contour_seed",
    "contour_energy_target",
]


def require(condition, message):
    if not condition:
        raise SystemExit(f"ERROR: {message}")


def epsilon_tag(epsilon):
    text = f"{float(epsilon):g}"
    if "e" in text.lower():
        text = f"{float(epsilon):.8f}".rstrip("0").rstrip(".")
    return "eps" + text.replace(".", "")


def step_tag(n_steps):
    return f"steps{int(n_steps)}"



def structure_prefix(input_path):
    name = Path(input_path).stem.lower()
    if name.startswith("tl"):
        return "tl"
    return "".join(ch for ch in name if ch.isalnum())[:12]


def blank_row():
    return {column: "" for column in COLUMNS}


def validate_config(config):
    required_keys = [
        "input_path",
        "device",
        "epsilons",
        "relax_fmax",
        "relax_max_steps",
        "relax_optimizer",
        "models",
        "attacks",
    ]

    for key in required_keys:
        require(key in config, f"Missing key in tests_comprehensive.json: {key}")

    require((BASE_DIR / config["input_path"]).exists(), f"Missing input structure: {config['input_path']}")
    require(len(config["epsilons"]) > 0, "epsilons must contain at least one value")
    require(len(config["models"]) > 0, "models must contain at least one model")
    require(len(config["attacks"]) > 0, "attacks must contain at least one attack")

    for model in config["models"]:
        calculator = str(model.get("calculator", "")).lower()
        require(calculator in {"mace", "uma"}, f"Unknown calculator: {calculator}")
        require("model_path" in model, f"Model entry missing model_path: {model}")

        if calculator == "mace":
            require((BASE_DIR / model["model_path"]).exists(), f"Missing MACE model file: {model['model_path']}")
            require("mace_head" in model, "MACE model entry should include mace_head")

        if calculator == "uma":
            model_path = Path(str(model["model_path"]))
            possible_local_file = BASE_DIR / model_path
            possible_pt_file = BASE_DIR / f"{model_path.stem}.pt"
            require(
                possible_local_file.exists() or possible_pt_file.exists(),
                f"Missing UMA model file. Expected {model_path} or {model_path.stem}.pt",
            )
            require("uma_task" in model, "UMA model entry should include uma_task")
            require("uma_charge" in model, "UMA model entry should include uma_charge")
            require("uma_spin" in model, "UMA model entry should include uma_spin")

    for attack in config["attacks"]:
        require("name" in attack, f"Attack entry missing name: {attack}")
        require("attack_type" in attack, f"Attack entry missing attack_type: {attack}")
        require("n_steps" in attack, f"Attack entry missing n_steps: {attack}")

        attack_type = str(attack["attack_type"]).lower()
        require(attack_type in {"fgsm", "pgd"}, f"attack_type must be fgsm or pgd, got: {attack_type}")

        n_steps = int(attack["n_steps"])
        require(n_steps > 0, f"n_steps must be positive for attack: {attack['name']}")


def build_row(config, prefix, model, attack, epsilon, n_steps, run_suffix=""):
    calculator = model["calculator"].lower()
    attack_name = attack["name"].lower()
    row = blank_row()

    alpha = attack.get("alpha")
    if alpha is None and attack.get("alpha_ratio") is not None:
        alpha = float(epsilon) * float(attack["alpha_ratio"])

    suffix = f"_{run_suffix}" if run_suffix else ""
    row["run_id"] = f"{prefix}_{calculator}_{attack_name}_{epsilon_tag(epsilon)}{suffix}"
    row["input_path"] = config["input_path"]
    row["model_path"] = model["model_path"]
    row["attack_type"] = attack["attack_type"]
    row["epsilon"] = f"{float(epsilon):g}"
    row["n_steps"] = int(n_steps)
    row["alpha"] = "" if alpha is None else f"{float(alpha):g}"
    row["clip"] = "" if attack.get("clip") is None else attack["clip"]
    row["device"] = config.get("device", "cpu")
    row["output_dir"] = "outputs"
    row["relax_fmax"] = config.get("relax_fmax", 0.01)
    row["relax_max_steps"] = config.get("relax_max_steps", 300)
    row["relax_optimizer"] = config.get("relax_optimizer", "LBFGS")

    if calculator == "mace":
        row["mace_head"] = model.get("mace_head", "")

    if calculator == "uma":
        row["uma_task"] = model.get("uma_task", "")
        row["uma_charge"] = model.get("uma_charge", "")
        row["uma_spin"] = model.get("uma_spin", "")

    return row


def build_rows(config):
    rows = []
    prefix = structure_prefix(config["input_path"])

    for model in config["models"]:
        for attack in config["attacks"]:
            for epsilon in config["epsilons"]:
                rows.append(
                    build_row(
                        config=config,
                        prefix=prefix,
                        model=model,
                        attack=attack,
                        epsilon=epsilon,
                        n_steps=attack["n_steps"],
                    )
                )

    for sweep in config.get("n_step_sweeps", []):
        epsilon = sweep["epsilon"]
        for model in config["models"]:
            for attack in sweep["attacks"]:
                for n_steps in sweep["n_steps"]:
                    rows.append(
                        build_row(
                            config=config,
                            prefix=prefix,
                            model=model,
                            attack=attack,
                            epsilon=epsilon,
                            n_steps=n_steps,
                            run_suffix=step_tag(n_steps),
                        )
                    )

    seen = set()
    duplicates = []
    for row in rows:
        run_id = row["run_id"]
        if run_id in seen:
            duplicates.append(run_id)
        seen.add(run_id)

    require(not duplicates, f"Duplicate run_id values generated: {sorted(set(duplicates))}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate generated_tests.csv from tests_comprehensive.json.")
    parser.add_argument("--config", default="tests_comprehensive.json")
    parser.add_argument("--output", default="generated_tests.csv")
    args = parser.parse_args()

    config_path = BASE_DIR / args.config
    output_path = BASE_DIR / args.output

    require(config_path.exists(), f"Missing config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    validate_config(config)
    rows = build_rows(config)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows in {output_path.name}")
    print("Expected run count:")
    print(f"  models: {len(config['models'])}")
    print(f"  attacks: {len(config['attacks'])}")
    print(f"  epsilons: {len(config['epsilons'])}")
    print(f"  total: {len(config['models']) * len(config['attacks']) * len(config['epsilons'])}")


if __name__ == "__main__":
    main()
