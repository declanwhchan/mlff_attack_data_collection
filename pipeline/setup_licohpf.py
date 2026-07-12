#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import json
import math
import shutil

from ase.io import read, write


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = (
    BASE_DIR
    / "datasets"
    / "licohpf_database"
    / "tests_comprehensive.json"
)

COLUMNS = [
    "run_id",
    "material_label",
    "material_slug",
    "run_folder",
    "input_path",
    "model_id",
    "display_name",
    "calculator",
    "calculator_backend",
    "model_path",
    "attack_type",
    "epsilon",
    "n_steps",
    "alpha",
    "clip",
    "device",
    "dtype_str",
    "seed",
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


def relative_to_base(path):
    return str(path.resolve().relative_to(BASE_DIR.resolve())).replace(
        "\\",
        "/",
    )


def load_config(path):
    require(path.exists(), f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required_keys = {
        "input_xyz",
        "expected_structures",
        "models",
        "attacks",
        "epsilons",
    }

    missing = sorted(required_keys.difference(config))
    require(
        not missing,
        f"Configuration is missing keys: {', '.join(missing)}",
    )

    require(config["models"], "models must not be empty")
    require(config["attacks"], "attacks must not be empty")
    require(config["epsilons"], "epsilons must not be empty")

    return config


def validate_models(config):
    valid_model_ids = {
        "mace_mh",
        "uma",
        "mtp",
        "chgnet",
        "mace_model",
    }

    expected_backends = {
        "mace_mh": "mace",
        "uma": "uma",
        "mtp": "mtp",
        "chgnet": "chgnet",
        "mace_model": "mace",
    }

    seen = set()

    for model in config["models"]:
        model_id = str(model.get("model_id", "")).strip().lower()
        backend = str(
            model.get(
                "calculator_backend",
                model.get("calculator", ""),
            )
        ).strip().lower()

        require(
            model_id in valid_model_ids,
            f"Invalid model_id: {model_id!r}",
        )
        require(
            model_id not in seen,
            f"Duplicate model_id: {model_id}",
        )
        require(
            backend == expected_backends[model_id],
            (
                f"{model_id} requires backend "
                f"{expected_backends[model_id]!r}, got {backend!r}"
            ),
        )
        require(
            model.get("model_path"),
            f"{model_id} is missing model_path",
        )
        require(
            model.get("dtypes"),
            f"{model_id} is missing dtypes",
        )

        seen.add(model_id)

        dtypes = {
            str(value).strip().lower()
            for value in model["dtypes"]
        }

        require(
            dtypes.issubset({"float32", "float64"}),
            f"{model_id} contains an invalid dtype: {sorted(dtypes)}",
        )

        if model_id == "mtp":
            require(
                dtypes == {"float64"},
                "MTP must contain only float64",
            )
            require(
                str(model.get("device", "")).lower() == "cpu",
                "MTP must use the CPU",
            )

        if model_id == "mace_model":
            require(
                str(model.get("device", "")).lower().startswith("cuda"),
                "mace_model must use CUDA",
            )
        else:
            require(
                str(model.get("device", "")).lower() == "cpu",
                f"{model_id} must use the CPU",
            )

        if backend in {"mace", "mtp"}:
            model_path = BASE_DIR / str(model["model_path"])
            require(
                model_path.is_file(),
                f"Missing model file: {model_path}",
            )

        if model_id == "uma":
            requested_path = BASE_DIR / str(model["model_path"])
            alternate_path = BASE_DIR / (
                Path(str(model["model_path"])).stem + ".pt"
            )
            require(
                requested_path.is_file() or alternate_path.is_file(),
                (
                    f"Missing UMA model file: {requested_path} "
                    f"or {alternate_path}"
                ),
            )

        if model_id == "mtp":
            elements_path = BASE_DIR / str(
                model.get("elements_path", "pot.almtp.elements")
            )
            require(
                elements_path.is_file(),
                f"Missing MTP elements file: {elements_path}",
            )

    require(
        seen == valid_model_ids,
        (
            "The configuration must contain exactly these model IDs: "
            + ", ".join(sorted(valid_model_ids))
        ),
    )


def validate_frames(frames, expected_count):
    require(
        len(frames) == expected_count,
        (
            f"Expected {expected_count} structures, "
            f"but the XYZ contains {len(frames)}"
        ),
    )

    first_formula = frames[0].get_chemical_formula()
    first_atom_count = len(frames[0])

    require(first_atom_count > 0, "The first structure is empty")

    for index, atoms in enumerate(frames, start=1):
        require(
            len(atoms) == first_atom_count,
            (
                f"Structure {index} contains {len(atoms)} atoms; "
                f"expected {first_atom_count}"
            ),
        )

        require(
            atoms.get_chemical_formula() == first_formula,
            (
                f"Structure {index} has formula "
                f"{atoms.get_chemical_formula()}; "
                f"expected {first_formula}"
            ),
        )

        require(
            all(atoms.get_pbc()),
            f"Structure {index} does not have full periodic boundaries",
        )

        volume = float(atoms.get_volume())
        require(
            math.isfinite(volume) and volume > 0,
            f"Structure {index} has invalid volume {volume}",
        )

    return first_atom_count, first_formula


def write_frames(frames, structures_dir):
    structures_dir.mkdir(parents=True, exist_ok=True)

    frame_information = []

    for index, atoms in enumerate(frames, start=1):
        material_slug = f"licohpf_{index:03d}"
        material_label = f"LiCOHPF structure {index:03d}"
        output_path = structures_dir / f"{material_slug}.xyz"

        output_atoms = atoms.copy()
        output_atoms.info["material_slug"] = material_slug
        output_atoms.info["material_label"] = material_label
        output_atoms.info["source_frame"] = index - 1

        write(
            output_path,
            output_atoms,
            format="extxyz",
        )

        frame_information.append(
            {
                "material_slug": material_slug,
                "material_label": material_label,
                "input_path": relative_to_base(output_path),
            }
        )

    return frame_information


def epsilon_tag(epsilon):
    value = float(epsilon)
    text = f"{value:g}"

    if "e" in text.lower():
        text = f"{value:.12f}".rstrip("0").rstrip(".")

    return "eps" + text.replace(".", "")


def blank_row():
    return {column: "" for column in COLUMNS}


def attack_alpha(attack, epsilon):
    alpha = attack.get("alpha")

    if alpha is not None:
        return float(alpha)

    alpha_ratio = attack.get("alpha_ratio")
    if alpha_ratio is not None:
        return float(epsilon) * float(alpha_ratio)

    return None


def make_row(
    config,
    frame,
    model,
    dtype_str,
    attack,
    epsilon,
    n_steps,
    sweep,
):
    model_id = str(model["model_id"]).strip().lower()
    backend = str(
        model.get(
            "calculator_backend",
            model["calculator"],
        )
    ).strip().lower()

    attack_name = str(attack["name"]).strip().lower()
    attack_type = str(attack["attack_type"]).strip().lower()
    epsilon = float(epsilon)
    n_steps = int(n_steps)

    if sweep:
        run_folder = (
            f"{attack_name}_{epsilon_tag(epsilon)}"
            f"_steps{n_steps:03d}"
        )
    else:
        run_folder = f"{attack_name}_{epsilon_tag(epsilon)}"

    run_id = (
        f"{frame['material_slug']}_{model_id}_"
        f"{dtype_str}_{run_folder}"
    )

    alpha = attack_alpha(attack, epsilon)

    row = blank_row()
    row.update(
        {
            "run_id": run_id,
            "material_label": frame["material_label"],
            "material_slug": frame["material_slug"],
            "run_folder": run_folder,
            "input_path": frame["input_path"],
            "model_id": model_id,
            "display_name": model.get("display_name", model_id),
            "calculator": backend,
            "calculator_backend": backend,
            "model_path": model["model_path"],
            "attack_type": attack_type,
            "epsilon": f"{epsilon:g}",
            "n_steps": n_steps,
            "alpha": "" if alpha is None else f"{alpha:g}",
            "clip": (
                ""
                if attack.get("clip") is None
                else str(bool(attack["clip"])).lower()
            ),
            "device": model["device"],
            "dtype_str": dtype_str,
            "seed": "",
            "output_dir": "outputs",
            "mace_head": (
                ""
                if model.get("mace_head") is None
                else model["mace_head"]
            ),
            "uma_task": (
                ""
                if model.get("uma_task") is None
                else model["uma_task"]
            ),
            "uma_charge": (
                ""
                if model.get("uma_charge") is None
                else model["uma_charge"]
            ),
            "uma_spin": (
                ""
                if model.get("uma_spin") is None
                else model["uma_spin"]
            ),
            "target_energy": "",
            "relax_fmax": config.get("relax_fmax", 0.01),
            "relax_max_steps": config.get(
                "relax_max_steps",
                300,
            ),
            "relax_optimizer": config.get(
                "relax_optimizer",
                "LBFGS",
            ),
            "contour_steps": config.get("contour_steps", 500),
            "contour_maxstep": config.get(
                "contour_maxstep",
                0.01,
            ),
            "contour_parallel_drift": str(
                bool(
                    config.get(
                        "contour_parallel_drift",
                        False,
                    )
                )
            ).lower(),
            "contour_angle_limit": config.get(
                "contour_angle_limit",
                20,
            ),
            "contour_seed": config.get(
                "contour_seed",
                12345,
            ),
            "contour_energy_target": (
                ""
                if config.get("contour_energy_target") is None
                else config["contour_energy_target"]
            ),
        }
    )

    return row


def build_rows(config, frames):
    rows = []

    for frame in frames:
        for model in config["models"]:
            supported_dtypes = [
                str(value).strip().lower()
                for value in model["dtypes"]
            ]

            for dtype_str in supported_dtypes:
                for attack in config["attacks"]:
                    for epsilon in config["epsilons"]:
                        rows.append(
                            make_row(
                                config=config,
                                frame=frame,
                                model=model,
                                dtype_str=dtype_str,
                                attack=attack,
                                epsilon=epsilon,
                                n_steps=attack["n_steps"],
                                sweep=False,
                            )
                        )

                for sweep in config.get("n_step_sweeps", []):
                    epsilon = sweep["epsilon"]

                    for attack in sweep["attacks"]:
                        for n_steps in sweep["n_steps"]:
                            rows.append(
                                make_row(
                                    config=config,
                                    frame=frame,
                                    model=model,
                                    dtype_str=dtype_str,
                                    attack=attack,
                                    epsilon=epsilon,
                                    n_steps=n_steps,
                                    sweep=True,
                                )
                            )

    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=COLUMNS,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} rows to {path}")


def validate_generated_rows(rows, expected_structures):
    require(rows, "No test rows were generated")

    material_slugs = {
        row["material_slug"]
        for row in rows
    }
    require(
        len(material_slugs) == expected_structures,
        (
            f"Generated rows contain {len(material_slugs)} structures; "
            f"expected {expected_structures}"
        ),
    )

    expected_combinations = {
        ("mace_mh", "float32"),
        ("mace_mh", "float64"),
        ("uma", "float32"),
        ("uma", "float64"),
        ("mtp", "float64"),
        ("chgnet", "float32"),
        ("chgnet", "float64"),
        ("mace_model", "float32"),
        ("mace_model", "float64"),
    }

    actual_combinations = {
        (row["model_id"], row["dtype_str"])
        for row in rows
    }

    require(
        actual_combinations == expected_combinations,
        (
            "Unexpected model/dtype combinations. "
            f"Expected {sorted(expected_combinations)}, "
            f"got {sorted(actual_combinations)}"
        ),
    )

    mtp_rows = [
        row
        for row in rows
        if row["model_id"] == "mtp"
    ]
    require(mtp_rows, "No MTP rows were generated")
    require(
        all(row["dtype_str"] == "float64" for row in mtp_rows),
        "A non-float64 MTP row was generated",
    )
    require(
        all(row["device"] == "cpu" for row in mtp_rows),
        "A non-CPU MTP row was generated",
    )

    gpu_rows = [
        row
        for row in rows
        if row["model_id"] == "mace_model"
    ]
    require(gpu_rows, "No MACE_model rows were generated")
    require(
        all(
            str(row["device"]).startswith("cuda")
            for row in gpu_rows
        ),
        "A non-CUDA MACE_model row was generated",
    )

    cpu_rows = [
        row
        for row in rows
        if row["model_id"] != "mace_model"
    ]
    require(
        all(row["device"] == "cpu" for row in cpu_rows),
        "A non-CPU row was generated in the CPU workflow",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate and split 20_licohpf.xyz, then create "
            "five-model CPU/GPU test tables."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=(
            BASE_DIR
            / "datasets"
            / "licohpf_database"
            / "structures"
        ),
    )
    parser.add_argument(
        "--all-tests-out",
        type=Path,
        default=BASE_DIR / "generated_licohpf_tests.csv",
    )
    parser.add_argument(
        "--cpu-tests-out",
        type=Path,
        default=BASE_DIR / "generated_licohpf_cpu_tests.csv",
    )
    parser.add_argument(
        "--gpu-tests-out",
        type=Path,
        default=BASE_DIR / "generated_licohpf_gpu_tests.csv",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path

    config = load_config(config_path)
    validate_models(config)

    xyz_path = BASE_DIR / config["input_xyz"]
    require(
        xyz_path.is_file(),
        f"Input XYZ does not exist: {xyz_path}",
    )

    frames = read(xyz_path, index=":")
    require(
        isinstance(frames, list),
        "ASE did not return a list of XYZ structures",
    )

    expected_count = int(config["expected_structures"])
    atom_count, formula = validate_frames(
        frames,
        expected_count,
    )

    structures_dir = args.structures_dir
    if not structures_dir.is_absolute():
        structures_dir = BASE_DIR / structures_dir

    frame_information = write_frames(
        frames,
        structures_dir,
    )

    rows = build_rows(
        config,
        frame_information,
    )
    validate_generated_rows(rows, expected_count)

    cpu_rows = [
        row
        for row in rows
        if row["model_id"] != "mace_model"
    ]
    gpu_rows = [
        row
        for row in rows
        if row["model_id"] == "mace_model"
    ]

    all_tests_out = args.all_tests_out
    cpu_tests_out = args.cpu_tests_out
    gpu_tests_out = args.gpu_tests_out

    if not all_tests_out.is_absolute():
        all_tests_out = BASE_DIR / all_tests_out
    if not cpu_tests_out.is_absolute():
        cpu_tests_out = BASE_DIR / cpu_tests_out
    if not gpu_tests_out.is_absolute():
        gpu_tests_out = BASE_DIR / gpu_tests_out

    write_csv(all_tests_out, rows)
    write_csv(cpu_tests_out, cpu_rows)
    write_csv(gpu_tests_out, gpu_rows)

    print()
    print("LiCOHPF setup passed")
    print(f"Structures: {len(frames)}")
    print(f"Atoms per structure: {atom_count}")
    print(f"Formula: {formula}")
    print(f"All rows: {len(rows):,}")
    print(f"CPU rows: {len(cpu_rows):,}")
    print(f"GPU rows: {len(gpu_rows):,}")
    print()
    print("Model/dtype row counts:")

    for model_id in [
        "mace_mh",
        "uma",
        "mtp",
        "chgnet",
        "mace_model",
    ]:
        for dtype_str in ["float32", "float64"]:
            count = sum(
                1
                for row in rows
                if row["model_id"] == model_id
                and row["dtype_str"] == dtype_str
            )
            if count:
                print(
                    f"  {model_id:12s} "
                    f"{dtype_str:7s}: {count:,}"
                )


if __name__ == "__main__":
    main()
