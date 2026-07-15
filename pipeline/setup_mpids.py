#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import importlib.util
import json
import os
import re


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_BACKENDS = {
    "mace_mh": "mace",
    "uma": "uma",
    "mtp": "mtp",
    "chgnet": "chgnet",
    "mace_model": "mace",
}

EXPECTED_MODELS = set(MODEL_BACKENDS)

CPU_MODELS = {
    "mace_mh",
    "uma",
    "mtp",
    "chgnet",
}

GPU_MODELS = {
    "mace_model",
}

VALID_DTYPES = {
    "float32",
    "float64",
}

CHGNET_MODELS = {
    "chgnet-0.2.0",
    "chgnet-0.3.0",
    "chgnet-r2scan",
}

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
    "elements_path",
    "dtype_str",
    "attack_type",
    "epsilon",
    "n_steps",
    "alpha",
    "clip",
    "device",
    "output_dir",
    "seed",
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


def require_package(module_name, package_name=None):
    if importlib.util.find_spec(module_name) is None:
        package_name = package_name or module_name
        raise SystemExit(
            f"ERROR: Missing Python package {package_name!r}. "
            f"Install it in the environment used by setup.sh."
        )


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def slug(value):
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def epsilon_tag(epsilon):
    value = float(epsilon)
    text = f"{value:g}"

    if "e" in text.lower():
        text = f"{value:.10f}".rstrip("0").rstrip(".")

    return "eps" + text.replace(".", "")


def blank_row():
    return {
        column: ""
        for column in COLUMNS
    }


def read_materials(path):
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    require(rows, f"No materials were found in {path}")

    seen_mpids = set()
    seen_slugs = set()

    for row_number, row in enumerate(rows, start=2):
        material_label = clean_text(
            row.get("material_label")
        )
        mpid = clean_text(row.get("mpid"))

        require(
            material_label,
            f"{path}:{row_number} is missing material_label",
        )
        require(
            mpid,
            f"{path}:{row_number} is missing mpid",
        )

        material_slug = slug(material_label)

        require(
            material_slug,
            f"{path}:{row_number} has an invalid material_label",
        )
        require(
            mpid not in seen_mpids,
            f"Duplicate mpid in {path}: {mpid}",
        )
        require(
            material_slug not in seen_slugs,
            f"Duplicate material slug in {path}: "
            f"{material_slug}",
        )

        seen_mpids.add(mpid)
        seen_slugs.add(material_slug)

    return rows


def load_config(path):
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = json.load(handle)

    for key in (
        "models",
        "attacks",
        "epsilons",
    ):
        require(
            key in config,
            f"{path} must define {key!r}",
        )
        require(
            config[key],
            f"{path} must define a non-empty {key!r}",
        )

    return config


def normalize_dtypes(model):
    dtypes = model.get("dtypes")

    require(
        isinstance(dtypes, list) and dtypes,
        f"Model {model.get('model_id')!r} must define "
        "a non-empty dtypes list",
    )

    normalized = [
        clean_text(dtype).lower()
        for dtype in dtypes
    ]

    require(
        len(normalized) == len(set(normalized)),
        f"Model {model.get('model_id')!r} contains "
        "duplicate dtypes",
    )

    invalid = set(normalized).difference(
        VALID_DTYPES
    )

    require(
        not invalid,
        f"Model {model.get('model_id')!r} has invalid "
        f"dtypes: {sorted(invalid)}",
    )

    return normalized


def validate_local_model_file(model_path, model_id):
    path = BASE_DIR / model_path

    require(
        path.is_file(),
        f"Missing model file for {model_id}: {path}",
    )


def validate_models(config):
    models = config["models"]
    seen = set()

    for index, model in enumerate(models, start=1):
        require(
            isinstance(model, dict),
            f"Model entry {index} must be an object",
        )

        model_id = clean_text(
            model.get("model_id")
        ).lower()

        require(
            model_id in EXPECTED_MODELS,
            f"Invalid model_id {model_id!r}. Expected one "
            "of: "
            + ", ".join(sorted(EXPECTED_MODELS)),
        )
        require(
            model_id not in seen,
            f"Duplicate model_id: {model_id}",
        )

        backend = clean_text(
            model.get(
                "calculator_backend",
                model.get("calculator"),
            )
        ).lower()

        expected_backend = MODEL_BACKENDS[model_id]

        require(
            backend == expected_backend,
            f"{model_id} requires calculator_backend="
            f"{expected_backend!r}, not {backend!r}",
        )

        model_path = clean_text(
            model.get("model_path")
        )

        require(
            model_path,
            f"{model_id} is missing model_path",
        )

        device = clean_text(
            model.get("device")
        ).lower()

        dtypes = normalize_dtypes(model)

        if model_id == "mtp":
            require(
                dtypes == ["float64"],
                "MTP must define exactly "
                '\"dtypes\": [\"float64\"]',
            )
            require(
                device == "cpu",
                "MTP must use device=cpu",
            )

        elif model_id == "mace_model":
            require(
                device == "cuda",
                "mace_model must use device=cuda",
            )
            require(
                set(dtypes) == {
                    "float32",
                    "float64",
                },
                "mace_model must define float32 and "
                "float64",
            )

        else:
            require(
                device == "cpu",
                f"{model_id} must use device=cpu",
            )
            require(
                set(dtypes) == {
                    "float32",
                    "float64",
                },
                f"{model_id} must define float32 and "
                "float64",
            )

        if model_id in {
            "mace_mh",
            "mace_model",
            "mtp",
        }:
            validate_local_model_file(
                model_path,
                model_id,
            )

        elif model_id == "uma":
            path = BASE_DIR / model_path
            alternate = BASE_DIR / (
                Path(model_path).stem + ".pt"
            )

            require(
                path.is_file() or alternate.is_file(),
                f"Missing UMA model file: {path}",
            )

        elif model_id == "chgnet":
            require(
                model_path in CHGNET_MODELS,
                f"Unsupported CHGNet model: "
                f"{model_path}",
            )

        if model_id == "mtp":
            elements_path = clean_text(
                model.get(
                    "elements_path",
                    "pot.almtp.elements",
                )
            )

            require(
                elements_path,
                "MTP is missing elements_path",
            )
            require(
                (BASE_DIR / elements_path).is_file(),
                "Missing MTP elements file: "
                f"{BASE_DIR / elements_path}",
            )

        seen.add(model_id)

    require(
        seen == EXPECTED_MODELS,
        "The configuration must contain exactly these "
        "five models: "
        + ", ".join(sorted(EXPECTED_MODELS)),
    )


def validate_attacks(config):
    attacks = config["attacks"]

    for index, attack in enumerate(attacks, start=1):
        require(
            isinstance(attack, dict),
            f"Attack entry {index} must be an object",
        )

        name = clean_text(
            attack.get("name")
        ).lower()
        attack_type = clean_text(
            attack.get("attack_type")
        ).lower()

        require(
            name,
            f"Attack entry {index} is missing name",
        )
        require(
            attack_type in {"fgsm", "pgd"},
            f"Attack {name!r} must use fgsm or pgd",
        )

        n_steps = int(attack.get("n_steps", 0))

        require(
            n_steps > 0,
            f"Attack {name!r} must have n_steps > 0",
        )

    epsilons = [
        float(value)
        for value in config["epsilons"]
    ]

    require(
        all(value > 0 for value in epsilons),
        "All epsilon values must be greater than zero",
    )
    require(
        len(epsilons) == len(set(epsilons)),
        "The epsilon list contains duplicates",
    )

    for sweep_index, sweep in enumerate(
        config.get("n_step_sweeps", []),
        start=1,
    ):
        epsilon = float(sweep.get("epsilon", 0))

        require(
            epsilon > 0,
            f"n_step_sweeps entry {sweep_index} has "
            "an invalid epsilon",
        )
        require(
            sweep.get("attacks"),
            f"n_step_sweeps entry {sweep_index} is "
            "missing attacks",
        )
        require(
            sweep.get("n_steps"),
            f"n_step_sweeps entry {sweep_index} is "
            "missing n_steps",
        )

        steps = [
            int(value)
            for value in sweep["n_steps"]
        ]

        require(
            all(value > 0 for value in steps),
            "All n-step sweep values must be greater "
            "than zero",
        )


def validate_config(config):
    validate_models(config)
    validate_attacks(config)


def download_structures(
    materials,
    structures_dir,
    force=False,
):
    require(
        os.environ.get("MP_API_KEY"),
        "Set MP_API_KEY in .env before running setup.sh",
    )

    require_package("mp_api", "mp-api")
    require_package("pymatgen", "pymatgen")

    from mp_api.client import MPRester

    structures_dir = Path(structures_dir)
    structures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = {}

    with MPRester(os.environ["MP_API_KEY"]) as mpr:
        for material in materials:
            mpid = clean_text(material["mpid"])
            material_slug = slug(
                material["material_label"]
            )
            cif_path = (
                structures_dir
                / f"{mpid}_{material_slug}.cif"
            )

            if cif_path.is_file() and not force:
                print(
                    f"Using existing {cif_path}",
                    flush=True,
                )
                output_paths[mpid] = cif_path
                continue

            print(
                f"Downloading {mpid} -> {cif_path}",
                flush=True,
            )

            structure = (
                mpr.get_structure_by_material_id(mpid)
            )

            require(
                structure is not None,
                f"Materials Project returned no structure "
                f"for {mpid}",
            )

            structure.to(filename=str(cif_path))

            require(
                cif_path.is_file(),
                f"Failed to write {cif_path}",
            )

            output_paths[mpid] = cif_path

    return output_paths


def optional_number(value):
    if value is None:
        return ""
    if clean_text(value) == "":
        return ""
    return f"{float(value):g}"


def optional_integer(value):
    if value is None:
        return ""
    if clean_text(value) == "":
        return ""
    return str(int(value))


def optional_boolean(value):
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    text = clean_text(value).lower()

    if text in {"true", "1", "yes", "y"}:
        return "true"

    if text in {"false", "0", "no", "n"}:
        return "false"

    raise SystemExit(
        f"ERROR: Invalid Boolean value: {value!r}"
    )


def attack_alpha(attack, epsilon):
    alpha = attack.get("alpha")

    if alpha is not None:
        return optional_number(alpha)

    alpha_ratio = attack.get("alpha_ratio")

    if alpha_ratio is not None:
        return optional_number(
            float(epsilon) * float(alpha_ratio)
        )

    return ""


def add_row(
    rows,
    config,
    material_label,
    material_slug,
    input_path,
    model,
    dtype_str,
    attack,
    epsilon,
    n_steps=None,
    sweep=False,
):
    model_id = clean_text(
        model["model_id"]
    ).lower()
    backend = MODEL_BACKENDS[model_id]
    attack_name = clean_text(
        attack["name"]
    ).lower()
    attack_type = clean_text(
        attack["attack_type"]
    ).lower()

    if n_steps is None:
        n_steps = int(attack["n_steps"])
    else:
        n_steps = int(n_steps)

    epsilon = float(epsilon)

    if sweep:
        run_folder = (
            f"{attack_name}_"
            f"{epsilon_tag(epsilon)}_"
            f"steps{n_steps:03d}"
        )
    else:
        run_folder = (
            f"{attack_name}_"
            f"{epsilon_tag(epsilon)}"
        )

    row = blank_row()

    row["run_id"] = (
        f"{material_slug}_"
        f"{model_id}_"
        f"{dtype_str}_"
        f"{run_folder}"
    )
    row["material_label"] = material_label
    row["material_slug"] = material_slug
    row["run_folder"] = run_folder
    row["input_path"] = str(
        input_path
    ).replace("\\", "/")
    row["model_id"] = model_id
    row["display_name"] = clean_text(
        model.get("display_name", model_id)
    )
    row["calculator"] = backend
    row["calculator_backend"] = backend
    row["model_path"] = clean_text(
        model["model_path"]
    )
    row["dtype_str"] = dtype_str
    row["attack_type"] = attack_type
    row["epsilon"] = f"{epsilon:g}"
    row["n_steps"] = str(n_steps)
    row["alpha"] = attack_alpha(
        attack,
        epsilon,
    )
    row["clip"] = optional_boolean(
        attack.get("clip")
    )
    row["device"] = clean_text(
        model["device"]
    ).lower()
    row["output_dir"] = "outputs"
    row["seed"] = optional_integer(
        config.get("seed")
    )
    row["target_energy"] = optional_number(
        attack.get(
            "target_energy",
            config.get("target_energy"),
        )
    )
    row["relax_fmax"] = optional_number(
        config.get("relax_fmax", 0.01)
    )
    row["relax_max_steps"] = optional_integer(
        config.get("relax_max_steps", 300)
    )
    row["relax_optimizer"] = clean_text(
        config.get(
            "relax_optimizer",
            "LBFGS",
        )
    ).upper()
    row["contour_steps"] = optional_integer(
        config.get("contour_steps", 500)
    )
    row["contour_maxstep"] = optional_number(
        config.get("contour_maxstep", 0.01)
    )
    row["contour_parallel_drift"] = (
        optional_boolean(
            config.get(
                "contour_parallel_drift",
                False,
            )
        )
    )
    row["contour_angle_limit"] = optional_number(
        config.get("contour_angle_limit", 20)
    )
    row["contour_seed"] = optional_integer(
        config.get("contour_seed", 12345)
    )
    row["contour_energy_target"] = (
        optional_number(
            config.get("contour_energy_target")
        )
    )

    if backend == "mace":
        row["mace_head"] = clean_text(
            model.get("mace_head")
        )

    if model_id == "uma":
        row["uma_task"] = clean_text(
            model.get("uma_task", "omat")
        )
        row["uma_charge"] = optional_integer(
            model.get("uma_charge")
        )
        row["uma_spin"] = optional_integer(
            model.get("uma_spin")
        )

    if model_id == "mtp":
        row["elements_path"] = clean_text(
            model.get(
                "elements_path",
                "pot.almtp.elements",
            )
        )

    rows.append(row)


def build_rows(
    materials,
    structure_paths,
    config,
):
    validate_config(config)

    rows = []

    for material in materials:
        material_label = clean_text(
            material["material_label"]
        )
        material_slug = slug(material_label)
        mpid = clean_text(material["mpid"])

        structure_path = structure_paths[mpid]
        input_path = structure_path.relative_to(
            BASE_DIR
        )

        for model in config["models"]:
            dtypes = normalize_dtypes(model)

            for dtype_str in dtypes:
                for attack in config["attacks"]:
                    for epsilon in config["epsilons"]:
                        add_row(
                            rows=rows,
                            config=config,
                            material_label=material_label,
                            material_slug=material_slug,
                            input_path=input_path,
                            model=model,
                            dtype_str=dtype_str,
                            attack=attack,
                            epsilon=epsilon,
                            sweep=False,
                        )

                for sweep in config.get(
                    "n_step_sweeps",
                    [],
                ):
                    epsilon = sweep["epsilon"]

                    for attack in sweep["attacks"]:
                        for n_steps in sweep["n_steps"]:
                            add_row(
                                rows=rows,
                                config=config,
                                material_label=(
                                    material_label
                                ),
                                material_slug=(
                                    material_slug
                                ),
                                input_path=input_path,
                                model=model,
                                dtype_str=dtype_str,
                                attack=attack,
                                epsilon=epsilon,
                                n_steps=n_steps,
                                sweep=True,
                            )

    return rows


def validate_generated_rows(rows):
    require(
        rows,
        "No test rows were generated",
    )

    run_ids = [
        row["run_id"]
        for row in rows
    ]

    require(
        len(run_ids) == len(set(run_ids)),
        "Duplicate run_id values were generated",
    )

    generated_models = {
        row["model_id"]
        for row in rows
    }

    require(
        generated_models == EXPECTED_MODELS,
        "Generated model set does not match the "
        "required five models",
    )

    for row in rows:
        model_id = row["model_id"]
        dtype_str = row["dtype_str"]
        device = row["device"]
        backend = row["calculator_backend"]

        require(
            backend == MODEL_BACKENDS[model_id],
            f"Invalid backend for {model_id}",
        )

        if model_id == "mtp":
            require(
                dtype_str == "float64",
                "Generated an invalid float32 MTP row",
            )
            require(
                device == "cpu",
                "Generated a non-CPU MTP row",
            )

        elif model_id == "mace_model":
            require(
                device == "cuda",
                "Generated a non-CUDA mace_model row",
            )

        else:
            require(
                device == "cpu",
                f"Generated a non-CPU {model_id} row",
            )


def write_tests(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows):,} rows to "
        f"{output_path}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--materials",
        default=(
            "datasets/2d_structures/"
            "tests_materials.csv"
        ),
    )
    parser.add_argument(
        "--config",
        default=(
            "datasets/2d_structures/"
            "tests_comprehensive.json"
        ),
    )
    parser.add_argument(
        "--tests-out",
        default="generated_material_tests.csv",
    )
    parser.add_argument(
        "--structures-dir",
        default="mp_structures",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
    )

    args = parser.parse_args()

    materials_path = (
        BASE_DIR / args.materials
    ).resolve()
    config_path = (
        BASE_DIR / args.config
    ).resolve()
    tests_out = (
        BASE_DIR / args.tests_out
    ).resolve()
    structures_dir = (
        BASE_DIR / args.structures_dir
    ).resolve()

    require(
        materials_path.is_file(),
        f"Missing materials file: {materials_path}",
    )
    require(
        config_path.is_file(),
        f"Missing configuration file: {config_path}",
    )

    materials = read_materials(materials_path)
    config = load_config(config_path)

    validate_config(config)

    structure_paths = download_structures(
        materials=materials,
        structures_dir=structures_dir,
        force=args.force_download,
    )

    if args.download_only:
        print(
            "Download-only mode complete.",
            flush=True,
        )
        return

    rows = build_rows(
        materials=materials,
        structure_paths=structure_paths,
        config=config,
    )

    validate_generated_rows(rows)
    write_tests(rows, tests_out)


if __name__ == "__main__":
    main()
