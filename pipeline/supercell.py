#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import json
import re
import shlex

from ase.io import read, write


BASE_DIR = Path(__file__).resolve().parent.parent

REPEAT_TUPLES = [
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 1),
    (1, 2, 2),
    (2, 1, 1),
    (2, 1, 2),
    (2, 2, 1),
    (2, 2, 2),
]

MODEL_ORDER = [
    "mace_mh",
    "uma",
    "mtp",
    "chgnet",
    "mace_model",
]

MODEL_BACKENDS = {
    "mace_mh": "mace",
    "uma": "uma",
    "mtp": "mtp",
    "chgnet": "chgnet",
    "mace_model": "mace",
}

EPSILON = 0.01
DTYPE = "float64"
SEED = 42

BASE_COLUMNS = [
    "run_id",
    "material_label",
    "material_slug",
    "run_folder",
    "input_path",
    "model_path",
    "elements_path",
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

EXTRA_COLUMNS = [
    "model_id",
    "display_name",
    "calculator",
    "calculator_backend",
    "dtype_str",
    "seed",
    "base_material_slug",
    "base_material_label",
    "base_input_path",
    "supercell_repeat_x",
    "supercell_repeat_y",
    "supercell_repeat_z",
    "supercell_repeat_tuple",
    "unit_cell_atoms",
    "supercell_atoms",
]

COLUMNS = BASE_COLUMNS + EXTRA_COLUMNS

METADATA_COLUMNS = [
    "base_material_slug",
    "base_material_label",
    "supercell_material_slug",
    "base_input_path",
    "supercell_input_path",
    "repeat_x",
    "repeat_y",
    "repeat_z",
    "repeat_tuple",
    "unit_cell_atoms",
    "supercell_atoms",
]


def require(condition, message):
    if not condition:
        raise SystemExit(f"ERROR: {message}")


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def slug(value):
    value = clean(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def epsilon_tag(epsilon):
    text = f"{float(epsilon):g}"

    if "e" in text.lower():
        text = (
            f"{float(epsilon):.10f}"
            .rstrip("0")
            .rstrip(".")
        )

    return "eps" + text.replace(".", "")


def resolve_from_base(path):
    path = Path(path)

    if path.is_absolute():
        return path.resolve()

    return (BASE_DIR / path).resolve()


def read_csv_rows(path):
    path = Path(path)

    require(
        path.is_file(),
        f"Missing CSV file: {path}",
    )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_config(path):
    path = Path(path)

    require(
        path.is_file(),
        f"Missing configuration file: {path}",
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def blank_row():
    return {
        column: ""
        for column in COLUMNS
    }


def discover_materials(material_rows):
    materials = []
    seen = set()

    for row in material_rows:
        material_slug = clean(
            row.get("material_slug")
        )

        if not material_slug:
            input_path = clean(
                row.get("input_path")
            )
            material_slug = slug(
                Path(input_path).stem
            )

        require(
            material_slug,
            f"Cannot determine material slug from {row}",
        )

        if material_slug in seen:
            continue

        input_path = clean(
            row.get("input_path")
        )

        require(
            input_path,
            f"{material_slug} is missing input_path",
        )

        materials.append({
            "material_slug": material_slug,
            "material_label": clean(
                row.get(
                    "material_label",
                    material_slug,
                )
            ) or material_slug,
            "input_path": input_path,
        })

        seen.add(material_slug)

    require(
        len(materials) == 20,
        "Expected exactly 20 unique materials, "
        f"found {len(materials)}",
    )

    return materials


def structure_path(material, structures_dir):
    input_path = clean(
        material.get("input_path")
    )

    if input_path:
        return resolve_from_base(input_path)

    material_slug = clean(
        material.get("material_slug")
    )

    require(
        material_slug,
        f"Cannot determine structure path for {material}",
    )

    return (
        Path(structures_dir)
        / f"{material_slug}.xyz"
    ).resolve()


def model_map(config):
    models = {}

    for model in config.get("models", []):
        model_id = clean(
            model.get("model_id")
        ).lower()

        require(
            model_id in MODEL_BACKENDS,
            f"Unknown model_id: {model_id!r}",
        )
        require(
            model_id not in models,
            f"Duplicate model_id: {model_id}",
        )

        backend = clean(
            model.get(
                "calculator_backend",
                model.get("calculator"),
            )
        ).lower()

        require(
            backend == MODEL_BACKENDS[model_id],
            f"{model_id} requires backend "
            f"{MODEL_BACKENDS[model_id]!r}, "
            f"not {backend!r}",
        )

        supported_dtypes = {
            clean(value).lower()
            for value in model.get("dtypes", [])
        }

        require(
            DTYPE in supported_dtypes,
            f"{model_id} does not support {DTYPE}",
        )

        models[model_id] = model

    require(
        set(models) == set(MODEL_ORDER),
        "Configuration must contain exactly: "
        + ", ".join(MODEL_ORDER),
    )

    return models


def attack_alpha(attack):
    alpha = attack.get("alpha")

    if alpha is not None:
        return f"{float(alpha):g}"

    alpha_ratio = attack.get("alpha_ratio")

    if alpha_ratio is not None:
        return (
            f"{EPSILON * float(alpha_ratio):g}"
        )

    return ""


def make_run_row(
    config,
    context,
    supercell,
    model,
    attack,
):
    model_id = clean(
        model["model_id"]
    ).lower()
    backend = MODEL_BACKENDS[model_id]

    attack_name = clean(
        attack["name"]
    ).lower()
    attack_type = clean(
        attack["attack_type"]
    ).lower()
    n_steps = int(attack["n_steps"])

    require(
        attack_type in {"fgsm", "pgd"},
        f"Invalid attack type: {attack_type}",
    )
    require(
        n_steps > 0,
        f"Invalid n_steps for {attack_name}",
    )

    run_folder = (
        f"{attack_name}_{epsilon_tag(EPSILON)}"
    )

    row = blank_row()

    row.update({
        "run_id": (
            f"{supercell['material_slug']}_"
            f"{model_id}_{DTYPE}_{run_folder}"
        ),
        "material_label": (
            supercell["material_label"]
        ),
        "material_slug": (
            supercell["material_slug"]
        ),
        "run_folder": run_folder,
        "input_path": supercell["input_path"],
        "model_path": clean(
            model["model_path"]
        ),
        "elements_path": clean(
            model.get("elements_path")
        ),
        "attack_type": attack_type,
        "epsilon": f"{EPSILON:g}",
        "n_steps": str(n_steps),
        "alpha": attack_alpha(attack),
        "clip": (
            ""
            if attack.get("clip") is None
            else str(attack["clip"]).lower()
        ),
        "device": clean(
            model["device"]
        ).lower(),
        "output_dir": "outputs",
        "mace_head": clean(
            model.get("mace_head")
        ),
        "uma_task": clean(
            model.get("uma_task")
        ),
        "uma_charge": clean(
            model.get("uma_charge")
        ),
        "uma_spin": clean(
            model.get("uma_spin")
        ),
        "target_energy": clean(
            config.get("target_energy")
        ),
        "relax_fmax": clean(
            config.get("relax_fmax", 0.01)
        ),
        "relax_max_steps": clean(
            config.get("relax_max_steps", 300)
        ),
        "relax_optimizer": clean(
            config.get(
                "relax_optimizer",
                "LBFGS",
            )
        ).upper(),
        "contour_steps": clean(
            config.get("contour_steps", 500)
        ),
        "contour_maxstep": clean(
            config.get("contour_maxstep", 0.01)
        ),
        "contour_parallel_drift": clean(
            config.get(
                "contour_parallel_drift",
                False,
            )
        ).lower(),
        "contour_angle_limit": clean(
            config.get("contour_angle_limit", 20)
        ),
        "contour_seed": clean(
            config.get("contour_seed", 12345)
        ),
        "contour_energy_target": clean(
            config.get("contour_energy_target")
        ),
        "model_id": model_id,
        "display_name": clean(
            model.get(
                "display_name",
                model_id,
            )
        ),
        "calculator": backend,
        "calculator_backend": backend,
        "dtype_str": DTYPE,
        "seed": str(SEED),
        "base_material_slug": (
            context["base_material_slug"]
        ),
        "base_material_label": (
            context["base_material_label"]
        ),
        "base_input_path": (
            context["base_input_path"]
        ),
        "supercell_repeat_x": (
            supercell["repeat_x"]
        ),
        "supercell_repeat_y": (
            supercell["repeat_y"]
        ),
        "supercell_repeat_z": (
            supercell["repeat_z"]
        ),
        "supercell_repeat_tuple": (
            supercell["repeat_tuple"]
        ),
        "unit_cell_atoms": (
            supercell["unit_cell_atoms"]
        ),
        "supercell_atoms": (
            supercell["supercell_atoms"]
        ),
    })

    return row


def generate(args):
    output_root = Path(
        args.output_root
    ).resolve()
    materials_path = resolve_from_base(
        args.materials
    )
    config_path = resolve_from_base(
        args.config
    )
    structures_dir = resolve_from_base(
        args.structures_dir
    )

    config = load_config(config_path)
    models = model_map(config)

    require(
        config.get("attacks"),
        "Configuration contains no attacks",
    )

    material_rows = read_csv_rows(
        materials_path
    )
    materials = discover_materials(
        material_rows
    )

    structures_root = (
        output_root / "supercell_structures"
    )
    structures_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    tests = []
    metadata = []

    for material in materials:
        base_slug = material["material_slug"]
        base_label = material["material_label"]

        base_path = structure_path(
            material,
            structures_dir,
        )

        require(
            base_path.is_file(),
            f"Missing base structure: {base_path}",
        )

        atoms = read(base_path)
        unit_cell_atoms = len(atoms)

        require(
            unit_cell_atoms > 0,
            f"Structure contains no atoms: {base_path}",
        )

        context = {
            "base_material_slug": base_slug,
            "base_material_label": base_label,
            "base_input_path": str(base_path),
        }

        for repeat_x, repeat_y, repeat_z in (
            REPEAT_TUPLES
        ):
            repeat_label = (
                f"{repeat_x}x"
                f"{repeat_y}x"
                f"{repeat_z}"
            )
            supercell_slug = (
                f"{base_slug}_r{repeat_label}"
            )

            cif_path = (
                structures_root
                / base_slug
                / f"{supercell_slug}.cif"
            )
            cif_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if args.force or not cif_path.is_file():
                supercell_atoms_object = atoms.repeat(
                    (
                        repeat_x,
                        repeat_y,
                        repeat_z,
                    )
                )
                write(
                    cif_path,
                    supercell_atoms_object,
                )

            require(
                cif_path.is_file(),
                f"Failed to create {cif_path}",
            )

            atom_count = (
                unit_cell_atoms
                * repeat_x
                * repeat_y
                * repeat_z
            )

            supercell = {
                "material_label": (
                    f"{base_label}_r{repeat_label}"
                ),
                "material_slug": supercell_slug,
                "input_path": str(
                    cif_path.resolve()
                ),
                "repeat_x": repeat_x,
                "repeat_y": repeat_y,
                "repeat_z": repeat_z,
                "repeat_tuple": repeat_label,
                "unit_cell_atoms": unit_cell_atoms,
                "supercell_atoms": atom_count,
            }

            metadata.append({
                "base_material_slug": base_slug,
                "base_material_label": base_label,
                "supercell_material_slug": (
                    supercell_slug
                ),
                "base_input_path": str(base_path),
                "supercell_input_path": str(
                    cif_path.resolve()
                ),
                "repeat_x": repeat_x,
                "repeat_y": repeat_y,
                "repeat_z": repeat_z,
                "repeat_tuple": repeat_label,
                "unit_cell_atoms": unit_cell_atoms,
                "supercell_atoms": atom_count,
            })

            for model_id in MODEL_ORDER:
                model = models[model_id]

                for attack in config["attacks"]:
                    tests.append(
                        make_run_row(
                            config=config,
                            context=context,
                            supercell=supercell,
                            model=model,
                            attack=attack,
                        )
                    )

    expected_metadata = (
        len(materials)
        * len(REPEAT_TUPLES)
    )
    expected_tests = (
        expected_metadata
        * len(MODEL_ORDER)
        * len(config["attacks"])
    )

    require(
        len(metadata) == expected_metadata,
        f"Expected {expected_metadata} metadata "
        f"rows, generated {len(metadata)}",
    )
    require(
        len(tests) == expected_tests,
        f"Expected {expected_tests} test rows, "
        f"generated {len(tests)}",
    )

    run_ids = [
        row["run_id"]
        for row in tests
    ]

    require(
        len(run_ids) == len(set(run_ids)),
        "Duplicate supercell run_id values generated",
    )

    write_csv_rows(
        output_root
        / "generated_supercell_tests.csv",
        tests,
        COLUMNS,
    )
    write_csv_rows(
        output_root
        / "supercell_metadata.csv",
        metadata,
        METADATA_COLUMNS,
    )

    print(
        f"Wrote {len(tests):,} rows to "
        f"{output_root / 'generated_supercell_tests.csv'}"
    )
    print(
        f"Wrote {len(metadata):,} rows to "
        f"{output_root / 'supercell_metadata.csv'}"
    )


def build_task_list(output_root):
    metadata_path = (
        output_root / "supercell_metadata.csv"
    )
    metadata = read_csv_rows(metadata_path)

    require(
        len(metadata) == 160,
        "Expected 160 supercell metadata rows, "
        f"found {len(metadata)}",
    )

    tasks = []

    for row in metadata:
        base_material_slug = clean(
            row["base_material_slug"]
        )
        repeat_label = clean(
            row["repeat_tuple"]
        )

        for model_id in MODEL_ORDER:
            tasks.append(
                (
                    base_material_slug,
                    repeat_label,
                    model_id,
                )
            )

    require(
        len(tasks) == 800,
        f"Expected 800 supercell tasks, "
        f"generated {len(tasks)}",
    )

    return tasks


def shell_assignment(name, value):
    print(
        f"{name}={shlex.quote(str(value))}"
    )


def task_info(args):
    output_root = Path(
        args.output_root
    ).resolve()
    task_id = int(args.task_id)
    tasks = build_task_list(output_root)

    require(
        1 <= task_id <= len(tasks),
        f"task-id must be 1..{len(tasks)}, "
        f"got {task_id}",
    )

    material, repeat_label, model_id = (
        tasks[task_id - 1]
    )

    tests_path = (
        output_root
        / "generated_supercell_tests.csv"
    )
    rows = read_csv_rows(tests_path)

    selected = [
        row
        for row in rows
        if (
            clean(row["base_material_slug"])
            == material
            and clean(
                row["supercell_repeat_tuple"]
            )
            == repeat_label
            and clean(row["model_id"])
            == model_id
        )
    ]

    require(
        selected,
        f"No rows found for {material}, "
        f"{repeat_label}, {model_id}",
    )

    expected_attacks = {
        clean(row["run_folder"])
        for row in selected
    }

    require(
        len(expected_attacks) == 3,
        f"Expected three attacks for {material}, "
        f"{repeat_label}, {model_id}; "
        f"found {len(expected_attacks)}",
    )

    selected_path = (
        output_root
        / "material_tests"
        / DTYPE
        / (
            f"{model_id}_{material}_"
            f"r{repeat_label}.csv"
        )
    )

    write_csv_rows(
        selected_path,
        selected,
        COLUMNS,
    )

    summary_path = (
        output_root
        / "array_summaries"
        / (
            f"{DTYPE}_{model_id}_{material}_"
            f"r{repeat_label}_summary.csv"
        )
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shell_assignment("MATERIAL", material)
    shell_assignment("REPEAT", repeat_label)
    shell_assignment("MODEL_ID", model_id)
    shell_assignment(
        "CALCULATOR_BACKEND",
        MODEL_BACKENDS[model_id],
    )
    shell_assignment(
        "DEVICE",
        selected[0]["device"],
    )
    shell_assignment(
        "TEST_CSV",
        selected_path,
    )
    shell_assignment(
        "SUMMARY_FILE",
        summary_path,
    )


def combine(args):
    """
    Combine every available supercell summary.

    Missing materials, repeats, models and rows generate warnings
    instead of preventing the available data from being combined.
    """
    output_root = Path(
        args.output_root
    ).resolve()

    summary_dir = (
        output_root / "array_summaries"
    )

    if not summary_dir.is_dir():
        print(
            "WARNING: missing supercell summary "
            f"directory: {summary_dir}"
        )
        return

    total_models = 0
    total_rows = 0

    for model_id in MODEL_ORDER:
        summary_paths = sorted(
            summary_dir.glob(
                f"{DTYPE}_{model_id}_"
                "*_summary.csv"
            )
        )

        if not summary_paths:
            print(
                f"WARNING: no supercell summary "
                f"files for {model_id}"
            )
            continue

        combined_rows = []
        columns = []

        for summary_path in summary_paths:
            try:
                rows = read_csv_rows(
                    summary_path
                )
            except BaseException as error:
                print(
                    f"WARNING: could not read "
                    f"{summary_path}: {error}"
                )
                continue

            if not rows:
                print(
                    f"WARNING: empty supercell "
                    f"summary: {summary_path}"
                )
                continue

            for row in rows:
                for column in row:
                    if column not in columns:
                        columns.append(column)

                row["calculator"] = model_id
                row["model_id"] = model_id
                combined_rows.append(row)

        if not combined_rows:
            print(
                f"WARNING: no usable supercell "
                f"rows for {model_id}"
            )
            continue

        for required_column in [
            "calculator",
            "model_id",
        ]:
            if required_column not in columns:
                columns.append(
                    required_column
                )

        # If the command is rerun, retain only the newest copy of
        # each run rather than duplicating rows.
        if "run_id" in columns:
            rows_by_run_id = {}

            for row_number, row in enumerate(
                combined_rows
            ):
                run_id = clean(
                    row.get("run_id")
                )

                key = (
                    run_id
                    if run_id
                    else f"missing_run_id_{row_number}"
                )

                rows_by_run_id[key] = row

            combined_rows = list(
                rows_by_run_id.values()
            )

        output_path = (
            output_root
            / f"outputs_{DTYPE}"
            / model_id
            / "summary.csv"
        )

        write_csv_rows(
            output_path,
            combined_rows,
            columns,
        )

        total_models += 1
        total_rows += len(
            combined_rows
        )

        print(
            f"Wrote {len(combined_rows):,} "
            f"available {model_id} rows to "
            f"{output_path}"
        )

        if (
            len(summary_paths) < 160
            or len(combined_rows) < 480
        ):
            print(
                f"WARNING: {model_id} supercell "
                "results are partial: "
                f"{len(summary_paths)}/160 summary "
                "files and "
                f"{len(combined_rows)}/480 rows"
            )

    print(
        "Combined available supercell data: "
        f"{total_models} models and "
        f"{total_rows:,} rows"
    )


def main():
    parser = argparse.ArgumentParser()

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    generate_parser = commands.add_parser(
        "generate"
    )
    generate_parser.add_argument(
        "--output-root",
        required=True,
    )
    generate_parser.add_argument(
        "--materials",
        default="generated_material_tests.csv",
    )
    generate_parser.add_argument(
        "--config",
        default=(
            "datasets/2d_structures/"
            "tests_comprehensive.json"
        ),
    )
    generate_parser.add_argument(
        "--structures-dir",
        default="mp_structures",
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
    )

    task_parser = commands.add_parser(
        "task-info"
    )
    task_parser.add_argument(
        "--output-root",
        required=True,
    )
    task_parser.add_argument(
        "--task-id",
        required=True,
    )

    combine_parser = commands.add_parser(
        "combine"
    )
    combine_parser.add_argument(
        "--output-root",
        required=True,
    )

    args = parser.parse_args()

    if args.command == "generate":
        generate(args)
    elif args.command == "task-info":
        task_info(args)
    else:
        combine(args)


if __name__ == "__main__":
    main()
