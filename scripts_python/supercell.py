#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import json
import re

from ase.io import read, write


BASE_DIR = Path(__file__).resolve().parent.parent

MATERIALS = [
    "beta_cristobalite_sio2",
    "beta_quartz_sio2",
    "berlinite_alpo4",
    "zn_cn2",
    "scf3",
    "reo3",
    "graphite",
    "h_bn",
    "mos2_2h",
    "ws2_2h",
    "mose2_2h",
    "wse2_2h",
    "sns2",
    "zrs2",
    "srtio3",
    "ktao3",
    "bazro3",
    "bahfo3",
    "cspbbr3",
    "cspbi3",
]

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

CALCULATORS = ["mace", "uma", "chgnet"]
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
    "calculator",
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


def slug(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def epsilon_tag(epsilon):
    text = f"{float(epsilon):g}"
    return "eps" + text.replace(".", "")


def read_csv_rows(path):
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

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
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def structure_path(material, structures_dir):
    return (
        Path(structures_dir)
        / f"{material['mpid']}_{slug(material['material_label'])}.cif"
    )


def blank_row():
    return {
        column: ""
        for column in COLUMNS
    }


def make_run_row(
    context,
    supercell,
    model,
    attack,
):
    calculator = str(model["calculator"]).lower()
    attack_name = str(attack["name"]).lower()
    attack_steps = int(attack["n_steps"])

    alpha = attack.get("alpha")

    if (
        alpha is None
        and attack.get("alpha_ratio") is not None
    ):
        alpha = (
            EPSILON
            * float(attack["alpha_ratio"])
        )

    run_folder = (
        f"{attack_name}_{epsilon_tag(EPSILON)}"
    )

    row = blank_row()

    row.update({
        "run_id": (
            f"{supercell['material_slug']}_"
            f"{calculator}_{run_folder}"
        ),
        "material_label": supercell["material_label"],
        "material_slug": supercell["material_slug"],
        "run_folder": run_folder,
        "input_path": supercell["input_path"],
        "model_path": model["model_path"],
        "attack_type": attack["attack_type"],
        "epsilon": f"{EPSILON:g}",
        "n_steps": attack_steps,
        "alpha": (
            ""
            if alpha is None
            else f"{float(alpha):g}"
        ),
        "clip": (
            ""
            if attack.get("clip") is None
            else attack["clip"]
        ),
        "device": context["device"],
        "output_dir": "outputs",
        "relax_fmax": context["relax_fmax"],
        "relax_max_steps": context["relax_max_steps"],
        "relax_optimizer": context["relax_optimizer"],
        "calculator": calculator,
        "dtype_str": DTYPE,
        "seed": SEED,
        "base_material_slug": (
            context["base_material_slug"]
        ),
        "base_material_label": (
            context["base_material_label"]
        ),
        "base_input_path": (
            context["base_input_path"]
        ),
        "supercell_repeat_x": supercell["repeat_x"],
        "supercell_repeat_y": supercell["repeat_y"],
        "supercell_repeat_z": supercell["repeat_z"],
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

    if calculator == "mace":
        row["mace_head"] = model.get(
            "mace_head",
            "",
        )

    elif calculator == "uma":
        row["uma_task"] = model.get(
            "uma_task",
            "",
        )
        row["uma_charge"] = model.get(
            "uma_charge",
            "",
        )
        row["uma_spin"] = model.get(
            "uma_spin",
            "",
        )

    return row


def selected_materials(material_rows):
    by_slug = {
        slug(row["material_label"]): row
        for row in material_rows
    }

    missing = [
        material
        for material in MATERIALS
        if material not in by_slug
    ]

    if missing:
        raise SystemExit(
            "Missing requested materials: "
            + ", ".join(missing)
        )

    return [
        by_slug[material]
        for material in MATERIALS
    ]


def generate(args):
    output_root = Path(args.output_root).resolve()
    structures_dir = (
        BASE_DIR / args.structures_dir
    ).resolve()

    config = load_config(
        BASE_DIR / args.config
    )
    material_rows = read_csv_rows(
        BASE_DIR / args.materials
    )
    materials = selected_materials(material_rows)

    tests = []
    metadata = []

    structures_root = (
        output_root / "supercell_structures"
    )
    structures_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for material in materials:
        base_slug = slug(material["material_label"])
        base_path = structure_path(
            material,
            structures_dir,
        ).resolve()

        if not base_path.exists():
            raise SystemExit(
                f"Missing base CIF: {base_path}"
            )

        atoms = read(base_path)
        unit_cell_atoms = len(atoms)

        context = {
            "base_material_slug": base_slug,
            "base_material_label": (
                material["material_label"]
            ),
            "base_input_path": str(base_path),
            "device": config.get(
                "device",
                "cpu",
            ),
            "relax_fmax": config.get(
                "relax_fmax",
                0.01,
            ),
            "relax_max_steps": config.get(
                "relax_max_steps",
                300,
            ),
            "relax_optimizer": config.get(
                "relax_optimizer",
                "LBFGS",
            ),
        }

        for repeat_x, repeat_y, repeat_z in REPEAT_TUPLES:
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

            if args.force or not cif_path.exists():
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

            atom_count = (
                unit_cell_atoms
                * repeat_x
                * repeat_y
                * repeat_z
            )

            supercell = {
                "material_label": (
                    f"{material['material_label']}_"
                    f"r{repeat_label}"
                ),
                "material_slug": supercell_slug,
                "input_path": str(cif_path.resolve()),
                "repeat_x": repeat_x,
                "repeat_y": repeat_y,
                "repeat_z": repeat_z,
                "repeat_tuple": repeat_label,
                "unit_cell_atoms": unit_cell_atoms,
                "supercell_atoms": atom_count,
            }

            metadata.append({
                "base_material_slug": base_slug,
                "base_material_label": (
                    material["material_label"]
                ),
                "supercell_material_slug": (
                    supercell_slug
                ),
                "base_input_path": str(base_path),
                "supercell_input_path": (
                    str(cif_path.resolve())
                ),
                "repeat_x": repeat_x,
                "repeat_y": repeat_y,
                "repeat_z": repeat_z,
                "repeat_tuple": repeat_label,
                "unit_cell_atoms": unit_cell_atoms,
                "supercell_atoms": atom_count,
            })

            for model in config["models"]:
                for attack in config["attacks"]:
                    tests.append(
                        make_run_row(
                            context,
                            supercell,
                            model,
                            attack,
                        )
                    )

    write_csv_rows(
        output_root / "generated_supercell_tests.csv",
        tests,
        COLUMNS,
    )

    metadata_columns = [
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

    write_csv_rows(
        output_root / "supercell_metadata.csv",
        metadata,
        metadata_columns,
    )

    expected_tests = (
        len(MATERIALS)
        * len(REPEAT_TUPLES)
        * len(config["models"])
        * len(config["attacks"])
    )

    if len(tests) != expected_tests:
        raise SystemExit(
            f"Expected {expected_tests} tests, "
            f"generated {len(tests)}"
        )

    print(
        f"Wrote {len(tests)} rows to "
        f"{output_root / 'generated_supercell_tests.csv'}"
    )
    print(
        f"Wrote {len(metadata)} rows to "
        f"{output_root / 'supercell_metadata.csv'}"
    )


def task_list():
    return [
        (material, repeat_label, calculator)
        for material in MATERIALS
        for repeat_label in [
            f"{x}x{y}x{z}"
            for x, y, z in REPEAT_TUPLES
        ]
        for calculator in CALCULATORS
    ]


def task_info(args):
    task_id = int(args.task_id)
    tasks = task_list()

    if task_id < 1 or task_id > len(tasks):
        raise SystemExit(
            f"task-id must be 1..{len(tasks)}, "
            f"got {task_id}"
        )

    material, repeat_label, calculator = (
        tasks[task_id - 1]
    )

    output_root = Path(args.output_root).resolve()
    tests_path = (
        output_root / "generated_supercell_tests.csv"
    )
    rows = read_csv_rows(tests_path)

    selected = [
        row
        for row in rows
        if (
            row["base_material_slug"] == material
            and row["supercell_repeat_tuple"]
            == repeat_label
            and row["calculator"] == calculator
        )
    ]

    if not selected:
        raise SystemExit(
            f"No rows for {material} "
            f"{repeat_label} {calculator}"
        )

    selected_path = (
        output_root
        / "material_tests"
        / DTYPE
        / (
            f"{calculator}_{material}_"
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
            f"{DTYPE}_{calculator}_{material}_"
            f"r{repeat_label}_summary.csv"
        )
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"MATERIAL={material}")
    print(f"REPEAT={repeat_label}")
    print(f"CALCULATOR={calculator}")
    print(f"TEST_CSV={selected_path}")
    print(f"SUMMARY_FILE={summary_path}")


def combine(args):
    output_root = Path(args.output_root).resolve()
    summary_dir = output_root / "array_summaries"

    for calculator in CALCULATORS:
        summary_paths = sorted(
            summary_dir.glob(
                f"{DTYPE}_{calculator}_*_summary.csv"
            )
        )

        if not summary_paths:
            raise SystemExit(
                f"No {calculator} summaries found in "
                f"{summary_dir}"
            )

        combined_rows = []
        columns = []

        for summary_path in summary_paths:
            rows = read_csv_rows(summary_path)

            for row in rows:
                for column in row:
                    if column not in columns:
                        columns.append(column)

                combined_rows.append(row)

        output_path = (
            output_root
            / f"outputs_{DTYPE}"
            / calculator
            / "summary.csv"
        )

        write_csv_rows(
            output_path,
            combined_rows,
            columns,
        )

        print(
            f"Wrote {len(combined_rows)} rows to "
            f"{output_path}"
        )


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument(
        "--output-root",
        required=True,
    )
    generate_parser.add_argument(
        "--materials",
        default="tests_materials.csv",
    )
    generate_parser.add_argument(
        "--config",
        default="tests_comprehensive.json",
    )
    generate_parser.add_argument(
        "--structures-dir",
        default="mp_structures",
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
    )

    task_parser = commands.add_parser("task-info")
    task_parser.add_argument(
        "--output-root",
        required=True,
    )
    task_parser.add_argument(
        "--task-id",
        required=True,
    )

    combine_parser = commands.add_parser("combine")
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