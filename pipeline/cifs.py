#!/usr/bin/env python3
"""Relax once, then save immediate perturbations without post-relaxation."""

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
from ase.io import write

from mlff_attack.attacks import make_attack
from mlff_attack.relaxation import load_structure, run_relaxation, setup_calculator


BASE_DIR = Path(__file__).resolve().parent.parent
BACKENDS = {
    "mace_mh": "mace",
    "uma": "uma",
    "mtp": "mtp",
    "chgnet": "chgnet",
    "mace_model": "mace",
}


def optional(row, name, default=None):
    value = row.get(name, default)
    if value is None or pd.isna(value) or str(value).strip() == "":
        return default
    return value


def optional_int(row, name):
    value = optional(row, name)
    return None if value is None else int(value)


def optional_float(row, name):
    value = optional(row, name)
    return None if value is None else float(value)


def optional_bool(row, name):
    value = optional(row, name)
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def select_model_path(row, backend):
    value = str(row["model_path"])
    if backend in {"mace", "mtp"}:
        return BASE_DIR / value
    if backend == "uma":
        return Path(value).stem
    return value


def safe_name(value):
    return "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in str(value)
    )


def run(args):
    rows = pd.read_csv(Path(args.tests).resolve(), keep_default_na=False)
    rows = rows[
        (rows["model_id"] == args.model_id)
        & (rows["dtype_str"] == "float64")
        & (rows["material_slug"] == args.material_slug)
    ].copy()
    if rows.empty:
        raise SystemExit(
            f"ERROR: no float64 rows for {args.model_id} {args.material_slug}"
        )

    first = rows.iloc[0]
    backend = BACKENDS[args.model_id]
    seed = int(os.environ.get("MLFF_SEED", "42"))
    device = str(first["device"])
    model_path = select_model_path(first, backend)

    atoms = load_structure(BASE_DIR / str(first["input_path"]))
    if atoms is None:
        raise RuntimeError(f"Could not load {first['input_path']}")

    relaxed = setup_calculator(
        atoms.copy(), model_path, device=device, dtype_str="float64",
        seed=seed, calculator=backend,
        mace_head=optional(first, "mace_head"),
        uma_task=optional(first, "uma_task", "omat"),
        uma_charge=optional_int(first, "uma_charge"),
        uma_spin=optional_int(first, "uma_spin"),
    )
    if relaxed is None:
        raise RuntimeError("Calculator setup failed")

    with tempfile.TemporaryDirectory(prefix="licohpf_relax_") as temporary:
        succeeded = run_relaxation(
            relaxed, Path(temporary) / "relaxation.traj",
            fmax=float(optional(first, "relax_fmax", 0.01)),
            max_steps=int(optional(first, "relax_max_steps", 300)),
            optimizer=str(optional(first, "relax_optimizer", "LBFGS")).upper(),
            verbose=True,
        )
        if not succeeded:
            raise RuntimeError("Pre-attack relaxation failed")

    root = Path(args.output_root).resolve()
    structures = root / "structures"
    structures_perturbed = root / "structures_perturbed"

    for _, row in rows.iterrows():
        case_name = safe_name(
            optional(
                row,
                "run_folder",
                row["run_id"],
            )
        )

        relaxed_case_directory = (
            structures / case_name
        )
        perturbed_case_directory = (
            structures_perturbed / case_name
        )

        relaxed_case_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        perturbed_case_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            f"{args.model_id}__"
            f"{args.material_slug}.cif"
        )

        relaxed_output = (
            relaxed_case_directory / filename
        )
        perturbed_output = (
            perturbed_case_directory / filename
        )

        write(
            relaxed_output,
            relaxed,
        )

        make_attack(
            atoms=relaxed.copy(),
            model_path=model_path,
            device=device,
            dtype_str="float64",
            seed=seed,
            output_cif=perturbed_output,
            attack_type=str(
                row["attack_type"]
            ).lower(),
            epsilon=float(row["epsilon"]),
            alpha=optional_float(
                row,
                "alpha",
            ),
            n_steps=int(row["n_steps"]),
            target_energy=optional_float(
                row,
                "target_energy",
            ),
            clip=optional_bool(
                row,
                "clip",
            ),
            verbose=False,
            calculator=backend,
            mace_head=optional(
                row,
                "mace_head",
            ),
            uma_task=optional(
                row,
                "uma_task",
                "omat",
            ),
            uma_charge=optional_int(
                row,
                "uma_charge",
            ),
            uma_spin=optional_int(
                row,
                "uma_spin",
            ),
        )

    print(
        f"Saved {len(rows)} "
        "relaxed/perturbed CIF pairs"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", required=True)
    parser.add_argument("--model-id", choices=tuple(BACKENDS), required=True)
    parser.add_argument("--material-slug", required=True)
    parser.add_argument("--output-root", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
