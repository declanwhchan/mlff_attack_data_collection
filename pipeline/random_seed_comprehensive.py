#!/usr/bin/env python3
import re
from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use("Agg")

import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from load_dft import dft_coverage_table, read_dft_structure
from ase.io import read as ase_read
from force_rms_plots import (
    add_post_attack_rms_columns,
    add_post_attack_relaxed_rms_at_fmax_column,
    save_random_seed_rms_plots,
)


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_tests import (
    coordination_by_atom,
    edge_jaccard_distance,
    neighbor_edge_set,
    rdf_l1_distance,
)


TRIALS = [
    ("trial1_seed42", 42),
    ("trial2_seed43", 43),
    ("trial3_seed44", 44),
    ("trial4_seed45", 45),
    ("trial5_seed46", 46),
]

ADVERSARIAL_ATTACKS = [
    "FGSM",
    "I-FGSM",
    "PGD",
]

METHODS = [
    "Contour",
    *ADVERSARIAL_ATTACKS,
]

CALCULATORS = [
    "mace_mh",
    "uma",
    # "mtp",         # intentionally excluded
    # "chgnet",      # intentionally excluded
    # "mace_model",  # intentionally excluded
    "dft_mace_mh",
    "dft_uma",
    # "dft_chgnet",  # intentionally excluded
]

MODEL_LABELS = {
    "mace_mh": "MACE-MH-1",
    "uma": "UMA-S-1p1",
    "mtp": "MTP",
    "chgnet": "CHGNet",
    "mace_model": "MACE Model",
    "dft_mace_mh": "DFT (MACE-MH)",
    "dft_uma": "DFT (UMA)",
    "dft_chgnet": "DFT (CHGNet)",
}

COLORS = {
    "mace_mh": "#0072B2",
    "uma": "#D55E00",
    "mtp": "#CC79A7",
    "chgnet": "#009E73",
    "mace_model": "#E69F00",
    # Monochrome DFT reference colors
    "dft_mace_mh": "#7A7A7A",
    "dft_uma": "#B0B0B0",
    "dft_chgnet": "#D0D0D0",
}

RELAXATION_STEP_UPPER_LIMIT = 600


def relaxation_step_upper_limit(project_root):
    """Return the fixed relaxation-step ceiling for the input dataset."""
    if "2d_structures" in str(project_root).lower():
        return 300
    return 600


def model_label(model_id):
    return MODEL_LABELS.get(
        str(model_id),
        str(model_id),
    )

SEED_STYLES = {
    42: ("-", "o"),
    43: ("--", "s"),
    44: ("-.", "^"),
    45: (":", "D"),
    46: ((0, (3, 1, 1, 1)), "P"),
}

STAGES = [
    "before_attack_after_relaxation",
    "after_attack_before_relaxation",
    "after_attack_after_relaxation",
]

TOPOLOGY_METRICS = [
    (
        "neighbor_jaccard_distance",
        "Neighbor Jaccard distance",
    ),
    (
        "rdf_l1_distance",
        r"RDF L1 distance ($\AA$)",
    ),
    (
        "coordination_change_max",
        r"Max $\Delta$ CN",
    ),
]


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    return value if np.isfinite(value) else np.nan


def read_force_csv(path):
    try:
        data = pd.read_csv(Path(path))
    except (
        OSError,
        ValueError,
        pd.errors.ParserError,
    ):
        return None

    required = {
        "atom_index",
        "x",
        "y",
        "z",
        "fx",
        "fy",
        "fz",
    }

    if not required.issubset(data.columns):
        return None

    return data


def compare_force_csvs(before_path, after_path):
    before = read_force_csv(before_path)
    after = read_force_csv(after_path)

    if before is None or after is None:
        return np.nan, np.nan

    merged = before.merge(
        after,
        on="atom_index",
        suffixes=("_before", "_after"),
    )

    if merged.empty:
        return np.nan, np.nan

    before_positions = merged[
        ["x_before", "y_before", "z_before"]
    ].to_numpy(dtype=float)

    after_positions = merged[
        ["x_after", "y_after", "z_after"]
    ].to_numpy(dtype=float)

    before_forces = merged[
        ["fx_before", "fy_before", "fz_before"]
    ].to_numpy(dtype=float)

    after_forces = merged[
        ["fx_after", "fy_after", "fz_after"]
    ].to_numpy(dtype=float)

    displacement = np.linalg.norm(
        after_positions - before_positions,
        axis=1,
    )

    delta_force = np.linalg.norm(
        after_forces - before_forces,
        axis=1,
    )

    return (
        float(np.median(displacement)),
        float(np.median(delta_force)),
    )


def compare_relaxation_trajectory(path):
    path = Path(path)

    try:
        initial = ase_read(path, index=0)
        relaxed = ase_read(path, index=-1)
    except Exception:
        return np.nan, np.nan, None, None

    if len(initial) != len(relaxed):
        return np.nan, np.nan, None, None

    displacement = np.linalg.norm(
        relaxed.positions - initial.positions,
        axis=1,
    )

    median_displacement = float(
        np.median(displacement)
    )

    median_delta_force = np.nan

    try:
        initial_forces = initial.get_forces()
        relaxed_forces = relaxed.get_forces()

        delta_force = np.linalg.norm(
            relaxed_forces - initial_forces,
            axis=1,
        )

        median_delta_force = float(
            np.median(delta_force)
        )
    except Exception:
        pass

    return (
        median_displacement,
        median_delta_force,
        initial,
        relaxed,
    )


def topology_metrics(initial, final):
    if initial is None or final is None:
        return {
            "neighbor_jaccard_distance": np.nan,
            "rdf_l1_distance": np.nan,
            "coordination_change_max": np.nan,
        }

    try:
        initial_edges = neighbor_edge_set(initial)
        final_edges = neighbor_edge_set(final)

        jaccard = edge_jaccard_distance(
            initial_edges,
            final_edges,
        )

        initial_coordination = coordination_by_atom(
            initial_edges,
            initial,
        )
        final_coordination = coordination_by_atom(
            final_edges,
            final,
        )

        atom_keys = (
            set(initial_coordination)
            | set(final_coordination)
        )

        changes = [
            abs(
                final_coordination.get(atom, 0)
                - initial_coordination.get(atom, 0)
            )
            for atom in atom_keys
        ]

        coordination_max = (
            float(np.max(changes))
            if changes
            else 0.0
        )

        rdf_distance = rdf_l1_distance(
            initial,
            final,
        )

        return {
            "neighbor_jaccard_distance": float(
                jaccard
            ),
            "rdf_l1_distance": float(
                rdf_distance
            ),
            "coordination_change_max": (
                coordination_max
            ),
        }

    except Exception:
        return {
            "neighbor_jaccard_distance": np.nan,
            "rdf_l1_distance": np.nan,
            "coordination_change_max": np.nan,
        }


def stage_column(stage, metric):
    return f"{stage}__{metric}"


def resolve_record_artifact(
    row,
    column,
    default_name,
    extra_columns=(),
    extra_names=(),
):
    """
    Resolve an artifact recorded by any calculator backend.

    The normal recorded column is tried first. Additional
    calculator-specific columns/names can be supplied for DFT
    reference artifacts.
    """
    run_dir = Path(
        str(row.get("run_dir", ""))
    )

    candidates = []

    candidate_columns = [
        column,
        *extra_columns,
    ]

    candidate_names = [
        default_name,
        *extra_names,
    ]

    # ---------------------------------------------------------
    # 1. Explicit artifact columns.
    # ---------------------------------------------------------

    for column_name in candidate_columns:
        value = row.get(column_name)

        if (
            value is None
            or pd.isna(value)
            or not str(value).strip()
        ):
            continue

        recorded = Path(
            str(value).strip()
        )

        candidates.append(
            recorded
        )

        if not recorded.is_absolute():
            candidates.append(
                run_dir / recorded
            )

    # ---------------------------------------------------------
    # 2. actual_output_dir + conventional filename.
    # ---------------------------------------------------------

    actual_output = row.get(
        "actual_output_dir"
    )

    if (
        actual_output is not None
        and not pd.isna(actual_output)
        and str(actual_output).strip()
    ):
        output_dir = Path(
            str(actual_output).strip()
        )

        for name in candidate_names:
            candidates.append(
                output_dir / name
            )

            if not output_dir.is_absolute():
                candidates.append(
                    run_dir
                    / output_dir
                    / name
                )

    # ---------------------------------------------------------
    # 3. run_dir + conventional filename.
    # ---------------------------------------------------------

    for name in candidate_names:
        candidates.append(
            run_dir / name
        )

    # ---------------------------------------------------------
    # 4. De-duplicate and return first existing file.
    # ---------------------------------------------------------

    seen = set()

    for candidate in candidates:
        candidate_key = str(
            candidate.resolve()
            if candidate.exists()
            else candidate
        )

        if candidate_key in seen:
            continue

        seen.add(candidate_key)

        if candidate.is_file():
            return candidate

    # Preserve existing missing-file behavior.
    return run_dir / default_name


def resolve_force_artifacts(row):
    """
    Resolve before/perturbed/after force CSVs.

    DFT rows are allowed to use either the normal force-artifact
    columns or DFT-specific artifact columns if the dataset records
    them separately.
    """
    is_dft = str(
        row.get("calculator", "")
    ).startswith("dft_")

    if is_dft:
        before_columns = (
            "dft_before_force_csv",
            "reference_before_force_csv",
            "before_dft_force_csv",
            "before_force_csv",
        )

        perturbed_columns = (
            "dft_perturbed_force_csv",
            "reference_perturbed_force_csv",
            "perturbed_dft_force_csv",
            "perturbed_force_csv",
        )

        after_columns = (
            "dft_after_force_csv",
            "reference_after_force_csv",
            "after_dft_force_csv",
            "after_force_csv",
        )

        before_names = (
            "dft_before_forces.csv",
            "reference_before_forces.csv",
            "before_dft_forces.csv",
            "before_forces.csv",
        )

        perturbed_names = (
            "dft_perturbed_forces.csv",
            "reference_perturbed_forces.csv",
            "perturbed_dft_forces.csv",
            "perturbed_forces.csv",
        )

        after_names = (
            "dft_after_forces.csv",
            "reference_after_forces.csv",
            "after_dft_forces.csv",
            "after_forces.csv",
        )

    else:
        before_columns = (
            "before_force_csv",
        )

        perturbed_columns = (
            "perturbed_force_csv",
        )

        after_columns = (
            "after_force_csv",
        )

        before_names = (
            "before_forces.csv",
        )

        perturbed_names = (
            "perturbed_forces.csv",
        )

        after_names = (
            "after_forces.csv",
        )

    before_path = resolve_record_artifact(
        row,
        before_columns[0],
        before_names[0],
        extra_columns=before_columns[1:],
        extra_names=before_names[1:],
    )

    perturbed_path = resolve_record_artifact(
        row,
        perturbed_columns[0],
        perturbed_names[0],
        extra_columns=perturbed_columns[1:],
        extra_names=perturbed_names[1:],
    )

    after_path = resolve_record_artifact(
        row,
        after_columns[0],
        after_names[0],
        extra_columns=after_columns[1:],
        extra_names=after_names[1:],
    )

    return (
        before_path,
        perturbed_path,
        after_path,
    )


def calculate_stage_metrics(row):
    (
        before_force_path,
        perturbed_force_path,
        after_force_path,
    ) = resolve_force_artifacts(row)
    trajectory_path = resolve_record_artifact(
        row,
        "before_relax_traj",
        "before_attack_relaxation.traj",
    )

    (
        baseline_displacement,
        baseline_delta_force,
        baseline_initial,
        baseline_relaxed,
    ) = compare_relaxation_trajectory(
        trajectory_path
    )

    baseline_topology = topology_metrics(
        baseline_initial,
        baseline_relaxed,
    )

    (
        immediate_displacement,
        immediate_delta_force,
    ) = compare_force_csvs(
        before_force_path,
        perturbed_force_path,
    )

    (
        final_displacement,
        final_delta_force,
    ) = compare_force_csvs(
        before_force_path,
        after_force_path,
    )

    if str(row.get("calculator", "")).lower().startswith("dft_"):
        baseline_delta_force = np.nan
        immediate_delta_force = np.nan
        final_delta_force = np.nan

    return {
        "before_attack_after_relaxation": {
            "median_displacement_a": (
                baseline_displacement
            ),
            "median_delta_force_ev_a": (
                baseline_delta_force
            ),
            "relax_steps": finite_float(
                row.get("before_relax_steps")
            ),
            **baseline_topology,
        },
        "after_attack_before_relaxation": {
            "median_displacement_a": (
                immediate_displacement
            ),
            "median_delta_force_ev_a": (
                immediate_delta_force
            ),
            # This is the relaxation that occurs next.
            "relax_steps": finite_float(
                row.get("after_relax_steps")
            ),
            "neighbor_jaccard_distance": finite_float(
                row.get(
                    "perturbed_neighbor_jaccard_distance"
                )
            ),
            "rdf_l1_distance": finite_float(
                row.get(
                    "perturbed_rdf_l1_distance"
                )
            ),
            "coordination_change_max": finite_float(
                row.get(
                    "perturbed_coordination_change_max"
                )
            ),
        },
        "after_attack_after_relaxation": {
            "median_displacement_a": (
                final_displacement
            ),
            "median_delta_force_ev_a": (
                final_delta_force
            ),
            "relax_steps": finite_float(
                row.get("after_relax_steps")
            ),
            "neighbor_jaccard_distance": finite_float(
                row.get(
                    "neighbor_jaccard_distance"
                )
            ),
            "rdf_l1_distance": finite_float(
                row.get("rdf_l1_distance")
            ),
            "coordination_change_max": finite_float(
                row.get(
                    "coordination_change_max"
                )
            ),
        },
    }


def load_trials(project_root):
    """
    Load every readable trial.

    Missing, empty or partially generated trials are recorded but do
    not prevent plots from being generated.
    """
    frames = []
    missing = []

    required_columns = [
        "run_id",
        "material_slug",
        "calculator",
        "attack_label",
        "epsilon",
        "epsilon_percent_displacement",
        "run_dir",
        "trial",
        "seed",
    ]

    for trial_name, seed in TRIALS:
        path = (
            project_root
            / trial_name
            / "outputs_comprehensive"
            / "float64"
            / "combined_dataset.csv"
        )

        if not path.is_file():
            missing.append({
                "trial": trial_name,
                "seed": seed,
                "reason": f"missing file: {path}",
            })
            continue

        try:
            data = pd.read_csv(path)
        except Exception as error:
            missing.append({
                "trial": trial_name,
                "seed": seed,
                "reason": f"unreadable dataset: {error}",
            })
            continue

        if data.empty:
            missing.append({
                "trial": trial_name,
                "seed": seed,
                "reason": "empty dataset",
            })
            continue

        data = data.copy()
        data["trial"] = trial_name
        data["seed"] = seed
        frames.append(data)

    if not frames:
        # Return an empty table with the correct schema. This allows
        # the plotting functions to create "No matched seed data"
        # figures instead of terminating.
        return (
            pd.DataFrame(
                columns=required_columns
            ),
            missing,
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    return combined, missing


def empty_stage_values():
    values = {}

    for stage in STAGES:
        for metric in (
            "median_displacement_a",
            "median_delta_force_ev_a",
            "relax_steps",
            "neighbor_jaccard_distance",
            "rdf_l1_distance",
            "coordination_change_max",
        ):
            values[stage_column(stage, metric)] = np.nan

    return values


def contour_endpoint_inputs(frame_data):
    """Return the final sampled contour state for each endpoint."""
    if frame_data.empty:
        return pd.DataFrame()

    keys = [
        "material_slug",
        "calculator",
        "beta",
    ]

    if not set(keys).issubset(frame_data.columns):
        return pd.DataFrame()

    data = frame_data.copy()
    data["_contour_order"] = (
        numeric(data["step"])
        if "step" in data.columns
        else np.arange(len(data), dtype=float)
    )

    data = (
        data.sort_values("_contour_order")
        .groupby(keys, as_index=False, dropna=False)
        .tail(1)
    )

    columns = keys + [
        "contour_median_displacement_percent_min_lattice",
    ]

    columns = [
        column
        for column in columns
        if column in data.columns
    ]

    return data[columns].rename(columns={
        "contour_median_displacement_percent_min_lattice": (
            "contour_input_displacement_percent_min_lattice"
        ),
    })


def load_contour_trials(project_root):
    """
    Convert existing contour metric tables into the same plotting schema
    as the adversarial records.

    Contour uses measured displacement from its relaxed starting structure
    on the x-axis; it does not have a nominal adversarial epsilon.
    """
    records = []
    missing = []

    for trial_name, seed in TRIALS:
        contour_root = (
            project_root
            / trial_name
            / "outputs_comprehensive"
            / "float64"
            / "contour"
        )

        frame_path = contour_root / "contour_frame_metrics.csv"
        relaxed_path = (
            contour_root
            / "contour_relaxed_endpoint_metrics.csv"
        )

        try:
            frame_data = (
                pd.read_csv(frame_path)
                if frame_path.is_file()
                else pd.DataFrame()
            )
        except Exception as error:
            frame_data = pd.DataFrame()
            missing.append({
                "trial": trial_name,
                "seed": seed,
                "reason": f"unreadable contour frames: {error}",
            })

        try:
            relaxed_data = (
                pd.read_csv(relaxed_path)
                if relaxed_path.is_file()
                else pd.DataFrame()
            )
        except Exception as error:
            relaxed_data = pd.DataFrame()
            missing.append({
                "trial": trial_name,
                "seed": seed,
                "reason": f"unreadable relaxed contour endpoints: {error}",
            })

        if frame_data.empty and relaxed_data.empty:
            missing.append({
                "trial": trial_name,
                "seed": seed,
                "reason": f"missing contour metric tables under {contour_root}",
            })
            continue

        if not frame_data.empty:
            required = {
                "material_slug",
                "calculator",
                "contour_median_displacement_percent_min_lattice",
            }

            if required.issubset(frame_data.columns):
                for frame_index, row in frame_data.iterrows():
                    x_value = finite_float(
                        row.get(
                            "contour_median_displacement_percent_min_lattice"
                        )
                    )

                    if not np.isfinite(x_value) or x_value <= 0.0:
                        continue

                    values = empty_stage_values()
                    stage = "after_attack_before_relaxation"

                    values[stage_column(
                        stage,
                        "median_displacement_a",
                    )] = finite_float(
                        row.get("contour_median_displacement_a")
                    )
                    values[stage_column(
                        stage,
                        "median_delta_force_ev_a",
                    )] = finite_float(
                        row.get("contour_median_force_delta_ev_a")
                    )
                    values[stage_column(
                        stage,
                        "neighbor_jaccard_distance",
                    )] = finite_float(
                        row.get("contour_neighbor_jaccard_distance")
                    )
                    values[stage_column(
                        stage,
                        "rdf_l1_distance",
                    )] = finite_float(
                        row.get("contour_rdf_l1_distance")
                    )
                    values[stage_column(
                        stage,
                        "coordination_change_max",
                    )] = finite_float(
                        row.get("contour_coordination_change_max")
                    )

                    beta_value = finite_float(
                        row.get("beta")
                    )
                    step_value = finite_float(
                        row.get("step")
                    )

                    if (
                        np.isfinite(beta_value)
                        and np.isfinite(step_value)
                    ):
                        contour_key = (
                            beta_value
                            + step_value * 1.0e-6
                        )
                    else:
                        contour_key = x_value

                    records.append({
                        "run_id": (
                            f"{trial_name}_contour_frame_{frame_index}"
                        ),
                        "material_slug": row.get("material_slug"),
                        "calculator": row.get("calculator"),
                        "attack_label": "Contour",
                        "epsilon": contour_key,
                        "epsilon_percent_displacement": x_value,
                        "seed": seed,
                        "trial": trial_name,
                        "method_source": "contour_frame",
                        **values,
                    })

        if not relaxed_data.empty:
            endpoint_inputs = contour_endpoint_inputs(frame_data)
            keys = [
                "material_slug",
                "calculator",
                "beta",
            ]

            if (
                not endpoint_inputs.empty
                and set(keys).issubset(relaxed_data.columns)
            ):
                relaxed_data = relaxed_data.merge(
                    endpoint_inputs,
                    on=keys,
                    how="left",
                )

            for endpoint_index, row in relaxed_data.iterrows():
                x_value = finite_float(
                    row.get(
                        "contour_input_displacement_percent_min_lattice"
                    )
                )

                if not np.isfinite(x_value):
                    x_value = finite_float(
                        row.get(
                            "contour_relaxed_displacement_percent_min_lattice"
                        )
                    )

                if not np.isfinite(x_value) or x_value <= 0.0:
                    continue

                values = empty_stage_values()
                immediate_stage = "after_attack_before_relaxation"
                final_stage = "after_attack_after_relaxation"
                relaxation_steps = finite_float(
                    row.get("contour_endpoint_relaxation_steps")
                )

                values[stage_column(
                    immediate_stage,
                    "relax_steps",
                )] = relaxation_steps
                values[stage_column(
                    final_stage,
                    "relax_steps",
                )] = relaxation_steps
                values[stage_column(
                    final_stage,
                    "median_displacement_a",
                )] = finite_float(
                    row.get("contour_relaxed_median_displacement_a")
                )
                values[stage_column(
                    final_stage,
                    "median_delta_force_ev_a",
                )] = finite_float(
                    row.get("contour_relaxed_median_force_delta_ev_a")
                )
                values[stage_column(
                    final_stage,
                    "neighbor_jaccard_distance",
                )] = finite_float(
                    row.get("contour_relaxed_neighbor_jaccard_distance")
                )
                values[stage_column(
                    final_stage,
                    "rdf_l1_distance",
                )] = finite_float(
                    row.get("contour_relaxed_rdf_l1_distance")
                )
                values[stage_column(
                    final_stage,
                    "coordination_change_max",
                )] = finite_float(
                    row.get("contour_relaxed_coordination_change_max")
                )

                beta_value = finite_float(
                    row.get("beta")
                )

                contour_key = (
                    beta_value
                    if np.isfinite(beta_value)
                    else x_value
                )

                records.append({
                    "run_id": (
                        f"{trial_name}_contour_endpoint_{endpoint_index}"
                    ),
                    "material_slug": row.get("material_slug"),
                    "calculator": row.get("calculator"),
                    "attack_label": "Contour",
                    "epsilon": contour_key,
                    "epsilon_percent_displacement": x_value,
                    "seed": seed,
                    "trial": trial_name,
                    "method_source": "contour_relaxed_endpoint",
                    **values,
                })

    return pd.DataFrame(records), missing


def prepare_records(records):
    """
    Prepare all usable rows without requiring every trial, material,
    attack, epsilon or MLFF to be present.
    """
    required_columns = [
        "run_id",
        "material_slug",
        "calculator",
        "attack_label",
        "epsilon",
        "epsilon_percent_displacement",
        "n_steps",
        "run_dir",
        "seed",
    ]

    data = records.copy()

    # Add missing columns instead of aborting.
    for column in required_columns:
        if column not in data.columns:
            data[column] = np.nan

    # Retain every valid available row, including the ``_stepsNNN``
    # run IDs used by iterative attacks. Those are distinct attack cases,
    # not auxiliary records.
    valid_calculators = set(
        CALCULATORS
    )

    valid_attacks = set(
        ADVERSARIAL_ATTACKS
    )

    data = data[
        data["attack_label"].astype(str).isin(
            valid_attacks
        )
        & data["calculator"].astype(str).isin(
            valid_calculators
        )
    ].copy()

    print(
        "Calculator counts after filtering:"
    )

    print(
        data["calculator"]
        .value_counts(dropna=False)
        .to_string()
    )

    data["epsilon"] = numeric(
        data["epsilon"]
    )

    data["epsilon_percent_displacement"] = numeric(
        data["epsilon_percent_displacement"]
    )

    data["n_steps"] = numeric(data["n_steps"]).fillna(1.0)

    stage_results = [
        calculate_stage_metrics(row)
        for _, row in data.iterrows()
    ]

    metric_names = [
        "median_displacement_a",
        "median_delta_force_ev_a",
        "relax_steps",
        "neighbor_jaccard_distance",
        "rdf_l1_distance",
        "coordination_change_max",
    ]

    for stage in STAGES:
        for metric in metric_names:
            column = stage_column(
                stage,
                metric,
            )

            data[column] = [
                result[stage][metric]
                for result in stage_results
            ]

    present_models = sorted(
        data["calculator"]
        .dropna()
        .astype(str)
        .unique()
    )

    present_seeds = sorted(
        pd.to_numeric(
            data["seed"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    print(
        "Random-seed plotting will use available models: "
        f"{present_models}"
    )
    print(
        "Random-seed plotting will use available seeds: "
        f"{present_seeds}"
    )
    print(
        f"Random-seed plotting will use {len(data)} valid rows."
    )

    # -------------------------------------------------------------
    # DFT delta-force coverage diagnostic.
    # -------------------------------------------------------------

    for calculator in (
        "mace_mh",
        "uma",
        "dft_mace_mh",
        "dft_uma",
    ):
        calculator_rows = data[
            data["calculator"].astype(str)
            == calculator
        ]

        if calculator_rows.empty:
            continue

        delta_columns = [
            stage_column(
                stage,
                "median_delta_force_ev_a",
            )
            for stage in STAGES
        ]

        finite_counts = {
            column: int(
                numeric(
                    calculator_rows[column]
                ).notna().sum()
            )
            for column in delta_columns
        }

        print(
            f"Delta-force coverage for {calculator}: "
            f"{finite_counts}"
        )

    return data


def physical_metrics(stage):
    if stage == "before_attack_after_relaxation":
        step_label = "Relaxation steps"
    elif stage == "after_attack_before_relaxation":
        step_label = "Subsequent post-attack relaxation steps"
    else:
        step_label = "Relaxation steps"

    return [
        (
            stage_column(
                stage,
                "median_displacement_a",
            ),
            r"Median displacement ($\AA$)",
        ),
        (
            stage_column(
                stage,
                "median_delta_force_ev_a",
            ),
            r"Median $\Delta$ force (eV/$\AA$)",
        ),
        (
            stage_column(stage, "relax_steps"),
            step_label,
        ),
    ]


def topology_metrics_for_stage(stage):
    return [
        (
            stage_column(stage, metric),
            label,
        )
        for metric, label in TOPOLOGY_METRICS
    ]


def seed_curves(records, metric):
    clean = records.copy()

    if "n_steps" not in clean.columns:
        clean["n_steps"] = 1.0
    else:
        clean["n_steps"] = numeric(clean["n_steps"]).fillna(1.0)

    clean[metric] = numeric(
        clean[metric]
    )

    # Keep a diagnostic before dropping NaNs.
    for calculator in CALCULATORS:
        calculator_mask = (
            clean["calculator"].astype(str)
            == calculator
        )

        if not calculator_mask.any():
            continue

        finite_count = int(
            clean.loc[
                calculator_mask,
                metric,
            ].notna().sum()
        )

        if finite_count == 0:
            print(
                f"WARNING: no finite values for "
                f"{calculator} / {metric}"
            )

    clean = clean.dropna(
        subset=[
            "seed",
            "calculator",
            "attack_label",
            "epsilon",
            "epsilon_percent_displacement",
            metric,
        ]
    )

    return (
        clean.groupby(
            [
                "seed",
                "calculator",
                "attack_label",
                "epsilon",
                "n_steps",
            ],
            as_index=False,
        )
        .agg(
            epsilon_percent_displacement=(
                "epsilon_percent_displacement",
                "median",
            ),
            value=(
                metric,
                "median",
            ),
            material_count=(
                "material_slug",
                "nunique",
            ),
        )
        .sort_values(
            ["epsilon", "n_steps"]
        )
    )


def aggregate_curves(curves):
    """
    Aggregate every available seed.

    One available seed is sufficient. The seed_count column records
    how many seeds contributed to each point.
    """
    columns = [
        "calculator",
        "attack_label",
        "epsilon",
        "n_steps",
        "epsilon_percent_displacement",
        "median",
        "q25",
        "q75",
        "seed_count",
    ]

    if curves.empty:
        return pd.DataFrame(
            columns=columns
        )

    rows = []

    for key, group in curves.groupby(
        [
            "calculator",
            "attack_label",
            "epsilon",
            "n_steps",
        ],
        dropna=False,
    ):
        values = numeric(
            group["value"]
        ).dropna().to_numpy(dtype=float)

        if len(values) == 0:
            continue

        epsilon_percent = numeric(
            group["epsilon_percent_displacement"]
        ).dropna().to_numpy(dtype=float)

        if len(epsilon_percent) == 0:
            continue

        rows.append({
            "calculator": key[0],
            "attack_label": key[1],
            "epsilon": float(key[2]),
            "n_steps": float(key[3]),
            "epsilon_percent_displacement": float(
                np.median(epsilon_percent)
            ),
            "median": float(
                np.median(values)
            ),
            "q25": float(
                np.percentile(values, 25)
            ),
            "q75": float(
                np.percentile(values, 75)
            ),
            "seed_count": int(
                group["seed"].nunique()
            ),
        })

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def configure_y_axis(
    ax,
    values,
    scale="linear",
):
    """
    Configure robust y-axis scaling.

    symlog is used for force metrics because it supports zero while
    still displaying values spanning many orders of magnitude.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if scale == "symlog":
        nonzero = np.abs(
            values[values != 0]
        )

        if len(nonzero):
            linthresh = max(
                float(
                    np.percentile(
                        nonzero,
                        10,
                    )
                ),
                float(np.max(nonzero)) * 1e-8,
                1e-12,
            )
        else:
            linthresh = 1e-12

        ax.set_yscale(
            "symlog",
            linthresh=linthresh,
            linscale=1.0,
            base=10,
        )

        if len(values):
            minimum = float(
                np.min(values)
            )
            maximum = float(
                np.max(values)
            )

            if minimum >= 0:
                if maximum > 0:
                    ax.set_ylim(
                        0,
                        maximum * 1.12,
                    )
                else:
                    ax.set_ylim(
                        -1,
                        1,
                    )
            else:
                limit = max(
                    abs(minimum),
                    abs(maximum),
                )

                if limit > 0:
                    ax.set_ylim(
                        -1.12 * limit,
                        1.12 * limit,
                    )

        ax.axhline(
            0,
            color="#888888",
            linewidth=0.65,
            alpha=0.45,
            zorder=0,
        )
        return

    if scale == "unit_interval":
        # Jaccard distance and the normalized maximum coordination-number
        # change are bounded metrics. A linear axis keeps zero-valued cases
        # visible and gives the requested decimal tick labels.
        ax.set_yscale("linear")
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        return

    if scale == "log":
        positive = values[
            values > 0
        ]

        ax.set_yscale(
            "log",
            nonpositive="mask",
        )

        if len(positive):
            low = float(
                np.min(positive)
            )
            high = float(
                np.max(positive)
            )

            if high > low:
                ax.set_ylim(
                    low / 1.25,
                    high * 1.25,
                )

        return

    if len(values) == 0:
        return

    low = float(
        np.percentile(
            values,
            0.5,
        )
    )
    high = float(
        np.percentile(
            values,
            99.5,
        )
    )

    if np.min(values) >= 0:
        low = 0.0
    else:
        low = min(
            0.0,
            low,
        )

    if high > low:
        padding = 0.07 * (
            high - low
        )

        ax.set_ylim(
            low - (
                0.0
                if low == 0
                else padding
            ),
            high + padding,
        )
    elif high == low:
        padding = max(
            abs(high) * 0.1,
            1e-8,
        )

        ax.set_ylim(
            low - padding,
            high + padding,
        )

    if ax.get_ylabel().strip() == "coordination_change_max":
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(0.2)
        )
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter("%.1f")
        )


def configure_rdf_l1_axis(ax, values):
    """Configure an RDF L1 axis without hiding zeros or tiny distances.

    RDF L1 distances regularly include exact zeros alongside much larger
    values. A plain logarithmic axis masks the zeros, and the former fixed
    lower bound (10^-3) also clipped legitimate small values. Symlog keeps a
    readable linear region near zero while preserving dynamic range above it.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    positive = values[values > 0]
    maximum = float(np.max(positive)) if positive.size else 1.0

    # Put the lower quarter of non-zero observations in the linear region.
    # This makes close DFT/MLFF curves at larger attack strengths legible
    # without discarding the very small RDF distances.
    linthresh = (
        max(
            float(np.percentile(positive, 25)),
            maximum * 1e-6,
            1e-12,
        )
        if positive.size
        else 1e-12
    )

    upper_limit = maximum * 1.12 if maximum > 0 else 1.0

    ax.set_yscale(
        "symlog",
        linthresh=linthresh,
        linscale=1.2,
        base=10.0,
    )
    ax.set_ylim(0.0, upper_limit)

    # When every value is in the linear portion, use ordinary evenly spaced
    # labels. Otherwise, show the symlog decades explicitly, including zero.
    if upper_limit <= linthresh * 1.001:
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=5, min_n_ticks=4)
        )
        ax.yaxis.set_major_formatter(
            mticker.ScalarFormatter(useMathText=True)
        )
    else:
        ax.yaxis.set_major_locator(
            mticker.SymmetricalLogLocator(
                base=10.0,
                linthresh=linthresh,
                subs=(1.0,),
            )
        )
        ax.yaxis.set_major_formatter(
            mticker.LogFormatterMathtext(
                base=10.0,
                labelOnlyBase=False,
                linthresh=linthresh,
            )
        )
        ax.yaxis.set_minor_locator(
            mticker.SymmetricalLogLocator(
                base=10.0,
                linthresh=linthresh,
                subs=(2.0, 5.0),
            )
        )

    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    for line in list(ax.lines):
        if line.get_gid() == "rdf-zero-reference":
            line.remove()

    zero_line = ax.axhline(
        0.0,
        color="#888888",
        linewidth=0.65,
        alpha=0.45,
        zorder=0,
    )
    zero_line.set_gid("rdf-zero-reference")

def smooth_seed_summary(x, y, points=320, bandwidth_decades=0.16):
    """Smooth an aggregate curve in log-strength space for display only."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return x, y
    order = np.argsort(x)
    log_x, y = np.log10(x[order]), y[order]
    grid_log = np.linspace(log_x[0], log_x[-1], max(points, len(x)))
    trend = np.interp(grid_log, log_x, y)
    if len(x) > 2 and grid_log[-1] > grid_log[0]:
        step = grid_log[1] - grid_log[0]
        sigma = max(1.0, bandwidth_decades / step)
        radius = int(np.ceil(3.0 * sigma))
        offsets = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
        trend = np.convolve(
            np.pad(trend, radius, mode="edge"), kernel, mode="valid"
        )
    return 10 ** grid_log, trend

def draw_metric_panel(
    ax,
    records,
    metric,
    attack,
    y_scale="linear",
):
    curves = seed_curves(
        records,
        metric,
    )

    curves = curves[
        curves["attack_label"] == attack
    ].copy()

    aggregate = aggregate_curves(
        curves
    )

    plotted = []

    for calculator in CALCULATORS:
        calculator_curves = curves[
            curves["calculator"]
            == calculator
        ]

        if calculator_curves.empty:
            continue

        color = COLORS.get(
            calculator,
            "#777777",
        )

        raw = calculator_curves.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["epsilon_percent_displacement", "value"]
        )
        raw = raw[raw["epsilon_percent_displacement"] > 0]
        if not raw.empty:
            ax.scatter(
                raw["epsilon_percent_displacement"], raw["value"],
                s=17, color=color, alpha=0.46, edgecolors="white",
                linewidths=0.35, zorder=3,
            )
            plotted.extend(raw["value"].tolist())

        summary = aggregate[
            aggregate["calculator"]
            == calculator
        ].sort_values(
            "epsilon_percent_displacement"
        )

        summary = (
            summary.replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna(
                subset=[
                    "epsilon_percent_displacement",
                    "median",
                    "q25",
                    "q75",
                ]
            )
        )

        summary = summary[
            summary[
                "epsilon_percent_displacement"
            ] > 0
        ]

        if summary.empty:
            continue

        x = summary[
            "epsilon_percent_displacement"
        ].to_numpy(dtype=float)

        center = summary[
            "median"
        ].to_numpy(dtype=float)

        q25 = summary[
            "q25"
        ].to_numpy(dtype=float)

        q75 = summary[
            "q75"
        ].to_numpy(dtype=float)

        smooth_x, smooth_center = smooth_seed_summary(x, center)
        _, smooth_q25 = smooth_seed_summary(x, q25)
        _, smooth_q75 = smooth_seed_summary(x, q75)
        ax.fill_between(
            smooth_x,
            np.minimum(smooth_q25, smooth_q75),
            np.maximum(smooth_q25, smooth_q75),
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            smooth_x,
            smooth_center,
            color=color,
            linewidth=2.25,
            zorder=4,
        )

        plotted.extend(
            center.tolist()
        )
        plotted.extend(
            q25.tolist()
        )
        plotted.extend(
            q75.tolist()
        )

    if not plotted:
        message = (
            "Not applicable: contour starts\n"
            "from the relaxed reference"
            if attack == "Contour"
            and metric.startswith(
                "before_attack_after_relaxation__"
            )
            else "No matched seed data"
        )

        ax.text(
            0.5,
            0.5,
            message,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#555555",
        )

    positive_x = numeric(
        curves[
            "epsilon_percent_displacement"
        ]
    ).dropna()

    positive_x = positive_x[
        positive_x > 0
    ]

    if len(positive_x):
        ax.set_xscale(
            "log"
        )

    configure_y_axis(
        ax,
        plotted,
        scale=y_scale,
    )

    ax.set_title(
        attack,
        pad=7,
    )

    ax.grid(
        True,
        which="major",
        alpha=0.24,
        linewidth=0.7,
    )

    ax.grid(
        True,
        which="minor",
        alpha=0.08,
        linewidth=0.45,
    )

    ax.tick_params(
        axis="both",
        labelsize=8,
    )


def figure_legend(records):
    present_calculators = set(records["calculator"].dropna().astype(str))
    model_handles = [
        Line2D(
            [0], [0], color=COLORS.get(calculator, "#777777"),
            linewidth=2.7, label=model_label(calculator),
        )
        for calculator in CALCULATORS
        if calculator in present_calculators
    ]
    return model_handles, []

def make_metric_figure(
    records,
    metrics,
    output_path,
    title,
    panel_scales=None,
):
    """
    Create a 3x4 random-seed figure.

    panel_scales maps panel letters to "linear", "log" or "symlog".
    Force panels use symlog because force changes can contain
    exact zeros and values spanning several orders of magnitude.
    """
    panel_scales = dict(
        panel_scales or {}
    )

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(18.2, 10.4),
        squeeze=False,
    )

    # IMPORTANT:
    # save_exact_random_seed_panels() identifies the experimental
    # stage from the Figure-level title. The previous version
    # accepted `title` but never attached it to the figure, so
    # the only Figure-level text was the footer note.
    fig.suptitle(
        title,
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )

    for row, (
        metric,
        ylabel,
    ) in enumerate(metrics):
        for column, attack in enumerate(
            METHODS
        ):
            ax = axes[
                row,
                column,
            ]

            panel_label = chr(
                ord("A")
                + row * 4
                + column
            )

            draw_metric_panel(
                ax,
                records,
                metric,
                attack,
                y_scale=panel_scales.get(
                    panel_label,
                    "linear",
                ),
            )

            if column == 0:
                ax.set_ylabel(
                    ylabel,
                    labelpad=7,
                )

            ax.set_xlabel(
                r"$\epsilon$ strength (% min lattice parameter)",
                labelpad=6,
            )

            # Keep panel labels inside the axes so they cannot collide
            # with y-axis labels or scientific-notation offset text.
            ax.text(
                0.018,
                0.965,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
                color="#111111",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 1.5,
                },
                zorder=10,
            )

    model_legend_handles, _ = figure_legend(
        records
    )

    if model_legend_handles:
        fig.legend(
            handles=model_legend_handles,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.35, 0.955),
            fontsize=8.2,
            handlelength=2.5,
            columnspacing=1.2,
            handletextpad=0.5,
            borderaxespad=0.0,
            title="Models",
            title_fontsize=9.0,
        )

    fig.tight_layout(
        rect=[
            0.035,
            0.05,
            0.995,
            0.875,
        ],
        h_pad=2.0,
        w_pad=1.8,
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    harmonize_random_seed_axes(fig)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    save_exact_random_seed_panels(fig)

    plt.close(fig)


def configure_bubble_rdf_axis(ax, values):
    """Keep negligible RDF values in a readable near-zero symlog region."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    positive = values[values > 0]
    upper = float(np.max(positive)) * 1.15 if positive.size else 1.0

    ax.set_yscale("symlog", linthresh=1e-2, linscale=1.0, base=10.0)
    ax.set_ylim(0.0, max(upper, 1e-2 * 1.15))
    ax.yaxis.set_major_locator(
        mticker.SymmetricalLogLocator(base=10.0, linthresh=1e-2)
    )
    ax.yaxis.set_major_formatter(
        mticker.LogFormatterMathtext(base=10.0, labelOnlyBase=False, linthresh=1e-2)
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.axhline(0.0, color="#888888", linewidth=0.65, alpha=0.45, zorder=0)

def draw_bubble_metric_panel(ax, records, metric, attack, y_scale="linear"):
    """Draw one attack panel with seed variability encoded as bubbles."""
    curves = seed_curves(records, metric)
    curves = curves[curves["attack_label"] == attack].copy()
    aggregate = aggregate_curves(curves)
    plotted, spreads, summaries = [], [], []

    for calculator in CALCULATORS:
        summary = aggregate[aggregate["calculator"] == calculator].sort_values(
            "epsilon_percent_displacement"
        )
        summary = summary.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["epsilon_percent_displacement", "median", "q25", "q75"]
        )
        summary = summary[summary["epsilon_percent_displacement"] > 0].copy()
        if summary.empty:
            continue
        summary["spread"] = (summary["q75"] - summary["q25"]).abs()
        summaries.append((calculator, summary))
        plotted.extend(summary[["median", "q25", "q75"]].to_numpy().ravel())
        spreads.extend(summary["spread"].tolist())

    finite_spreads = np.asarray(spreads, dtype=float)
    finite_spreads = finite_spreads[np.isfinite(finite_spreads)]
    max_spread = float(np.max(finite_spreads)) if finite_spreads.size else 0.0
    for calculator, summary in summaries:
        color = COLORS.get(calculator, "#777777")
        x = summary["epsilon_percent_displacement"].to_numpy(dtype=float)
        y = summary["median"].to_numpy(dtype=float)
        if max_spread > 0:
            bubble_area = 70.0 + 650.0 * np.clip(
                summary["spread"].to_numpy(dtype=float) / max_spread, 0.0, 1.0
            )
        else:
            bubble_area = np.full(len(summary), 120.0)
        ax.scatter(x, y, s=bubble_area, color=color, alpha=0.14,
                   edgecolors=color, linewidths=0.8, zorder=2)
        ax.scatter(x, y, s=22, color=color, edgecolors="white",
                   linewidths=0.45, zorder=4)

    if not summaries:
        ax.text(0.5, 0.5, "No matched seed data", transform=ax.transAxes,
                ha="center", va="center", color="#555555")
    positive_x = numeric(curves["epsilon_percent_displacement"]).dropna()
    if len(positive_x[positive_x > 0]):
        ax.set_xscale("log")
    if y_scale == "rdf_symlog":
        configure_bubble_rdf_axis(ax, plotted)
    else:
        configure_y_axis(ax, plotted, scale=y_scale)
    ax.set_title(attack, pad=7)
    ax.grid(True, which="major", alpha=0.24, linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.45)
    ax.tick_params(axis="both", labelsize=8)


def make_bubble_metric_figure(records, metrics, output_path, title, panel_scales=None):
    """Save a random-seed summary as bubbles, with no seed-trace lines."""
    panel_scales = dict(panel_scales or {})
    attacks = ADVERSARIAL_ATTACKS
    fig, axes = plt.subplots(len(metrics), len(attacks),
                             figsize=(13.6, max(4.2, 2.75 * len(metrics))),
                             squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    for row, (metric, ylabel) in enumerate(metrics):
        for column, attack in enumerate(attacks):
            ax = axes[row, column]
            panel_label = chr(ord("A") + row * len(attacks) + column)
            draw_bubble_metric_panel(ax, records, metric, attack,
                                     panel_scales.get(panel_label, "linear"))
            if column == 0:
                ax.set_ylabel(ylabel, labelpad=7)
            ax.set_xlabel(r"$\epsilon$ strength (% min lattice parameter)", labelpad=6)
            ax.text(0.018, 0.965, panel_label, transform=ax.transAxes,
                    ha="left", va="top", fontsize=9, fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "alpha": 0.82, "pad": 1.5}, zorder=10)

    model_handles, _ = figure_legend(records)
    if model_handles:
        fig.legend(handles=model_handles, loc="upper center",
                   ncol=min(4, len(model_handles)), frameon=False,
                   bbox_to_anchor=(0.5, 0.965), fontsize=8.5,
                   title="Model", title_fontsize=9.0)
    fig.text(0.5, 0.012,
             "Dot: median across seeds; bubble area: seed interquartile range",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0.035, 0.04, 0.995, 0.90], h_pad=2.0, w_pad=1.8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_bubble_plots(records, output_dir):
    """Export final post-attack/relaxation random-seed bubble plots."""
    output_dir = Path(output_dir) / "bubble_plots"
    final_stage = "after_attack_after_relaxation"
    metrics = physical_metrics(final_stage) + topology_metrics_for_stage(final_stage)
    scales = {
        **{letter: "symlog" for letter in "DEF"},
        **{letter: "unit_interval" for letter in "JKLPQR"},
        **{letter: "rdf_symlog" for letter in "MNO"},
    }
    make_bubble_metric_figure(
        records, metrics,
        output_dir / "seed_response_comprehensive_after_attack_after_relaxation_bubble.png",
        "Random-seed bubble comparison: after attack and relaxation",
        panel_scales=scales,
    )
    topology = {
        stage_column(final_stage, "neighbor_jaccard_distance"),
        stage_column(final_stage, "rdf_l1_distance"),
        stage_column(final_stage, "coordination_change_max"),
    }
    for metric, label in metrics:
        slug = metric.split("__", 1)[-1]
        if "delta_force" in metric:
            metric_scales = {letter: "symlog" for letter in "ABC"}
        elif metric == stage_column(final_stage, "rdf_l1_distance"):
            metric_scales = {letter: "rdf_symlog" for letter in "ABC"}
        elif metric in topology:
            metric_scales = {letter: "unit_interval" for letter in "ABC"}
        else:
            metric_scales = {}
        make_bubble_metric_figure(
            records, [(metric, label)],
            output_dir / f"seed_response_{slug}_after_attack_after_relaxation_bubble.png",
            f"Random-seed bubble comparison: {label} after attack and relaxation",
            panel_scales=metric_scales,
        )

def write_aggregate_table(records, output_path):
    tables = []

    for stage in STAGES:
        metrics = (
            physical_metrics(stage)
            + topology_metrics_for_stage(stage)
        )

        for metric, _ in metrics:
            aggregate = aggregate_curves(
                seed_curves(records, metric)
            )

            if aggregate.empty:
                continue

            aggregate["stage"] = stage
            aggregate["metric"] = metric
            tables.append(aggregate)

    result = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame()
    )

    result.to_csv(
        output_path,
        index=False,
    )


def save_exact_random_seed_panels(fig):
    """
    Save exact crops of the twelve axes displayed in each random-seed
    comprehensive figure.

    Six comprehensive figures are organized as:

        3 experimental stages
        x
        2 figure families: physical and topology

    Each stage folder therefore receives 24 panels.
    """

    # The comprehensive figure has already been saved. Increase only
    # the physical height used for standalone panel exports by 40%.
    fig.set_size_inches(
        fig.get_figwidth(),
        fig.get_figheight() * 1.40,
        forward=True,
    )

    def get_output_directory():
        arguments = sys.argv[1:]

        for index, argument in enumerate(arguments):
            if argument == "--output-dir":
                if index + 1 >= len(arguments):
                    raise RuntimeError(
                        "--output-dir has no value"
                    )

                return Path(
                    arguments[index + 1]
                ).resolve()

            if argument.startswith("--output-dir="):
                return Path(
                    argument.split("=", 1)[1]
                ).resolve()

        raise RuntimeError(
            "random_seed_comprehensive.py must be called "
            "with --output-dir"
        )

    def slugify(value):
        value = value.strip().lower()

        replacements = {
            "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â": "delta",
            "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â´": "delta",
            "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦": "angstrom",
            "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥": "angstrom",
            "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²": "2",
            "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³": "3",
        }

        for old, new in replacements.items():
            value = value.replace(
                old,
                new,
            )

        value = re.sub(
            r"[^a-z0-9]+",
            "_",
            value,
        )

        return value.strip("_")

    def individual_y_values(axis):
        """Collect data-space y values from one plotted panel."""
        values = []

        for line in axis.lines:
            current = np.asarray(
                line.get_ydata(),
                dtype=float,
            ).reshape(-1)
            current = current[np.isfinite(current)]

            if current.size:
                values.append(current)

        for collection in axis.collections:
            try:
                offsets = np.asarray(
                    collection.get_offsets(),
                    dtype=float,
                )

                if (
                    offsets.ndim == 2
                    and offsets.shape[1] >= 2
                ):
                    current = offsets[:, 1]
                    current = current[np.isfinite(current)]

                    if current.size:
                        values.append(current)
            except Exception:
                pass

            try:
                for path_item in collection.get_paths():
                    vertices = np.asarray(
                        path_item.vertices,
                        dtype=float,
                    )

                    if (
                        vertices.ndim == 2
                        and vertices.shape[1] >= 2
                    ):
                        current = vertices[:, 1]
                        current = current[np.isfinite(current)]

                        if current.size:
                            values.append(current)
            except Exception:
                pass

        if not values:
            return np.asarray([], dtype=float)

        return np.concatenate(values)

    plotting_axes = [
        axis
        for axis in fig.axes
        if axis.get_visible()
    ]

    plotting_axes = sorted(
        plotting_axes,
        key=lambda axis: (
            -axis.get_position().y0,
            axis.get_position().x0,
        ),
    )

    if len(plotting_axes) != 12:
        raise RuntimeError(
            "Expected 12 axes in the random-seed comprehensive "
            f"figure, but found {len(plotting_axes)}"
        )

    # Determine the comprehensive figure title.
    #
    # Most figures use fig.suptitle(), which is stored in
    # fig._suptitle. Some of the random-seed figures, however,
    # create the title as a regular Figure-level Text artist.
    # In that case _suptitle is None even though the figure has
    # a visible title.
    figure_title = ""

    if fig._suptitle is not None:
        figure_title = (
            fig._suptitle.get_text()
            .strip()
            .lower()
            .replace("-", " ")
        )

    # Fall back to Figure-level text artists when the title was
    # not created with fig.suptitle().
    if not figure_title:
        figure_text_candidates = [
            text_artist.get_text().strip()
            for text_artist in fig.texts
            if text_artist.get_text().strip()
        ]

        # Prefer text containing an experimental-stage phrase.
        stage_phrases = (
            "pre relaxation",
            "before attack",
            "after attack",
            "immediate",
            "post relaxation",
            "after relaxation",
            "attack and relaxation",
        )

        for candidate in figure_text_candidates:
            candidate_normalized = (
                candidate.lower()
                .replace("-", " ")
            )

            if any(
                phrase in candidate_normalized
                for phrase in stage_phrases
            ):
                figure_title = candidate_normalized
                break

    # If the figure genuinely contains no identifiable title,
    # provide the available Figure-level text in the error so the
    # actual title source can be located instead of failing blindly.
    if not figure_title:
        raise RuntimeError(
            "The random-seed figure has no identifiable title, "
            "so its experimental stage cannot be identified.\n"
            f"Figure text artists: {figure_text_candidates!r}"
        )

    axis_label_text = " ".join(
        axis.get_ylabel().strip().lower()
        for axis in plotting_axes
        if axis.get_ylabel().strip()
    )

    classification_text = (
        figure_title
        + " "
        + axis_label_text
    )

    # Classify the figure as physical or topology.
    if any(
        phrase in classification_text
        for phrase in (
            "displacement",
            "force",
            "relaxation step",
            "relax steps",
            "physical response",
        )
    ):
        figure_family = "physical"

    elif any(
        phrase in classification_text
        for phrase in (
            "jaccard",
            "rdf",
            "coordination",
            "topology",
            "neighbor",
            "neighbour",
        )
    ):
        figure_family = "topology"

    else:
        raise RuntimeError(
            "Could not classify the figure as physical or "
            "topology.\n"
            f"Figure title: {figure_title!r}\n"
            f"Axis labels: {axis_label_text!r}"
        )

    # Classify the experimental stage.
    if (
        "pre relaxation" in figure_title
        or (
            "before attack" in figure_title
            and "after relaxation" in figure_title
        )
    ):
        stage = "before_attack_after_relaxation"

    elif (
        "after attack" in figure_title
        and "before relaxation" in figure_title
    ):
        stage = "after_attack_before_relaxation"

    elif (
        "immediate" in figure_title
        and "after attack" in figure_title
    ):
        stage = "after_attack_before_relaxation"

    elif (
        "after attack and relaxation" in figure_title
        or (
            "after attack" in figure_title
            and "post relaxation" in figure_title
        )
    ):
        stage = "after_attack_after_relaxation"

    else:
        raise RuntimeError(
            "Could not identify the experimental stage from "
            f"the figure title: {figure_title!r}"
        )

    stage_directory = (
        get_output_directory()
        / stage
    )

    stage_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_metric_names = []

    for row_index in range(3):
        row_axes = plotting_axes[
            row_index * 4:
            (row_index + 1) * 4
        ]

        metric_name = next(
            (
                axis.get_ylabel().strip()
                for axis in row_axes
                if axis.get_ylabel().strip()
            ),
            "",
        )

        if not metric_name:
            raise RuntimeError(
                "Could not identify the metric for row "
                f"{row_index + 1}"
            )

        row_metric_names.append(
            metric_name
        )

    attack_names = (
        "contour",
        "fgsm",
        "ifgsm",
        "pgd",
    )

    panel_letters = "ABCDEFGHIJKL"

    # Reuse the comprehensive figure legend in every individual panel.
    individual_legend_handles = []
    individual_legend_labels = []

    if fig.legends:
        source_legend = fig.legends[0]

        individual_legend_handles = getattr(
            source_legend,
            "legend_handles",
            None,
        )

        if individual_legend_handles is None:
            individual_legend_handles = getattr(
                source_legend,
                "legendHandles",
                [],
            )

        individual_legend_labels = [
            item.get_text()
            for item in source_legend.get_texts()
            if item.get_text() != "Available-seed median"
        ]

        paired_items = [
            (handle, label)
            for handle, label in zip(
                individual_legend_handles,
                [
                    item.get_text()
                    for item in source_legend.get_texts()
                ],
            )
            if label != "Available-seed median"
        ]

        individual_legend_handles = [
            item[0]
            for item in paired_items
        ]

        individual_legend_labels = [
            item[1]
            for item in paired_items
        ]

    model_legend_items = [
        (handle, label)
        for handle, label in zip(
            individual_legend_handles,
            individual_legend_labels,
        )
        if not label.startswith("Seed ")
    ]

    seed_legend_items = [
        (handle, label)
        for handle, label in zip(
            individual_legend_handles,
            individual_legend_labels,
        )
        if label.startswith("Seed ")
    ]

    # The comprehensive figure has already been saved. Hide its global
    # legend, title and footer so they cannot leak into panel crops.
    for figure_legend_artist in fig.legends:
        figure_legend_artist.set_visible(False)

    for figure_text_artist in fig.texts:
        figure_text_artist.set_visible(False)

    # Render the comprehensive layout before calculating crop boxes.
    fig.canvas.draw()

    renderer = (
        fig.canvas.get_renderer()
    )

    for panel_index, axis in enumerate(
        plotting_axes
    ):
        row_index = panel_index // 4
        column_index = panel_index % 4

        metric_slug = slugify(
            row_metric_names[row_index]
        )

        attack_slug = (
            attack_names[column_index]
        )

        output_filename = (
            f"{figure_family}_"
            f"{panel_letters[panel_index]}_"
            f"{metric_slug}_"
            f"{attack_slug}.png"
        )

        output_path = (
            stage_directory
            / output_filename
        )

        # Hide all other axes before saving this individual panel.
        # This prevents neighbouring ticks, curves and labels from
        # appearing inside the expanded legend crop.
        for other_axis in plotting_axes:
            other_axis.set_visible(
                other_axis is axis
            )

        # Remove the A-L panel letter from the individual output.
        for text_artist in axis.texts:
            if (
                text_artist.get_text().strip()
                == panel_letters[panel_index]
            ):
                text_artist.set_visible(False)

        # Every standalone image needs complete axis labels, even when
        # its source panel was not in the leftmost or bottom row.
        axis.set_ylabel(
            row_metric_names[row_index],
            labelpad=7,
        )

        axis.set_xlabel(
            r"$\epsilon$ strength (% min lattice parameter)",
            labelpad=7,
        )

        # The comprehensive figure was already saved with harmonized
        # row limits. Only standalone panels are rescaled here using
        # the data actually visible in the current axis.
        panel_values = individual_y_values(axis)

        is_neighbor_jaccard = (
            "neighbor jaccard" in row_metric_names[row_index].lower()
        )

        if is_neighbor_jaccard:
            # Smoothing can slightly undershoot a zero-valued Jaccard
            # curve. The metric is bounded below by zero, so retain that
            # physical bound in each exported standalone panel.
            axis.set_yscale("linear")
            axis.set_ylim(0.0, 0.8)
            axis.set_yticks(np.arange(0.0, 0.81, 0.2))
            axis.yaxis.set_major_formatter(
                mticker.FormatStrFormatter("%.1f")
            )
        elif panel_values.size:
            configure_y_axis(
                axis,
                panel_values,
                scale=axis.get_yscale(),
            )

        # RDF panels use the same zero-safe symlog treatment as the
        # comprehensive figure, so extracted panels cannot clip tiny values.
        if "rdf" in row_metric_names[row_index].lower():
            configure_rdf_l1_axis(axis, panel_values)
        # Standalone linear panels must include every visible seed and
        # IQR value. The general helper uses robust percentiles, which
        # is useful for comprehensive figures but can crop the largest
        # value in a single extracted panel. Round upward to the next
        # clean major tick, with an extra tick when the maximum lies
        # exactly on a boundary.
        if (
            panel_values.size
            and axis.get_yscale() == "linear"
            and row_metric_names[row_index] != r"Max $\Delta$ CN"
            and not is_neighbor_jaccard
        ):
            finite_panel_values = panel_values[
                np.isfinite(panel_values)
            ]

            if finite_panel_values.size:
                panel_minimum = float(
                    np.min(finite_panel_values)
                )
                panel_maximum = float(
                    np.max(finite_panel_values)
                )

                panel_lower_limit = (
                    0.0
                    if panel_minimum >= 0
                    else panel_minimum
                )

                locator = mticker.MaxNLocator(
                    nbins=6,
                    min_n_ticks=4,
                    steps=[1, 2, 2.5, 4, 5, 10],
                )

                shared_ticks = locator.tick_values(
                    panel_lower_limit,
                    panel_maximum,
                )

                if panel_minimum >= 0:
                    shared_ticks = shared_ticks[
                        shared_ticks >= 0
                    ]

                    if (
                        shared_ticks.size == 0
                        or not np.isclose(
                            shared_ticks[0],
                            0.0,
                        )
                    ):
                        shared_ticks = np.insert(
                            shared_ticks,
                            0,
                            0.0,
                        )

                if shared_ticks.size >= 2:
                    tick_step = float(
                        shared_ticks[-1]
                        - shared_ticks[-2]
                    )

                    if (
                        shared_ticks[-1]
                        <= panel_maximum
                        or np.isclose(
                            shared_ticks[-1],
                            panel_maximum,
                        )
                    ):
                        shared_ticks = np.append(
                            shared_ticks,
                            shared_ticks[-1] + tick_step,
                        )

                    axis.set_ylim(
                        float(shared_ticks[0]),
                        float(shared_ticks[-1]),
                    )
                    axis.set_yticks(shared_ticks)
        # Attack relaxation runs are capped at 300/600 steps. Keep a small
        # amount of headroom above capped curves, but do not display a
        # misleading 700-step tick. Contour panels retain their own
        # independently scaled axes.
        if (
            "relaxation steps"
            in row_metric_names[row_index].lower()
        ):
            axis.set_yscale("linear")
            axis.set_ylim(0, RELAXATION_STEP_UPPER_LIMIT)
            axis.set_yticks(
                np.arange(0, RELAXATION_STEP_UPPER_LIMIT + 1, 100)
            )


        # Keep one-change panels readable without displaying an unreached 2.
        if row_metric_names[row_index] == r"Max $\Delta$ CN":
            line_value_sets = []

            for line in axis.lines:
                values = np.asarray(line.get_ydata(), dtype=float).reshape(-1)
                values = values[np.isfinite(values)]

                if values.size:
                    line_value_sets.append(values)

            maximum_cn_change = (
                float(np.max(np.concatenate(line_value_sets)))
                if line_value_sets
                else 10.0
            )
            axis.set_yscale("linear")

            if maximum_cn_change <= 1.0:
                axis.set_ylim(0.0, 1.1)
                axis.set_yticks(np.arange(0.0, 1.01, 0.2))
            else:
                panel_cn_upper_limit = float(math.ceil(maximum_cn_change + 0.1))
                axis.set_ylim(0.0, panel_cn_upper_limit)
                axis.yaxis.set_major_locator(
                    mticker.MaxNLocator(nbins=6, min_n_ticks=5)
                )
        panel_legends = []

        if model_legend_items:
            model_legend = axis.legend(
                handles=[
                    item[0]
                    for item in model_legend_items
                ],
                labels=[
                    item[1]
                    for item in model_legend_items
                ],
                loc="lower center",
                bbox_to_anchor=(0.28, 1.30),
                ncol=2,
                frameon=False,
                fontsize=7.2,
                handlelength=2.2,
                columnspacing=1.0,
                handletextpad=0.45,
                borderaxespad=0.0,
                title="Models",
                title_fontsize=7.4,
            )

            axis.add_artist(model_legend)
            panel_legends.append(model_legend)

        if seed_legend_items:
            seed_legend = axis.legend(
                handles=[
                    item[0]
                    for item in seed_legend_items
                ],
                labels=[
                    item[1]
                    for item in seed_legend_items
                ],
                loc="lower center",
                bbox_to_anchor=(0.78, 1.30),
                ncol=2,
                frameon=False,
                fontsize=7.2,
                handlelength=2.2,
                columnspacing=1.0,
                handletextpad=0.45,
                borderaxespad=0.0,
                title="Seeds",
                title_fontsize=7.4,
            )

            panel_legends.append(seed_legend)

        # Redraw so the legend has a valid bounding box.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        bounding_boxes = [
            axis.get_tightbbox(renderer),
        ]

        for panel_legend in panel_legends:
            bounding_boxes.append(
                panel_legend.get_window_extent(renderer)
            )

        bounding_box = (
            matplotlib.transforms.Bbox.union(
                bounding_boxes
            )
            .transformed(
                fig.dpi_scale_trans.inverted()
            )
            .padded(0.06)
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches=bounding_box,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )

        for panel_legend in panel_legends:
            panel_legend.remove()


def harmonize_random_seed_axes(fig):
    def finite_values(values):
        array = np.asarray(values, dtype=float).reshape(-1)
        return array[np.isfinite(array)]

    figure_title_text = ""
    if fig._suptitle is not None:
        figure_title_text = fig._suptitle.get_text().strip().lower()

    final_after_relaxation_figure = (
        "after attack and relaxation" in figure_title_text
    )

    def collect_axis_values(ax, coordinate):
        values = []

        for line in ax.lines:
            if coordinate == "x":
                current = finite_values(line.get_xdata())
            else:
                current = finite_values(line.get_ydata())

            if current.size:
                values.append(current)

        for collection in ax.collections:
            try:
                offsets = np.asarray(
                    collection.get_offsets(),
                    dtype=float,
                )

                if (
                    offsets.ndim == 2
                    and offsets.shape[1] >= 2
                    and offsets.size
                ):
                    current = finite_values(
                        offsets[:, 0 if coordinate == "x" else 1]
                    )

                    if current.size:
                        values.append(current)
            except Exception:
                pass

            try:
                for path in collection.get_paths():
                    vertices = np.asarray(
                        path.vertices,
                        dtype=float,
                    )

                    if (
                        vertices.ndim == 2
                        and vertices.shape[1] >= 2
                        and vertices.size
                    ):
                        current = finite_values(
                            vertices[
                                :,
                                0 if coordinate == "x" else 1
                            ]
                        )

                        if current.size:
                            values.append(current)
            except Exception:
                pass

        if not values:
            return np.asarray([], dtype=float)

        return np.concatenate(values)


    def next_power_of_ten(value):
        """Round upward to the next exact power of ten."""
        if not np.isfinite(value) or value <= 0:
            return 1.0

        return 10.0 ** math.ceil(math.log10(value))

    # Exclude empty/helper axes and group plotting axes by figure row.
    plotting_axes = [
        ax
        for ax in fig.axes
        if ax.get_visible() and ax.has_data()
    ]

    rows = []

    for ax in sorted(
        plotting_axes,
        key=lambda current: (
            -current.get_position().y0,
            current.get_position().x0,
        ),
    ):
        y_position = ax.get_position().y0

        matching_row = None

        for row in rows:
            if abs(row["y_position"] - y_position) < 0.03:
                matching_row = row
                break

        if matching_row is None:
            matching_row = {
                "y_position": y_position,
                "axes": [],
            }
            rows.append(matching_row)

        matching_row["axes"].append(ax)

    for row in rows:
        all_row_axes = sorted(
            row["axes"],
            key=lambda current: current.get_position().x0,
        )

        # Use mathematical powers of ten on every logarithmic axis,
        # including the independently scaled Contour panels.
        for current_axis in all_row_axes:
            if current_axis.get_xscale() == "log":
                current_axis.xaxis.set_major_formatter(
                    mticker.LogFormatterMathtext(
                        base=10.0,
                        labelOnlyBase=False,
                    )
                )

                current_axis.xaxis.set_minor_formatter(
                    mticker.NullFormatter()
                )

            if current_axis.get_yscale() == "log":
                current_axis.yaxis.set_major_locator(
                    mticker.LogLocator(
                        base=10.0,
                        subs=(1.0, 2.0, 5.0),
                        numticks=8,
                    )
                )

                current_axis.yaxis.set_major_formatter(
                    mticker.LogFormatterMathtext(
                        base=10.0,
                        labelOnlyBase=False,
                    )
                )

                current_axis.yaxis.set_minor_locator(
                    mticker.LogLocator(
                        base=10.0,
                        subs=np.arange(2, 10) * 0.1,
                        numticks=100,
                    )
                )

                current_axis.yaxis.set_minor_formatter(
                    mticker.NullFormatter()
                )

        # Contour displacement is measured rather than prescribed.
        # Preserve its independently configured x- and y-axis ranges;
        # only the three adversarial methods share row limits.
        row_axes = [
            ax
            for ax in all_row_axes
            if ax.get_title().strip().lower() != "contour"
        ]

        if len(row_axes) < 2:
            continue

        # Make every x-axis in this row identical.
        x_limits = [
            ax.get_xlim()
            for ax in row_axes
        ]

        common_x_min = min(
            min(limits)
            for limits in x_limits
        )
        common_x_max = max(
            max(limits)
            for limits in x_limits
        )

        row_uses_log_x = any(
            ax.get_xscale() == "log"
            for ax in row_axes
        )

        for ax in row_axes:
            if row_uses_log_x:
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(
                    mticker.LogLocator(base=10.0)
                )
                ax.xaxis.set_minor_locator(
                    mticker.LogLocator(
                        base=10.0,
                        subs=np.arange(2, 10) * 0.1,
                    )
                )
                ax.xaxis.set_major_formatter(
                    mticker.LogFormatterMathtext(
                        base=10.0,
                        labelOnlyBase=False,
                    )
                )

                ax.xaxis.set_minor_formatter(
                    mticker.NullFormatter()
                )

            ax.set_xlim(
                common_x_min,
                common_x_max,
            )

        row_label = " ".join(
            ax.get_ylabel().lower()
            for ax in all_row_axes
            if ax.get_ylabel()
        )

        y_values = [
            collect_axis_values(ax, "y")
            for ax in row_axes
        ]

        y_values = [
            values
            for values in y_values
            if values.size
        ]

        if y_values:
            combined_y = np.concatenate(y_values)
        else:
            combined_y = np.asarray([], dtype=float)

        positive_y = combined_y[
            combined_y > 0
        ]

        is_displacement = (
            "displacement" in row_label
        )

        is_delta_force = (
            "force" in row_label
            and (
                "delta" in row_label

                or "\u0394" in row_label

                or "\u03b4" in row_label
            )
        )

        is_relaxation_steps = (
            "relaxation step" in row_label
            or "relax steps" in row_label
        )

        is_neighbor_jaccard = (
            "neighbor jaccard" in row_label
        )

        is_unit_interval = any(
            phrase in row_label
            for phrase in (
                "jaccard",
                "retention",
                "fraction",
            )
        )

        is_rdf = "rdf" in row_label

        use_log_y_for_topology = (
            final_after_relaxation_figure
            and (
                is_unit_interval
                or is_rdf
            )
        )

        if is_delta_force:
            nonzero = np.abs(
                combined_y[
                    combined_y != 0
                ]
            )

            if nonzero.size:
                linthresh = max(
                    float(
                        np.percentile(
                            nonzero,
                            10,
                        )
                    ),
                    float(
                        np.max(nonzero)
                    ) * 1e-8,
                    1e-12,
                )
            else:
                linthresh = 1e-12

            for ax in row_axes:
                ax.set_yscale(
                    "symlog",
                    linthresh=linthresh,
                    linscale=1.0,
                    base=10.0,
                )

                if combined_y.size:
                    minimum = float(
                        np.min(combined_y)
                    )
                    maximum = float(
                        np.max(combined_y)
                    )

                    if minimum >= 0:
                        upper = (
                            maximum * 1.15
                            if maximum > 0
                            else 1.0
                        )

                        ax.set_ylim(
                            0.0,
                            upper,
                        )
                    else:
                        limit = max(
                            abs(minimum),
                            abs(maximum),
                        )

                        ax.set_ylim(
                            -1.15 * limit,
                            1.15 * limit,
                        )

                ax.yaxis.set_major_locator(
                    mticker.SymmetricalLogLocator(
                        linthresh=linthresh,
                        base=10.0,
                    )
                )

                ax.yaxis.set_major_formatter(
                    mticker.LogFormatterMathtext(
                        base=10.0,
                        labelOnlyBase=True,
                        linthresh=linthresh,
                    )
                )

                ax.yaxis.set_minor_formatter(
                    mticker.NullFormatter()
                )

                ax.axhline(
                    0.0,
                    color="#888888",
                    linewidth=0.65,
                    alpha=0.45,
                    zorder=0,
                )

        elif is_displacement:
            if positive_y.size:
                minimum_positive = float(
                    np.min(positive_y)
                )
                maximum_positive = float(
                    np.max(positive_y)
                )

                lower_limit = (
                    10.0
                    ** math.floor(
                        math.log10(
                            minimum_positive
                        )
                    )
                )

                upper_limit = (
                    10.0
                    ** math.ceil(
                        math.log10(
                            maximum_positive
                        )
                    )
                )

                if np.isclose(
                    lower_limit,
                    upper_limit,
                ):
                    upper_limit *= 10.0

                for ax in row_axes:
                    ax.set_yscale("log")
                    ax.set_ylim(
                        lower_limit,
                        upper_limit,
                    )

                    ax.yaxis.set_major_locator(
                        mticker.LogLocator(
                            base=10.0,
                            subs=(1.0, 2.0, 5.0),
                            numticks=8,
                        )
                    )

                    ax.yaxis.set_major_formatter(
                        mticker.LogFormatterMathtext(
                            base=10.0,
                            labelOnlyBase=False,
                        )
                    )

                    ax.yaxis.set_minor_locator(
                        mticker.LogLocator(
                            base=10.0,
                            subs=np.arange(
                                2.0,
                                10.0,
                            ) * 0.1,
                            numticks=100,
                        )
                    )

                    ax.yaxis.set_minor_formatter(
                        mticker.NullFormatter()
                    )

        # Change to 300 or 600
        elif is_relaxation_steps:
            upper_limit = RELAXATION_STEP_UPPER_LIMIT

            shared_ticks = np.arange(
                0.0,
                upper_limit + 1.0,
                100.0,
            )

            for ax in all_row_axes:
                ax.set_yscale("linear")
                ax.set_ylim(
                    0.0,
                    upper_limit,
                )
                ax.set_yticks(shared_ticks)

        elif is_neighbor_jaccard:
            for ax in all_row_axes:
                ax.set_yscale("linear")
                ax.set_ylim(0.0, 0.8)
                ax.set_yticks(
                    np.arange(0.0, 0.81, 0.2)
                )

        elif is_rdf:
            # Apply one scale across the full RDF row, including Contour,
            # so no panel masks zeros or cuts off small RDF distances.
            for ax in all_row_axes:
                configure_rdf_l1_axis(ax, combined_y)

        elif "max" in row_label and "cn" in row_label:
            upper_limit = max(
                1.0,
                float(math.ceil(np.max(combined_y)))
                if combined_y.size
                else 1.0,
            )

            for ax in all_row_axes:
                ax.set_yscale("linear")
                ax.set_ylim(0.0, upper_limit)
                ax.set_yticks(
                    np.arange(
                        0.0,
                        upper_limit + 1.0,
                        1.0,
                    )
                )

        elif is_unit_interval:
            if positive_y.size:

                minimum = float(np.min(positive_y))
                maximum = float(np.max(positive_y))

                lower_limit = 10 ** math.floor(
                    math.log10(minimum)
                )

                upper_limit = 10 ** math.ceil(
                    math.log10(maximum)
                )

                if np.isclose(lower_limit, upper_limit):
                    upper_limit = lower_limit * 10.0

                for ax in row_axes:
                    ax.set_yscale("log")

                    ax.set_ylim(
                        lower_limit,
                        upper_limit,
                    )

                    ax.yaxis.set_major_locator(
                        mticker.LogLocator(
                            base=10,
                            subs=(1.0, 2.0, 5.0),
                            numticks=8,
                        )
                    )

                    ax.yaxis.set_major_formatter(
                        mticker.LogFormatterMathtext(
                            base=10,
                            labelOnlyBase=False,
                        )
                    )

                    ax.yaxis.set_minor_locator(
                        mticker.LogLocator(
                            base=10,
                            subs=np.arange(2, 10) * 0.1,
                        )
                    )

                    ax.yaxis.set_minor_formatter(
                        mticker.NullFormatter()
                    )

        elif use_log_y_for_topology:
            if positive_y.size:
                minimum = float(np.min(positive_y))
                maximum = float(np.max(positive_y))

                lower_limit = 10 ** math.floor(
                    math.log10(minimum)
                )
                upper_limit = 10 ** math.ceil(
                    math.log10(maximum)
                )

                if np.isclose(lower_limit, upper_limit):
                    upper_limit = lower_limit * 10.0

                for ax in row_axes:
                    ax.set_yscale("log")
                    ax.set_ylim(
                        lower_limit,
                        upper_limit,
                    )

                    ax.yaxis.set_major_locator(
                        mticker.LogLocator(
                            base=10,
                            subs=(1.0, 2.0, 5.0),
                            numticks=8,
                        )
                    )
                    ax.yaxis.set_major_formatter(
                        mticker.LogFormatterMathtext(
                            base=10,
                            labelOnlyBase=False,
                        )
                    )
                    ax.yaxis.set_minor_locator(
                        mticker.LogLocator(
                            base=10,
                            subs=np.arange(2, 10) * 0.1,
                        )
                    )
                    ax.yaxis.set_minor_formatter(
                        mticker.NullFormatter()
                    )
            else:
                for ax in row_axes:
                    ax.set_yscale("linear")

        else:
            current_limits = [
                ax.get_ylim()
                for ax in row_axes
            ]

            data_minimum = (
                float(np.min(combined_y))
                if combined_y.size
                else min(
                    limits[0]
                    for limits in current_limits
                )
            )

            data_maximum = (
                float(np.max(combined_y))
                if combined_y.size
                else max(
                    limits[1]
                    for limits in current_limits
                )
            )

            if data_minimum >= 0:
                lower_limit = 0.0
            else:
                lower_limit = min(
                    limits[0]
                    for limits in current_limits
                )

            locator = mticker.MaxNLocator(
                nbins=6,
                min_n_ticks=4,
            )

            shared_ticks = locator.tick_values(
                lower_limit,
                data_maximum,
            )

            if data_minimum >= 0:
                shared_ticks = shared_ticks[
                    shared_ticks >= 0
                ]

                if (
                    shared_ticks.size == 0
                    or shared_ticks[0] != 0
                ):
                    shared_ticks = np.insert(
                        shared_ticks,
                        0,
                        0.0,
                    )

            shared_lower = float(
                shared_ticks[0]
            )
            shared_upper = float(
                shared_ticks[-1]
            )

            for ax in row_axes:
                ax.set_yscale("linear")
                ax.set_ylim(
                    shared_lower,
                    shared_upper,
                )
                ax.set_yticks(shared_ticks)



def direct_final_atoms(row):
    """Load a final post-attack relaxed structure for MLFF or DFT data."""
    is_dft = str(row.get("calculator", "")).startswith("dft_")
    if is_dft:
        saved = row.get("dft_relaxed_structure")
        if saved is not None and not pd.isna(saved) and str(saved).strip():
            path = Path(str(saved))
            if not path.is_absolute():
                path = Path(str(row.get("run_dir", ""))) / path
        else:
            path = resolve_record_artifact(
                row, "after_attack_relax_traj", "after_attack_relaxation.traj"
            )
        try:
            return read_dft_structure(path, index=-1)
        except Exception:
            return ase_read(path, index=-1)

    path = resolve_record_artifact(
        row, "after_attack_relax_traj", "after_attack_relaxation.traj"
    )
    return ase_read(path, index=-1)


def direct_final_force_vectors(row):
    """Return final forces indexed by atom, or ``None`` when unavailable."""
    after_path = resolve_force_artifacts(row)[2]
    data = read_force_csv(after_path)
    if data is None:
        return None
    vectors = data[["fx", "fy", "fz"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    atom_indices = pd.to_numeric(data["atom_index"], errors="coerce").to_numpy()
    if (
        vectors.ndim != 2
        or vectors.shape[1] != 3
        or not np.isfinite(vectors).all()
        or not np.isfinite(atom_indices).all()
        or len(np.unique(atom_indices)) != len(atom_indices)
    ):
        return None
    return {
        int(atom_index): vector
        for atom_index, vector in zip(atom_indices, vectors)
    }


def direct_final_force_rms_difference(mlff_row, dft_row):
    """Return the RMS per-atom MLFF--DFT final-force vector difference."""
    mlff_forces = direct_final_force_vectors(mlff_row)
    dft_forces = direct_final_force_vectors(dft_row)
    if mlff_forces is None or dft_forces is None:
        return np.nan
    if set(mlff_forces) != set(dft_forces) or not mlff_forces:
        return np.nan


def paired_mlff_dft_records(records, output_dir):
    """Build post-relaxation MLFF--DFT pairs for each random-seed trial."""
    output_dir = Path(output_dir)
    data = records.copy()
    mlff = data[~data["calculator"].astype(str).str.startswith("dft_")]
    dft = data[data["calculator"].astype(str).str.startswith("dft_")]
    by_trial_and_id = {
        (str(row.get("trial", "")), str(row["run_id"])): row
        for _, row in mlff.iterrows()
    }
    by_id = {str(row["run_id"]): row for _, row in mlff.iterrows()}
    pairs, missing = [], []

    for _, dft_row in dft.iterrows():
        source_id = dft_row.get("dft_source_run_id")
        if source_id is None or pd.isna(source_id):
            missing.append(f"No source ID for DFT run {dft_row.get('run_id')}")
            continue
        source = by_trial_and_id.get((str(dft_row.get("trial", "")), str(source_id)))
        source = source if source is not None else by_id.get(str(source_id))
        if source is None:
            missing.append(f"No MLFF match for DFT run {dft_row.get('run_id')}")
            continue
        try:
            mlff_atoms = direct_final_atoms(source)
            dft_atoms = direct_final_atoms(dft_row)
            if len(mlff_atoms) != len(dft_atoms):
                raise ValueError("atom-count mismatch")
            displacement = np.linalg.norm(
                mlff_atoms.positions - dft_atoms.positions, axis=1
            )
            mlff_edges = neighbor_edge_set(mlff_atoms)
            dft_edges = neighbor_edge_set(dft_atoms)
            mlff_cn = coordination_by_atom(mlff_edges, mlff_atoms)
            dft_cn = coordination_by_atom(dft_edges, dft_atoms)
            cn_delta = [
                abs(mlff_cn.get(atom, 0) - dft_cn.get(atom, 0))
                for atom in set(mlff_cn) | set(dft_cn)
            ]
        except Exception as error:
            missing.append(f"Could not calculate paired geometry for {source_id}: {error}")
            continue

        pairs.append({
            "run_id": source_id,
            "trial": source.get("trial"),
            "seed": source.get("seed"),
            "source_model": source.get("calculator"),
            "material_slug": source.get("material_slug"),
            "attack_label": source.get("attack_label"),
            "epsilon": source.get("epsilon"),
            "epsilon_percent_min_lattice": source.get("epsilon_percent_displacement"),
            "median_displacement_a": float(np.median(displacement)),
            "rms_force_difference_ev_a": direct_final_force_rms_difference(source, dft_row),
            "relaxation_step_difference": (
                abs(float(source.get("after_relax_steps")) - float(dft_row.get("after_relax_steps")))
                if pd.notna(source.get("after_relax_steps"))
                and pd.notna(dft_row.get("after_relax_steps"))
                else np.nan
            ),
            "neighbor_jaccard_distance": edge_jaccard_distance(mlff_edges, dft_edges),
            "coordination_number_difference_max": float(max(cn_delta)) if cn_delta else 0.0,
            "rdf_l1_distance": float(rdf_l1_distance(mlff_atoms, dft_atoms)),
        })

    pair_columns = [
        "run_id", "trial", "seed", "source_model", "material_slug",
        "attack_label", "epsilon", "epsilon_percent_min_lattice",
        "median_displacement_a", "rms_force_difference_ev_a",
        "relaxation_step_difference", "neighbor_jaccard_distance",
        "coordination_number_difference_max", "rdf_l1_distance",
    ]
    pairs = pd.DataFrame(pairs, columns=pair_columns)
    pairs.to_csv(output_dir / "paired_mlff_dft_post_attack_relaxation.csv", index=False)
    pd.DataFrame({"reason": missing}).to_csv(output_dir / "missing_pairs.csv", index=False)
    return pairs, missing


def draw_direct_comparison_panel(ax, pairs, metric, attack):
    """Draw random-seed styled MLFF--DFT trends for one metric and attack."""
    plotted = []
    for model in ("mace_mh", "uma"):
        values = pairs.loc[
            (pairs["source_model"] == model)
            & (pairs["attack_label"] == attack),
            ["epsilon", "epsilon_percent_min_lattice", metric],
        ].copy()
        values = values.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        values = values[
            (values["epsilon"] > 0)
            & (values["epsilon_percent_min_lattice"] > 0)
        ]
        if values.empty:
            continue
        color = COLORS[model]
        ax.scatter(
            values["epsilon_percent_min_lattice"], values[metric],
            s=17, color=color, alpha=0.46, edgecolors="white",
            linewidths=0.35, zorder=3,
        )
        summary = values.groupby("epsilon", as_index=False).agg(
            epsilon_percent_min_lattice=("epsilon_percent_min_lattice", "median"),
            median=(metric, "median"),
            q25=(metric, lambda series: series.quantile(0.25)),
            q75=(metric, lambda series: series.quantile(0.75)),
        ).sort_values("epsilon_percent_min_lattice")
        x = summary["epsilon_percent_min_lattice"].to_numpy(dtype=float)
        center = summary["median"].to_numpy(dtype=float)
        q25 = summary["q25"].to_numpy(dtype=float)
        q75 = summary["q75"].to_numpy(dtype=float)
        smooth_x, smooth_center = smooth_seed_summary(x, center)
        _, smooth_q25 = smooth_seed_summary(x, q25)
        _, smooth_q75 = smooth_seed_summary(x, q75)
        ax.fill_between(
            smooth_x, np.minimum(smooth_q25, smooth_q75),
            np.maximum(smooth_q25, smooth_q75), color=color,
            alpha=0.18, linewidth=0, zorder=1,
        )
        ax.plot(smooth_x, smooth_center, color=color, linewidth=2.25, zorder=4)
        plotted.extend(center.tolist() + q25.tolist() + q75.tolist())

    if not plotted:
        ax.text(0.5, 0.5, "No matched MLFF--DFT data", transform=ax.transAxes,
                ha="center", va="center", color="#555555")
    ax.set_xscale("log")
    configure_y_axis(ax, plotted, scale="linear")
    ax.set_title(attack, pad=7)
    ax.grid(True, which="major", alpha=0.24, linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.45)
    ax.tick_params(axis="both", labelsize=8)



def save_normalized_agreement_heatmap(pairs, output_dir):
    """Save the MLFF--DFT agreement heatmap beside the random-seed plots."""
    heatmap_metrics = [
        ("neighbor_jaccard_distance", "Neighbor Jaccard distance"),
        ("coordination_number_difference_max", r"Max $\Delta$ CN"),
        ("relaxation_step_difference", "Relaxation steps"),
        ("rdf_l1_distance", "RDF L1 distance"),
        ("rms_force_difference_ev_a", r"RMS $\Delta$ force"),
        ("median_displacement_a", "Displacement"),
    ]
    comparison_models = ("mace_mh", "uma")
    targets = np.asarray([
        1e-2, 5e-2, 1e-1, 5e-1, 1.0,
        5.0, 10.0, 50.0, 100.0, 500.0,
    ])
    line_data = pairs.loc[
        pairs["source_model"].isin(comparison_models)
    ].copy()
    for column in ("epsilon", "epsilon_percent_min_lattice"):
        line_data[column] = pd.to_numeric(line_data[column], errors="coerce")
    line_data = line_data.loc[
        np.isfinite(line_data["epsilon"])
        & (line_data["epsilon"] > 0)
        & np.isfinite(line_data["epsilon_percent_min_lattice"])
        & (line_data["epsilon_percent_min_lattice"] > 0)
    ].copy()

    matrices = {}
    for model in comparison_models:
        model_data = line_data.loc[line_data["source_model"] == model].copy()
        matrix = np.full((len(heatmap_metrics), len(targets)), np.nan)
        if not model_data.empty:
            log_distance = np.abs(
                np.log10(model_data["epsilon_percent_min_lattice"].to_numpy(float))[:, None]
                - np.log10(targets)[None, :]
            )
            model_data["epsilon_bin"] = log_distance.argmin(axis=1)
            for metric_index, (metric, _) in enumerate(heatmap_metrics):
                values = pd.to_numeric(model_data[metric], errors="coerce")
                grouped = values.groupby(model_data["epsilon_bin"]).median()
                for bin_index, value in grouped.items():
                    matrix[metric_index, int(bin_index)] = value
        matrices[model] = matrix

    normalized = {
        model: np.full_like(matrix, np.nan)
        for model, matrix in matrices.items()
    }
    for metric_index in range(len(heatmap_metrics)):
        combined = np.concatenate([
            matrices[model][metric_index][np.isfinite(matrices[model][metric_index])]
            for model in comparison_models
        ])
        if not combined.size:
            continue
        lower, upper = float(np.min(combined)), float(np.max(combined))
        for model in comparison_models:
            row = matrices[model][metric_index]
            if np.isclose(lower, upper):
                # A constant non-zero change is still a maximum change;
                # only a constant zero row represents no difference.
                normalized[model][metric_index] = np.where(
                    np.isfinite(row),
                    1.0 if upper > 0.0 else 0.0,
                    np.nan,
                )
            else:
                normalized[model][metric_index] = (row - lower) / (upper - lower)

    fig, ax = plt.subplots(figsize=(11.2, 5.8), facecolor="white")
    scalar_map = plt.cm.ScalarMappable(
        norm=plt.Normalize(0.0, 1.0),
        cmap=plt.colormaps["Blues"],
    )
    scalar_map.set_array(np.linspace(0.0, 1.0, 256))
    for row in range(len(heatmap_metrics)):
        for column in range(len(targets)):
            for model, corners, text_position in (
                ("mace_mh", ((column - 0.5, row - 0.5), (column + 0.5, row - 0.5),
                             (column + 0.5, row + 0.5)), (column + 0.18, row - 0.18)),
                ("uma", ((column - 0.5, row - 0.5), (column - 0.5, row + 0.5),
                         (column + 0.5, row + 0.5)), (column - 0.18, row + 0.18)),
            ):
                value = normalized[model][row, column]
                facecolor = scalar_map.to_rgba(value) if np.isfinite(value) else "#F1F5F9"
                ax.add_patch(Polygon(corners, closed=True, facecolor=facecolor,
                                     edgecolor="white", linewidth=0.85))
                if np.isfinite(value):
                    ax.text(*text_position, f"{value:.2f}", ha="center", va="center",
                            fontsize=8.4, color="white" if value >= 0.58 else "#0F172A")

    ax.set_xlim(-0.5, len(targets) - 0.5)
    ax.set_ylim(len(heatmap_metrics) - 0.5, -0.5)
    ax.set_aspect("auto")
    ax.set_facecolor("white")
    ax.set_xticks((0, 2, 4, 6, 8))
    ax.set_xticklabels([
        r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$",
    ], fontsize=12, color="#000000")
    ax.set_yticks(np.arange(len(heatmap_metrics)))
    ax.set_yticklabels([label for _, label in heatmap_metrics], fontsize=13)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle("Random-seed MLFF--DFT agreement after perturbation + relaxation",
                 x=0.50, y=0.982, ha="center", fontsize=17,
                 fontweight="bold", color="#0F172A")
    fig.legend(
        handles=[
            Patch(facecolor="#4B8CC0", label="MACE-MH, upper triangle"),
            Patch(facecolor="#4B8CC0", label="UMA, lower triangle"),
        ],
        loc="upper center", bbox_to_anchor=(0.50, 0.945), ncol=2,
        frameon=False, fontsize=11,
    )
    colourbar = fig.colorbar(scalar_map, ax=ax, fraction=0.036, pad=0.035)
    colourbar.set_ticks([0.0, 0.5, 1.0])
    colourbar.set_ticklabels(["0.0", "0.5", "1.0"])
    colourbar.ax.tick_params(labelsize=11, length=0)
    colourbar.set_label("Normalized difference from DFT", fontsize=12, color="#000000", labelpad=9)
    colourbar.outline.set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.90, bottom=0.17, top=0.78)
    fig.text(0.55, 0.075, r"$\epsilon$ strength (% min lattice)",
             ha="center", fontsize=14, color="#000000")
    fig.savefig(Path(output_dir) / "direct_comparison_normalized_heatmap.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
def save_direct_comparison_figures(records, output_dir):
    """Save standalone MLFF--DFT plots in the random-seed individual format."""
    output_dir = Path(output_dir) / "direct_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs, missing = paired_mlff_dft_records(records, output_dir)
    save_normalized_agreement_heatmap(pairs, output_dir)
    metrics = [
        ("median_displacement_a", r"Median displacement ($\AA$)"),
        ("rms_force_difference_ev_a", r"RMS $\Delta$ force (eV/$\AA$)"),
        ("relaxation_step_difference", "Relaxation-step difference"),
        ("neighbor_jaccard_distance", "Neighbor Jaccard distance"),
        ("coordination_number_difference_max", "Maximum CN difference"),
        ("rdf_l1_distance", "RDF L1 distance"),
    ]
    attacks = ADVERSARIAL_ATTACKS

    def save_individual(metric, ylabel, attack, column):
        """Export one plot with the exact standalone random-seed geometry."""
        fig, axes = plt.subplots(3, 4, figsize=(18.2, 10.4), squeeze=False)
        ax = axes[0, column]
        draw_direct_comparison_panel(ax, pairs, metric, attack)
        ax.set_ylabel(ylabel, labelpad=7)
        ax.set_xlabel(r"$\epsilon$ strength (% min lattice parameter)", labelpad=7)
        for row_axes in axes:
            for placeholder_axis in row_axes:
                placeholder_axis.set_xlabel(
                    r"$\epsilon$ strength (% min lattice parameter)", labelpad=6,
                )
        for row_axes in axes:
            row_axes[0].set_ylabel(ylabel, labelpad=7)
        fig.tight_layout(
            rect=[0.035, 0.05, 0.995, 0.875], h_pad=2.0, w_pad=1.8,
        )
        for row_axes in axes:
            for placeholder_axis in row_axes:
                placeholder_axis.set_visible(placeholder_axis is ax)

        # Match the 40% taller canvas and cropped axis-plus-legend geometry
        # used by the existing random-seed individual exports.
        fig.set_size_inches(18.2, 10.4 * 1.40, forward=True)
        fig.canvas.draw()

        model_legend = ax.legend(
            handles=[
                Line2D([0], [0], color=COLORS[model], linewidth=2.7,
                       label=model_label(model))
                for model in ("mace_mh", "uma")
            ],
            loc="lower center",
            bbox_to_anchor=(0.28, 1.30),
            ncol=2,
            frameon=False,
            fontsize=7.2,
            handlelength=2.2,
            columnspacing=1.0,
            handletextpad=0.45,
            borderaxespad=0.0,
            title="Models",
            title_fontsize=7.4,
        )
        ax.add_artist(model_legend)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounding_box = matplotlib.transforms.Bbox.union([
            ax.get_tightbbox(renderer),
            model_legend.get_window_extent(renderer),
        ]).transformed(fig.dpi_scale_trans.inverted()).padded(0.06)

        fig.savefig(
            output_dir / f"mlff_dft_{metric}_{attack.lower().replace('-', '')}.png",
            dpi=300,
            bbox_inches=bounding_box,
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

    for metric, label in metrics:
        for column, attack in enumerate(attacks):
            save_individual(metric, label, attack, column)
    return pairs, missing
def main():
    global CALCULATORS, RELAXATION_STEP_UPPER_LIMIT

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mace_mh", "uma", "mtp", "chgnet", "mace_model"],
        default=None,
        help="Plot only this base model set. Defaults to all LiCoHPF models.",
    )

    args = parser.parse_args()

    project_root = args.project_root.resolve()
    RELAXATION_STEP_UPPER_LIMIT = relaxation_step_upper_limit(project_root)

    # Exclude CHGNet from every 2D-structures random-seed figure.
    exclude_chgnet = "2d_structures" in str(project_root).lower()

    if exclude_chgnet:
        CALCULATORS = [
            calculator
            for calculator in CALCULATORS
            if "chgnet" not in calculator
        ]

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / "random_seed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records, missing = load_trials(project_root)

    if args.models is not None:
        selected_models = [
            model_id
            for model_id in args.models
            if model_id in {
                "mace_mh",
                "uma",
                "mtp",
                "chgnet",
                "mace_model",
            }
            and not (exclude_chgnet and model_id == "chgnet")
        ]

        present = set(
            records["calculator"]
            .dropna()
            .astype(str)
        )

        CALCULATORS = []

        for model_id in selected_models:
            if model_id in present:
                CALCULATORS.append(
                    model_id
                )

            dft_model_id = (
                f"dft_{model_id}"
            )

            if dft_model_id in present:
                CALCULATORS.append(
                    dft_model_id
                )

    records = prepare_records(records)

    dft_coverage = dft_coverage_table(records)
    dft_coverage.to_csv(
        output_dir / "random_seed_dft_coverage.csv",
        index=False,
    )
    if not dft_coverage.empty:
        print("Random-seed DFT coverage by model and attack:")
        print(
            dft_coverage.groupby(
                ["calculator", "attack_label"],
                dropna=False,
            )["records"]
            .sum()
            .to_string()
        )

    contour_records, missing_contour = load_contour_trials(
        project_root
    )

    if args.models is not None and not contour_records.empty:
        contour_records = contour_records[
            contour_records["calculator"].isin(args.models)
        ].copy()

    if not contour_records.empty:
        records = pd.concat(
            [records, contour_records],
            ignore_index=True,
            sort=False,
        )

        print(
            "Random-seed plotting added "
            f"{len(contour_records)} contour records."
        )
    records = add_post_attack_rms_columns(records)
    records = add_post_attack_relaxed_rms_at_fmax_column(
        records,
        fmax=0.05,
    )

    records.to_csv(
        output_dir / "random_seed_combined.csv",
        index=False,
    )

    pd.DataFrame(
        missing + missing_contour
    ).to_csv(
        output_dir
        / "random_seed_missing_trials.csv",
        index=False,
    )

    write_aggregate_table(
        records,
        output_dir / "random_seed_aggregate.csv",
    )
    save_random_seed_rms_plots(
        records,
        output_dir,
        MODEL_LABELS,
        COLORS,
    )
    direct_pairs, direct_missing = save_direct_comparison_figures(
        records,
        output_dir,
    )
    print(
        "Saved "
        f"{len(direct_pairs)} paired random-seed MLFF-vs-DFT comparisons to "
        f"{output_dir / 'direct_comparison'}"
    )




    # Existing final-response figures.
    final_stage = "after_attack_after_relaxation"

    make_metric_figure(
        records,
        physical_metrics(final_stage),
        output_dir
        / "seed_response_physical_metrics_after_attack_after_relaxation.png",
        "Random-seed comparison: physical response "
        "after attack and relaxation",
        panel_scales={
            "E": "symlog",
            "F": "symlog",
            "G": "symlog",
            "H": "symlog",
        },
    )

    make_metric_figure(
        records,
        topology_metrics_for_stage(final_stage),
        output_dir
        / "seed_response_topology_metrics_after_attack_after_relaxation.png",
        "Random-seed comparison: topology response after attack and relaxation",
        panel_scales={
            "A": "log",
            "B": "log",
            "C": "log",
            "D": "log",
            "E": "log",
            "F": "log",
            "G": "log",
            "H": "log",
        },
    )

    # Immediate post-attack figures.
    immediate_stage = "after_attack_before_relaxation"

    make_metric_figure(
        records,
        physical_metrics(immediate_stage),
        output_dir
        / (
            "seed_response_physical_metrics_"
            "after_attack_before_relaxation.png"
        ),
        "Random-seed comparison: immediate physical response "
        "after attack, before relaxation",
        panel_scales={
            "E": "symlog",
            "F": "symlog",
            "G": "symlog",
            "H": "symlog",
        },
    )

    make_metric_figure(
        records,
        topology_metrics_for_stage(immediate_stage),
        output_dir
        / (
            "seed_response_topology_metrics_"
            "after_attack_before_relaxation.png"
        ),
        "Random-seed comparison: immediate topology response "
        "after attack, before relaxation",
    )

    # Initial-relaxation figures.
    baseline_stage = "before_attack_after_relaxation"

    make_metric_figure(
        records,
        physical_metrics(baseline_stage),
        output_dir
        / (
            "seed_response_physical_metrics_"
            "before_attack_after_relaxation.png"
        ),
        "Random-seed comparison: physical response "
        "pre-relaxation",
        panel_scales={
            "E": "symlog",
            "F": "symlog",
            "G": "symlog",
            "H": "symlog",
        },
    )

    make_metric_figure(
        records,
        topology_metrics_for_stage(baseline_stage),
        output_dir
        / (
            "seed_response_topology_metrics_"
            "before_attack_after_relaxation.png"
        ),
        "Random-seed comparison: topology response "
        "pre-relaxation",
    )

    print(
        f"Saved random-seed outputs to {output_dir}"
    )


if __name__ == "__main__":
    main()
