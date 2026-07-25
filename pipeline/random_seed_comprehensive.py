#!/usr/bin/env python3

from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from ase.io import read as ase_read


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

ATTACKS = [
    "FGSM",
    "I-FGSM",
    "PGD",
]

CALCULATORS = [
    "mace_mh",
    "uma",
    "mtp",
    "chgnet",
    "mace_model",
]

MODEL_LABELS = {
    "mace_mh": "MACE-MH-1",
    "uma": "UMA-S-1p1",
    "mtp": "MTP",
    "chgnet": "CHGNet",
    "mace_model": "MACE Model",
}

COLORS = {
    "mace_mh": "#0072B2",
    "uma": "#D55E00",
    "mtp": "#CC79A7",
    "chgnet": "#009E73",
    "mace_model": "#E69F00",
}


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
        "RDF L1 distance",
    ),
    (
        "coordination_change_max",
        "Maximum coordination change",
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


def calculate_stage_metrics(row):
    run_dir = Path(str(row["run_dir"]))

    before_force_path = (
        run_dir / "before_forces.csv"
    )
    perturbed_force_path = (
        run_dir / "perturbed_forces.csv"
    )
    after_force_path = (
        run_dir / "after_forces.csv"
    )

    trajectory_value = row.get(
        "before_relax_traj"
    )

    if (
        trajectory_value is None
        or pd.isna(trajectory_value)
        or not str(trajectory_value).strip()
    ):
        trajectory_path = (
            run_dir / "before_attack_relaxation.traj"
        )
    else:
        trajectory_path = Path(
            str(trajectory_value)
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
        "run_dir",
        "seed",
    ]

    data = records.copy()

    # Add missing columns instead of aborting.
    for column in required_columns:
        if column not in data.columns:
            data[column] = np.nan

    run_ids = (
        data["run_id"]
        .fillna("")
        .astype(str)
    )

    data = data[
        ~run_ids.str.contains(
            "_steps",
            regex=False,
        )
    ].copy()

    # Retain every valid available row. Missing models, attacks and
    # epsilon values are allowed.
    data = data[
        data["attack_label"].isin(ATTACKS)
        & data["calculator"].isin(CALCULATORS)
    ].copy()

    data["epsilon"] = numeric(
        data["epsilon"]
    )

    data["epsilon_percent_displacement"] = numeric(
        data["epsilon_percent_displacement"]
    )

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

    return data


def physical_metrics(stage):
    if stage == "before_attack_after_relaxation":
        step_label = "Initial relaxation steps"
    elif stage == "after_attack_before_relaxation":
        step_label = "Subsequent post-attack relaxation steps"
    else:
        step_label = "Post-attack relaxation steps"

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
    clean[metric] = numeric(clean[metric])

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
            ],
            as_index=False,
        )
        .agg(
            epsilon_percent_displacement=(
                "epsilon_percent_displacement",
                "median",
            ),
            value=(metric, "median"),
            material_count=(
                "material_slug",
                "nunique",
            ),
        )
        .sort_values("epsilon")
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

        for seed, seed_data in (
            calculator_curves.groupby(
                "seed"
            )
        ):
            seed_data = (
                seed_data.sort_values(
                    "epsilon"
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna(
                    subset=[
                        "epsilon_percent_displacement",
                        "value",
                    ]
                )
            )

            seed_data = seed_data[
                seed_data[
                    "epsilon_percent_displacement"
                ] > 0
            ]

            if seed_data.empty:
                continue

            seed_number = int(
                seed
            )

            linestyle, marker = (
                SEED_STYLES.get(
                    seed_number,
                    ("-", "o"),
                )
            )

            x = seed_data[
                "epsilon_percent_displacement"
            ].to_numpy(dtype=float)

            y = seed_data[
                "value"
            ].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=2.6,
                markeredgewidth=0.3,
                linewidth=0.9,
                alpha=0.38,
                zorder=2,
            )

            plotted.extend(
                y.tolist()
            )

        summary = aggregate[
            aggregate["calculator"]
            == calculator
        ].sort_values(
            "epsilon"
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

        ax.fill_between(
            x,
            q25,
            q75,
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            x,
            center,
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
        ax.text(
            0.5,
            0.5,
            "No matched seed data",
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
    present_calculators = set(
        records["calculator"]
        .dropna()
        .astype(str)
    )

    present_seeds = set(
        pd.to_numeric(
            records["seed"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS.get(
                calculator,
                "#777777",
            ),
            linewidth=2.7,
            label=model_label(
                calculator
            ),
        )
        for calculator in CALCULATORS
        if calculator in present_calculators
    ]

    handles.extend(
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=SEED_STYLES.get(
                seed,
                ("-", "o"),
            )[0],
            marker=SEED_STYLES.get(
                seed,
                ("-", "o"),
            )[1],
            markersize=4,
            linewidth=1,
            label=f"Seed {seed}",
        )
        for seed in sorted(
            present_seeds
        )
    )

    if present_seeds:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#333333",
                linewidth=2.7,
                label="Available-seed median",
            )
        )

    return handles


def make_metric_figure(
    records,
    metrics,
    output_path,
    title,
    panel_scales=None,
):
    """
    Create a 3x3 random-seed figure.

    panel_scales maps panel letters to "linear", "log" or "symlog".
    Force panels should use symlog because force changes can contain
    both exact zeros and extremely large finite values.
    """
    panel_scales = dict(
        panel_scales or {}
    )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(14.5, 10.4),
        squeeze=False,
    )

    for row, (
        metric,
        ylabel,
    ) in enumerate(metrics):
        for column, attack in enumerate(
            ATTACKS
        ):
            ax = axes[
                row,
                column,
            ]

            panel_label = chr(
                ord("A")
                + row * 3
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

            if row == 2:
                ax.set_xlabel(
                    "Epsilon "
                    "(% minimum lattice length)",
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

    legend_handles = figure_legend(
        records
    )

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(
                0.5,
                0.955,
            ),
        )

    fig.suptitle(
        title,
        fontsize=15,
        y=0.995,
    )

    uses_symlog = any(
        scale == "symlog"
        for scale in panel_scales.values()
    )

    note = (
        "Thin lines: individual seeds; "
        "thick lines: available-seed median; "
        "shading: interquartile range"
    )

    if uses_symlog:
        note += (
            "; force panels use a symmetric-log scale"
        )

    fig.text(
        0.5,
        0.014,
        note,
        ha="center",
        fontsize=8.5,
        color="#555555",
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

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


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


def main():
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

    args = parser.parse_args()

    project_root = args.project_root.resolve()

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
    records = prepare_records(records)

    records.to_csv(
        output_dir / "random_seed_combined.csv",
        index=False,
    )

    pd.DataFrame(missing).to_csv(
        output_dir
        / "random_seed_missing_trials.csv",
        index=False,
    )

    write_aggregate_table(
        records,
        output_dir / "random_seed_aggregate.csv",
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
            "D": "symlog",
            "E": "symlog",
            "F": "symlog",
        },
    )

    make_metric_figure(
        records,
        topology_metrics_for_stage(final_stage),
        output_dir
        / "seed_response_topology_metrics_after_attack_after_relaxation.png",
        "Random-seed comparison: topology response "
        "after attack and relaxation",
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
            "D": "symlog",
            "E": "symlog",
            "F": "symlog",
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
        "during relaxation before attack",
        panel_scales={
            "D": "symlog",
            "E": "symlog",
            "F": "symlog",
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
        "during relaxation before attack",
    )

    print(
        f"Saved random-seed outputs to {output_dir}"
    )


if __name__ == "__main__":
    main()