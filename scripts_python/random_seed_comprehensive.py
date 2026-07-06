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

ATTACKS = ["FGSM", "I-FGSM", "PGD"]
CALCULATORS = ["mace", "uma"]

COLORS = {
    "mace": "#0072B2",
    "uma": "#D55E00",
}

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
    frames = []
    missing = []

    for trial_name, seed in TRIALS:
        path = (
            project_root
            / trial_name
            / "outputs_comprehensive"
            / "float64"
            / "combined_dataset.csv"
        )

        try:
            data = pd.read_csv(path)
        except Exception as error:
            missing.append({
                "trial": trial_name,
                "seed": seed,
                "reason": str(error),
            })
            continue

        if data.empty:
            missing.append({
                "trial": trial_name,
                "seed": seed,
                "reason": "empty dataset",
            })
            continue

        data["trial"] = trial_name
        data["seed"] = seed
        frames.append(data)

    if not frames:
        raise SystemExit(
            "ERROR: no float64 trial datasets were readable"
        )

    return (
        pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        ),
        missing,
    )


def prepare_records(records):
    required = {
        "run_id",
        "material_slug",
        "calculator",
        "attack_label",
        "epsilon",
        "epsilon_percent_displacement",
        "run_dir",
        "seed",
    }

    missing = sorted(
        required - set(records.columns)
    )

    if missing:
        raise SystemExit(
            "ERROR: missing columns: "
            + ", ".join(missing)
        )

    data = records[
        ~records["run_id"]
        .astype(str)
        .str.contains("_steps", regex=False)
    ].copy()

    data = data[
        data["attack_label"].isin(ATTACKS)
        & data["calculator"].isin(CALCULATORS)
    ].copy()

    data["epsilon"] = numeric(data["epsilon"])
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
            data[stage_column(stage, metric)] = [
                result[stage][metric]
                for result in stage_results
            ]

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
    rows = []

    for key, group in curves.groupby(
        ["calculator", "attack_label", "epsilon"]
    ):
        values = numeric(
            group["value"]
        ).dropna().to_numpy(dtype=float)

        if len(values) < 3:
            continue

        rows.append({
            "calculator": key[0],
            "attack_label": key[1],
            "epsilon": float(key[2]),
            "epsilon_percent_displacement": float(
                np.median(
                    group["epsilon_percent_displacement"]
                )
            ),
            "median": float(np.median(values)),
            "q25": float(
                np.percentile(values, 25)
            ),
            "q75": float(
                np.percentile(values, 75)
            ),
            "seed_count": len(values),
        })

    return pd.DataFrame(rows)


def cap_y_axis(ax, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return

    low = min(
        0.0,
        float(np.percentile(values, 1)),
    )
    high = float(np.percentile(values, 99))

    if high > low:
        padding = 0.08 * (high - low)
        ax.set_ylim(
            low - padding,
            high + padding,
        )


def draw_metric_panel(
    ax,
    records,
    metric,
    attack,
):
    curves = seed_curves(records, metric)
    curves = curves[
        curves["attack_label"] == attack
    ]

    aggregate = aggregate_curves(curves)
    plotted = []

    for calculator in CALCULATORS:
        calculator_curves = curves[
            curves["calculator"] == calculator
        ]
        color = COLORS[calculator]

        for seed, seed_data in calculator_curves.groupby(
            "seed"
        ):
            seed_data = seed_data.sort_values(
                "epsilon"
            )
            linestyle, marker = SEED_STYLES[
                int(seed)
            ]

            ax.plot(
                seed_data[
                    "epsilon_percent_displacement"
                ],
                seed_data["value"],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=2.5,
                linewidth=0.85,
                alpha=0.42,
            )

            plotted.extend(
                seed_data["value"].tolist()
            )

        summary = aggregate[
            aggregate["calculator"] == calculator
        ].sort_values("epsilon")

        if summary.empty:
            continue

        x = summary[
            "epsilon_percent_displacement"
        ].to_numpy(dtype=float)

        center = summary[
            "median"
        ].to_numpy(dtype=float)

        ax.fill_between(
            x,
            summary["q25"].to_numpy(dtype=float),
            summary["q75"].to_numpy(dtype=float),
            color=color,
            alpha=0.15,
            linewidth=0,
        )

        ax.plot(
            x,
            center,
            color=color,
            linewidth=2.2,
        )

    if not plotted:
        ax.text(
            0.5,
            0.5,
            "No matched seed data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    positive_x = numeric(
        curves["epsilon_percent_displacement"]
    ).dropna()

    if (positive_x > 0).any():
        ax.set_xscale("log")

    cap_y_axis(ax, plotted)
    ax.set_title(attack)
    ax.grid(True, alpha=0.25)


def figure_legend():
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[calculator],
            linewidth=2.5,
            label=calculator.upper(),
        )
        for calculator in CALCULATORS
    ]

    handles.extend(
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=SEED_STYLES[seed][0],
            marker=SEED_STYLES[seed][1],
            markersize=4,
            linewidth=1,
            label=f"Seed {seed}",
        )
        for seed in sorted(SEED_STYLES)
    )

    handles.append(
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=2.5,
            label="Cross-seed median",
        )
    )

    return handles


def make_metric_figure(
    records,
    metrics,
    output_path,
    title,
    log_panels=None,
):
    log_panels = set(log_panels or [])
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13.2, 10),
        squeeze=False,
    )

    for row, (metric, ylabel) in enumerate(metrics):
        for column, attack in enumerate(ATTACKS):
            ax = axes[row, column]

            draw_metric_panel(
                ax,
                records,
                metric,
                attack,
            )

            panel_label = chr(
                ord("A") + row * 3 + column
            )

            if panel_label in log_panels:
                ax.set_yscale("log", nonpositive="clip")

            if column == 0:
                ax.set_ylabel(ylabel)

            if row == 2:
                ax.set_xlabel(
                    "Epsilon (% minimum lattice length)"
                )

            ax.text(
                -0.11,
                1.05,
                panel_label,
                transform=ax.transAxes,
                fontweight="bold",
                va="top",
            )

    fig.legend(
        handles=figure_legend(),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )

    fig.suptitle(
        title,
        fontsize=15,
        y=1.015,
    )

    fig.text(
        0.5,
        0.012,
        (
            "Thin lines: individual seeds; "
            "thick lines: cross-seed median; "
            "shading: interquartile range"
        ),
        ha="center",
        fontsize=8.5,
        color="#555555",
    )

    fig.tight_layout(
        rect=[0.03, 0.04, 1, 0.92]
    )

    fig.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight",
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
        log_panels={"E", "F"},
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
        log_panels={"E", "F"},
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