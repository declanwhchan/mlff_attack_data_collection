#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


TRIALS = [
    ("trial1_seed42", 42),
    ("trial2_seed43", 43),
    ("trial3_seed44", 44),
    ("trial4_seed45", 45),
    ("trial5_seed46", 46),
]
ATTACKS = ["FGSM", "I-FGSM", "PGD"]
CALCULATORS = ["mace", "uma"]
COLORS = {"mace": "#0072B2", "uma": "#D55E00"}
SEED_STYLES = {
    42: ("-", "o"),
    43: ("--", "s"),
    44: ("-.", "^"),
    45: (":", "D"),
    46: ((0, (3, 1, 1, 1)), "P"),
}
PHYSICAL_METRICS = [
    ("median_displacement_a", r"Median displacement ($\AA$)"),
    ("median_delta_force_ev_a", r"Median $\Delta$ force (eV/$\AA$)"),
    ("after_relax_steps", "Attack-relaxation steps"),
]
TOPOLOGY_METRICS = [
    ("neighbor_jaccard_distance", "Neighbor Jaccard distance"),
    ("rdf_l1_distance", "RDF L1 distance"),
    ("coordination_change_max", "Maximum coordination change"),
]


def numeric(series):
    return pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def read_force_csv(path):
    try:
        data = pd.read_csv(Path(path))
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    required = {"atom_index", "x", "y", "z", "fx", "fy", "fz"}
    return data if required.issubset(data.columns) else None


def median_run_metrics(run_dir):
    run_dir = Path(str(run_dir))
    before = read_force_csv(run_dir / "before_forces.csv")
    after = read_force_csv(run_dir / "after_forces.csv")
    if before is None or after is None:
        return np.nan, np.nan

    merged = before.merge(after, on="atom_index", suffixes=("_before", "_after"))
    if merged.empty:
        return np.nan, np.nan

    before_positions = merged[["x_before", "y_before", "z_before"]].to_numpy(float)
    after_positions = merged[["x_after", "y_after", "z_after"]].to_numpy(float)
    before_forces = merged[["fx_before", "fy_before", "fz_before"]].to_numpy(float)
    after_forces = merged[["fx_after", "fy_after", "fz_after"]].to_numpy(float)
    displacement = np.linalg.norm(after_positions - before_positions, axis=1)
    delta_force = np.linalg.norm(after_forces - before_forces, axis=1)
    return float(np.median(displacement)), float(np.median(delta_force))


def load_trials(project_root):
    frames = []
    missing = []
    for trial_name, seed in TRIALS:
        path = project_root / trial_name / "outputs_comprehensive" / "float64" / "combined_dataset.csv"
        try:
            data = pd.read_csv(path)
        except Exception as error:
            missing.append({"trial": trial_name, "seed": seed, "reason": str(error)})
            continue
        if data.empty:
            missing.append({"trial": trial_name, "seed": seed, "reason": "empty dataset"})
            continue
        data["trial"] = trial_name
        data["seed"] = seed
        frames.append(data)

    if not frames:
        raise SystemExit("ERROR: no float64 trial datasets were readable")
    return pd.concat(frames, ignore_index=True, sort=False), missing


def prepare_records(records):
    required = {
        "run_id", "material_slug", "calculator", "attack_label", "epsilon",
        "epsilon_percent_displacement", "run_dir", "seed",
    }
    missing = sorted(required - set(records.columns))
    if missing:
        raise SystemExit("ERROR: missing columns: " + ", ".join(missing))

    data = records[
        ~records["run_id"].astype(str).str.contains("_steps", regex=False)
    ].copy()
    data = data[data["attack_label"].isin(ATTACKS)]
    data = data[data["calculator"].isin(CALCULATORS)].copy()

    for column in [
        "epsilon", "epsilon_percent_displacement", "after_relax_steps",
        "neighbor_jaccard_distance", "rdf_l1_distance", "coordination_change_max",
    ]:
        if column not in data.columns:
            data[column] = np.nan
        data[column] = numeric(data[column])

    run_metrics = [median_run_metrics(path) for path in data["run_dir"]]
    data["median_displacement_a"] = [item[0] for item in run_metrics]
    data["median_delta_force_ev_a"] = [item[1] for item in run_metrics]
    return data


def seed_curves(records, metric):
    clean = records.copy()
    clean[metric] = numeric(clean[metric])
    clean = clean.dropna(subset=[
        "seed", "calculator", "attack_label", "epsilon",
        "epsilon_percent_displacement", metric,
    ])
    return clean.groupby(
        ["seed", "calculator", "attack_label", "epsilon"], as_index=False
    ).agg(
        epsilon_percent_displacement=("epsilon_percent_displacement", "median"),
        value=(metric, "median"),
        material_count=("material_slug", "nunique"),
    ).sort_values("epsilon")


def aggregate_curves(curves):
    rows = []
    for key, group in curves.groupby(["calculator", "attack_label", "epsilon"]):
        values = numeric(group["value"]).dropna().to_numpy(float)
        if len(values) < 3:
            continue
        rows.append({
            "calculator": key[0],
            "attack_label": key[1],
            "epsilon": float(key[2]),
            "epsilon_percent_displacement": float(np.median(group["epsilon_percent_displacement"])),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "seed_count": len(values),
        })
    return pd.DataFrame(
        rows,
        columns=[
            "calculator",
            "attack_label",
            "epsilon",
            "epsilon_percent_displacement",
            "median",
            "q25",
            "q75",
            "seed_count",
        ],
    )


def cap_y_axis(ax, values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    low = min(0.0, float(np.percentile(values, 1)))
    high = float(np.percentile(values, 99))
    if high > low:
        pad = 0.08 * (high - low)
        ax.set_ylim(low - pad, high + pad)


def draw_metric_panel(ax, records, metric, attack):
    curves = seed_curves(records, metric)
    curves = curves[curves["attack_label"] == attack]
    aggregate = aggregate_curves(curves)
    plotted = []

    for calculator in CALCULATORS:
        calc_curves = curves[curves["calculator"] == calculator]
        color = COLORS[calculator]
        for seed, seed_data in calc_curves.groupby("seed"):
            seed_data = seed_data.sort_values("epsilon")
            linestyle, marker = SEED_STYLES[int(seed)]
            ax.plot(
                seed_data["epsilon_percent_displacement"], seed_data["value"],
                color=color, linestyle=linestyle, marker=marker,
                markersize=2.5, linewidth=0.85, alpha=0.42,
            )
            plotted.extend(seed_data["value"].tolist())

        summary = aggregate[aggregate["calculator"] == calculator].sort_values("epsilon")
        if summary.empty:
            continue
        x = summary["epsilon_percent_displacement"].to_numpy(float)
        center = summary["median"].to_numpy(float)
        ax.fill_between(
            x, summary["q25"].to_numpy(float), summary["q75"].to_numpy(float),
            color=color, alpha=0.15, linewidth=0,
        )
        ax.plot(x, center, color=color, linewidth=2.2)

    if not plotted:
        ax.text(0.5, 0.5, "No matched seed data", transform=ax.transAxes,
                ha="center", va="center")
    positive_x = numeric(curves["epsilon_percent_displacement"]).dropna()
    if (positive_x > 0).any():
        ax.set_xscale("log")
    cap_y_axis(ax, plotted)
    ax.set_title(attack)
    ax.grid(True, alpha=0.25)


def figure_legend():
    handles = [
        Line2D([0], [0], color=COLORS[calc], linewidth=2.5, label=calc.upper())
        for calc in CALCULATORS
    ]
    handles.extend(
        Line2D([0], [0], color="#555555", linestyle=SEED_STYLES[seed][0],
               marker=SEED_STYLES[seed][1], markersize=4, linewidth=1,
               label=f"Seed {seed}")
        for seed in sorted(SEED_STYLES)
    )
    handles.append(Line2D([0], [0], color="#333333", linewidth=2.5,
                          label="Cross-seed median"))
    return handles


def make_metric_figure(records, metrics, output_path, title):
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 10), squeeze=False)
    for row, (metric, ylabel) in enumerate(metrics):
        for column, attack in enumerate(ATTACKS):
            ax = axes[row, column]
            draw_metric_panel(ax, records, metric, attack)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("Epsilon (% minimum lattice length)")
            ax.text(-0.11, 1.05, chr(ord("A") + row * 3 + column),
                    transform=ax.transAxes, fontweight="bold", va="top")

    fig.legend(handles=figure_legend(), loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(title, fontsize=15, y=1.015)
    fig.text(
        0.5, 0.012,
        "Thin lines: individual seeds; thick lines: cross-seed median; shading: interquartile range",
        ha="center", fontsize=8.5, color="#555555",
    )
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.92])
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def write_aggregate_table(records, output_path):
    tables = []
    for metric, _ in PHYSICAL_METRICS + TOPOLOGY_METRICS:
        aggregate = aggregate_curves(seed_curves(records, metric))
        if aggregate.empty:
            continue
        aggregate["metric"] = metric
        tables.append(aggregate)
    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    result.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else project_root / "random_seed"
    output_dir.mkdir(parents=True, exist_ok=True)

    records, missing = load_trials(project_root)
    records = prepare_records(records)
    records.to_csv(output_dir / "random_seed_combined.csv", index=False)
    pd.DataFrame(missing).to_csv(output_dir / "random_seed_missing_trials.csv", index=False)
    write_aggregate_table(records, output_dir / "random_seed_aggregate.csv")

    make_metric_figure(
        records, PHYSICAL_METRICS,
        output_dir / "seed_response_physical_metrics.png",
        "Random-seed comparison: physical response",
    )
    make_metric_figure(
        records, TOPOLOGY_METRICS,
        output_dir / "seed_response_topology_metrics.png",
        "Random-seed comparison: topology response",
    )
    print(f"Saved random-seed outputs to {output_dir}")


if __name__ == "__main__":
    main()
