#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

from ase.io import read as ase_read

from run_tests import (
    coordination_by_atom,
    neighbor_edge_set,
    rdf_l1_distance,
)


BASE_DIR = Path(__file__).resolve().parent.parent

BETA_COLORS = {
    0.10: "#E69F00",
    0.05: "#009E73",
    0.00: "#7B3294",
}

CALC_COLORS = {
    "mace": "#0072B2",
    "uma": "#D55E00",
}

CONTOUR_BAND_COLOR = "#66C2A5"
ATTACK_ORDER = ["FGSM", "I-FGSM", "PGD"]


def add_panel_label(ax, label):
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def label_axes(axes):
    for index, ax in enumerate(np.asarray(axes).ravel()):
        add_panel_label(ax, chr(ord("A") + index))


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#111111",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "grid.color": "#D0D0D0",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "savefig.dpi": 600,
    })


def style_numeric_axis(ax, xbins=5, ybins=5):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))

    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter(useMathText=True, useOffset=False)
        formatter.set_powerlimits((-3, 3))
        axis.set_major_formatter(formatter)

    ax.tick_params(axis="both", labelsize=8, pad=2)


def positive_finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def format_power_tick(value):
    value = float(value)

    if value == 0:
        return "0"

    if value < 0 or not np.isfinite(value):
        return ""

    power = int(round(np.log10(value)))
    decade = 10.0 ** power

    if not np.isclose(value, decade, rtol=1e-8, atol=0.0):
        return ""

    if power >= 0:
        return f"{decade:g}"

    decimals = abs(power)
    return f"{decade:.{decimals}f}"


def decade_ticks(values):
    values = positive_finite_values(values)
    if len(values) == 0:
        return []

    min_power = int(np.floor(np.log10(np.min(values))))
    max_power = int(np.ceil(np.log10(np.max(values))))

    return [10.0 ** power for power in range(min_power, max_power + 1)]


def apply_log_decade_axis(ax, axis_name, values, label=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return

    positive = values[values > 0]
    if len(positive) == 0:
        return

    ticks = decade_ticks(positive)
    has_zero_or_negative = np.any(values <= 0)

    if axis_name == "x":
        set_scale = ax.set_xscale
        set_ticks = ax.set_xticks
        set_ticklabels = ax.set_xticklabels
        set_lim = ax.set_xlim
        set_label = ax.set_xlabel
        tick_axis = "x"
    else:
        set_scale = ax.set_yscale
        set_ticks = ax.set_yticks
        set_ticklabels = ax.set_yticklabels
        set_lim = ax.set_ylim
        set_label = ax.set_ylabel
        tick_axis = "y"

    if has_zero_or_negative:
        linthresh = max(float(np.min(positive)) / 2.0, 1e-12)
        set_scale("symlog", linthresh=linthresh)
        ticks = [0.0] + ticks
        low = min(0.0, float(np.min(values)))
        high = float(np.max(positive)) * 1.35
        set_lim(low, high)
    else:
        set_scale("log")
        low = float(np.min(positive)) / 1.35
        high = float(np.max(positive)) * 1.35
        set_lim(low, high)

    set_ticks(ticks)
    set_ticklabels([format_power_tick(tick) for tick in ticks])

    if label is not None:
        set_label(label)

    ax.tick_params(axis=tick_axis, labelrotation=0, pad=2)


def clean_axis_values(ax, axis_name):
    values = []

    for line in ax.lines:
        raw = line.get_xdata(orig=False) if axis_name == "x" else line.get_ydata(orig=False)
        try:
            values.extend(np.asarray(raw, dtype=float).ravel().tolist())
        except Exception:
            pass

    series = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return np.array([], dtype=float)
    return series.to_numpy(dtype=float)


def tight_axis_limits(values, pad=0.14):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    if np.allclose(values, values[0]):
        center = float(values[0])
        span = max(abs(center) * 0.20, 1e-9)
        if center >= 0 and center - span < 0:
            return 0.0, center + span
        return center - span, center + span

    low = float(np.percentile(values, 0.5))
    high = float(np.percentile(values, 99.5))
    span = high - low

    if span <= 0 or not np.isfinite(span):
        return None

    low -= pad * span
    high += pad * span

    if np.nanmin(values) >= 0 and low < 0:
        low = 0.0 if np.nanmin(values) < 0.08 * span else max(0.0, low)

    return low, high


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = pd.read_csv(path)
    except Exception:
        return None
    return data if not data.empty else None


def summary_rows(contour_dir):
    data = read_csv(Path(contour_dir) / "summary.csv")
    if data is None or "status" not in data.columns:
        return pd.DataFrame()
    return data[data["status"] == "success"].copy()


def color_for_beta(beta):
    return BETA_COLORS.get(round(float(beta), 2), "#444444")


def format_beta(beta):
    return rf"$\beta={float(beta):.2f}$"


def load_attack_dataset(comprehensive_dir):
    data = read_csv(Path(comprehensive_dir) / "combined_dataset.csv")
    return data if data is not None else pd.DataFrame()


def load_force_csv(path):
    data = read_csv(path)
    if data is None:
        return None
    required = {"atom_index", "x", "y", "z", "fx", "fy", "fz"}
    if not required.issubset(data.columns):
        return None
    return data


def attack_displacement(row, before_name, after_name):
    run_dir = Path(row["run_dir"])
    before = load_force_csv(run_dir / before_name)
    after = load_force_csv(run_dir / after_name)
    if before is None or after is None:
        return np.array([])

    merged = before.merge(after, on="atom_index", suffixes=("_before", "_after"))
    if merged.empty:
        return np.array([])

    before_xyz = merged[["x_before", "y_before", "z_before"]].to_numpy()
    after_xyz = merged[["x_after", "y_after", "z_after"]].to_numpy()
    return np.linalg.norm(after_xyz - before_xyz, axis=1)


def attack_force_delta(row, before_name, after_name):
    run_dir = Path(row["run_dir"])
    before = load_force_csv(run_dir / before_name)
    after = load_force_csv(run_dir / after_name)
    if before is None or after is None:
        return np.array([])

    merged = before.merge(after, on="atom_index", suffixes=("_before", "_after"))
    if merged.empty:
        return np.array([])

    before_f = merged[["fx_before", "fy_before", "fz_before"]].to_numpy()
    after_f = merged[["fx_after", "fy_after", "fz_after"]].to_numpy()
    return np.linalg.norm(after_f - before_f, axis=1)


def attack_force_angle(row, before_name, after_name):
    run_dir = Path(row["run_dir"])
    before = load_force_csv(run_dir / before_name)
    after = load_force_csv(run_dir / after_name)

    if before is None or after is None:
        return np.nan

    merged = before.merge(
        after,
        on="atom_index",
        suffixes=("_before", "_after"),
    )

    if merged.empty:
        return np.nan

    before_forces = merged[
        ["fx_before", "fy_before", "fz_before"]
    ].to_numpy(dtype=float)

    after_forces = merged[
        ["fx_after", "fy_after", "fz_after"]
    ].to_numpy(dtype=float)

    return median_force_angle(
        before_forces,
        after_forces,
    )


def contour_metric_values(rows, metric):
    values = []
    for _, row in rows.iterrows():
        metrics = read_csv(row["metrics_csv"])
        if metrics is not None and metric in metrics.columns:
            values.extend(
                metrics[metric]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .tolist()
            )
    return np.asarray(values, dtype=float)


def median_force_angle(initial_forces, current_forces):
    initial_forces = np.asarray(initial_forces, dtype=float)
    current_forces = np.asarray(current_forces, dtype=float)

    initial_norm = np.linalg.norm(initial_forces, axis=1)
    current_norm = np.linalg.norm(current_forces, axis=1)

    valid = (
        np.isfinite(initial_norm)
        & np.isfinite(current_norm)
        & (initial_norm > 1e-12)
        & (current_norm > 1e-12)
    )

    if not np.any(valid):
        return np.nan

    cosine = np.sum(
        initial_forces[valid] * current_forces[valid],
        axis=1,
    ) / (
        initial_norm[valid] * current_norm[valid]
    )

    angles = np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    )

    return float(np.median(angles))


def contour_topology_metrics(initial_atoms, current_atoms):
    initial_edges = neighbor_edge_set(initial_atoms)
    current_edges = neighbor_edge_set(current_atoms)
    union_edges = initial_edges | current_edges

    if union_edges:
        jaccard = 1.0 - (
            len(initial_edges & current_edges)
            / len(union_edges)
        )
    else:
        jaccard = 0.0

    initial_coordination = coordination_by_atom(
        initial_edges,
        initial_atoms,
    )
    current_coordination = coordination_by_atom(
        current_edges,
        current_atoms,
    )

    coordination_changes = [
        abs(
            current_coordination.get(atom, 0)
            - initial_coordination.get(atom, 0)
        )
        for atom in (
            set(initial_coordination)
            | set(current_coordination)
        )
    ]

    coordination_max = (
        float(np.max(coordination_changes))
        if coordination_changes
        else 0.0
    )

    return {
        "contour_neighbor_jaccard_distance": float(jaccard),
        "contour_coordination_change_max": coordination_max,
        "contour_rdf_l1_distance": rdf_l1_distance(
            initial_atoms,
            current_atoms,
        ),
    }


def contour_frame_table(summary_row, max_frames=101):
    metrics = read_csv(summary_row["metrics_csv"])
    trajectory_path = Path(summary_row["traj"])

    if metrics is None or not trajectory_path.exists():
        return pd.DataFrame()

    try:
        frames = ase_read(trajectory_path, ":")
    except Exception as error:
        print(
            f"Could not read contour trajectory "
            f"{trajectory_path}: {error}"
        )
        return pd.DataFrame()

    count = min(len(metrics), len(frames))

    if count == 0:
        return pd.DataFrame()

    sample_count = min(max_frames, count)
    indices = np.unique(
        np.linspace(
            0,
            count - 1,
            sample_count,
            dtype=int,
        )
    )

    selected = metrics.iloc[indices].copy().reset_index(drop=True)
    selected_frames = [frames[index] for index in indices]

    initial_atoms = frames[0]
    initial_positions = initial_atoms.get_positions()

    try:
        initial_forces = initial_atoms.get_forces()
    except Exception:
        initial_forces = None

    median_displacements = []
    median_force_changes = []
    jaccard_values = []
    coordination_values = []
    rdf_values = []
    force_angle_values = []

    for frame in selected_frames:
        current_positions = frame.get_positions()
        displacement = np.linalg.norm(
            current_positions - initial_positions,
            axis=1,
        )
        median_displacements.append(
            float(np.median(displacement))
        )

        current_forces = None

        try:
            current_forces = frame.get_forces()
        except Exception:
            pass

        if initial_forces is None or current_forces is None:
            median_force_changes.append(np.nan)
            force_angle_values.append(np.nan)
        else:
            force_change = np.linalg.norm(
                current_forces - initial_forces,
                axis=1,
            )
            median_force_changes.append(
                float(np.median(force_change))
            )
            force_angle_values.append(
                median_force_angle(
                    initial_forces,
                    current_forces,
                )
            )

        try:
            topology = contour_topology_metrics(
                initial_atoms,
                frame,
            )
            jaccard_values.append(
                topology["contour_neighbor_jaccard_distance"]
            )
            coordination_values.append(
                topology["contour_coordination_change_max"]
            )
            rdf_values.append(
                topology["contour_rdf_l1_distance"]
            )
        except Exception as error:
            print(
                f"Could not calculate contour topology for "
                f"{trajectory_path}: {error}"
            )
            jaccard_values.append(np.nan)
            coordination_values.append(np.nan)
            rdf_values.append(np.nan)

    selected["contour_median_displacement_a"] = (
        median_displacements
    )
    selected["contour_median_force_delta_ev_a"] = (
        median_force_changes
    )
    selected["contour_neighbor_jaccard_distance"] = (
        jaccard_values
    )
    selected["contour_coordination_change_max"] = (
        coordination_values
    )
    selected["contour_rdf_l1_distance"] = rdf_values
    selected["contour_force_angle_deg"] = force_angle_values

    selected["contour_convergence_mev_per_atom"] = (
        pd.to_numeric(
            selected["energy_deviation_mev_per_atom"],
            errors="coerce",
        ).abs()
    )

    selected["material_slug"] = summary_row["material_slug"]
    selected["calculator"] = summary_row["calculator"]
    selected["beta"] = float(summary_row["beta"])

    return selected


def build_contour_frame_dataset(summary_rows):
    tables = []

    for _, summary_row in summary_rows.iterrows():
        table = contour_frame_table(summary_row)

        if not table.empty:
            tables.append(table)

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True, sort=False)


CONTOUR_DISPLACEMENT_PLOTS = [
    (
        "contour_neighbor_jaccard_distance",
        "neighbor_jaccard_distance",
        "Neighbor Jaccard distance",
        "Neighbor Jaccard distance",
        "jaccard_vs_displacement.png",
        "Neighbor Jaccard distance vs median displacement",
    ),
    (
        "contour_rdf_l1_distance",
        "rdf_l1_distance",
        "RDF L1 distance",
        "RDF L1 distance",
        "rdf_vs_displacement.png",
        "RDF L1 distance vs median displacement",
    ),
    (
        "contour_coordination_change_max",
        "coordination_change_max",
        "Maximum coordination-number change",
        "Maximum coordination-number change",
        "coordination_vs_displacement.png",
        "Coordination change vs median displacement",
    ),
    (
        "contour_median_force_delta_ev_a",
        "after_attack_after_relaxation_median_force_delta_ev_a",
        r"Median force change (eV/$\AA$)",
        r"Median force change (eV/$\AA$)",
        "delta_force_vs_displacement.png",
        "Median force change vs median displacement",
    ),
    (
        "contour_convergence_mev_per_atom",
        "after_attack_after_relaxation_steps",
        r"$|E-E_{\mathrm{target}}|$ (meV/atom)",
        "Relaxation steps",
        "convergence_vs_displacement.png",
        "Contour target error and attack convergence",
    ),
    (
        "contour_force_angle_deg",
        "after_attack_after_relaxation_force_angle_deg",
        "Median force-vector angle change (degrees)",
        "Median force-vector angle change (degrees)",
        "delta_force_angle_vs_displacement.png",
        "Force-angle change vs median displacement",
    ),
]


def clean_scatter_data(data, x_column, y_column):
    if data.empty:
        return pd.DataFrame()

    if x_column not in data.columns or y_column not in data.columns:
        return pd.DataFrame()

    clean = data.copy()
    clean[x_column] = pd.to_numeric(
        clean[x_column],
        errors="coerce",
    )
    clean[y_column] = pd.to_numeric(
        clean[y_column],
        errors="coerce",
    )

    return clean.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(subset=[x_column, y_column])


def draw_contour_scatter(
    axis,
    data,
    beta,
    metric,
    ylabel,
):
    beta_values = pd.to_numeric(
        data["beta"],
        errors="coerce",
    )

    beta_data = data[
        np.isclose(beta_values, beta)
    ].copy()

    for calculator in ["mace", "uma"]:
        selected = clean_scatter_data(
            beta_data[
                beta_data["calculator"] == calculator
            ],
            "contour_median_displacement_a",
            metric,
        )

        if selected.empty:
            continue

        axis.scatter(
            selected["contour_median_displacement_a"],
            selected[metric],
            s=18,
            alpha=0.28,
            color=CALC_COLORS[calculator],
            marker="o" if calculator == "mace" else "s",
            edgecolor="none",
            label=calculator.upper(),
        )

    axis.set_title(rf"$\beta={beta:.2f}$")
    axis.set_xlabel(
        r"Median displacement from initial structure ($\AA$)"
    )
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.30)


def draw_attack_scatter(
    axis,
    data,
    attack,
    metric,
    ylabel,
):
    attack_data = data[
        (data["attack_label"] == attack)
        & (~data["is_step_sweep"])
    ].copy()

    for calculator in ["mace", "uma"]:
        selected = clean_scatter_data(
            attack_data[
                attack_data["calculator"] == calculator
            ],
            "after_attack_after_relaxation_median_displacement_a",
            metric,
        )

        if selected.empty:
            continue

        axis.scatter(
            selected[
                "after_attack_after_relaxation_median_displacement_a"
            ],
            selected[metric],
            s=22,
            alpha=0.42,
            color=CALC_COLORS[calculator],
            marker="o" if calculator == "mace" else "s",
            edgecolor="none",
            label=calculator.upper(),
        )

    axis.set_title(attack)
    axis.set_xlabel(
        r"Median displacement from initial structure ($\AA$)"
    )
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.30)


def plot_contour_metric_vs_displacement(
    contour_data,
    attack_data,
    contour_metric,
    attack_metric,
    contour_ylabel,
    attack_ylabel,
    output_path,
    title,
):
    if contour_data.empty or attack_data.empty:
        return

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.2, 8.0),
        squeeze=False,
    )

    for column, beta in enumerate([0.00, 0.05, 0.10]):
        draw_contour_scatter(
            axes[0, column],
            contour_data,
            beta,
            contour_metric,
            contour_ylabel,
        )

        draw_attack_scatter(
            axes[1, column],
            attack_data,
            ATTACK_ORDER[column],
            attack_metric,
            attack_ylabel,
        )

    label_axes(axes.ravel())

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=CALC_COLORS["mace"],
            label="MACE",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            color=CALC_COLORS["uma"],
            label="UMA",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )

    fig.text(
        0.012,
        0.72,
        "Contour exploration",
        rotation=90,
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.012,
        0.28,
        "After attack and relaxation",
        rotation=90,
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    fig.suptitle(title, y=1.015, fontsize=13)
    fig.tight_layout(rect=[0.035, 0.03, 1.0, 0.94])

    fig.savefig(
        output_path,
        dpi=500,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_contour_displacement_plots(
    frame_data,
    attack_data,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for (
        contour_metric,
        attack_metric,
        contour_ylabel,
        attack_ylabel,
        filename,
        title,
    ) in CONTOUR_DISPLACEMENT_PLOTS:
        plot_contour_metric_vs_displacement(
            contour_data=frame_data,
            attack_data=attack_data,
            contour_metric=contour_metric,
            attack_metric=attack_metric,
            contour_ylabel=contour_ylabel,
            attack_ylabel=attack_ylabel,
            output_path=output_dir / filename,
            title=title,
        )


def contour_frame_stats(frame_data, metric):
    if frame_data.empty or metric not in frame_data.columns:
        return None

    values = pd.to_numeric(
        frame_data[metric],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if values.empty:
        return None

    return {
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def add_contour_band(ax, stats):
    if stats is None:
        return

    lower = max(0.0, stats["p05"])
    upper = stats["p95"]

    ax.axhspan(
        lower,
        upper,
        color=CONTOUR_BAND_COLOR,
        alpha=0.28,
        linewidth=0,
        label="contour p05-p95",
        zorder=0,
    )

    ax.text(
        0.02,
        0.94,
        f"contour p95={upper:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#1B7837",
    )


def attack_label_from_row(row):
    label = str(row.get("attack_label", "")).strip()
    if label:
        return label

    attack_type = str(row.get("attack_type", "")).lower()
    n_steps = int(float(row.get("n_steps", 1)))
    if attack_type == "fgsm" and n_steps > 1:
        return "I-FGSM"
    if attack_type == "fgsm":
        return "FGSM"
    if attack_type == "pgd":
        return "PGD"
    return attack_type.upper()


def attack_metric_table(attacks):
    rows = []

    for _, row in attacks.iterrows():
        before_displacement = attack_displacement(
            row,
            "before_forces.csv",
            "perturbed_forces.csv",
        )
        before_force_delta = attack_force_delta(
            row,
            "before_forces.csv",
            "perturbed_forces.csv",
        )

        final_displacement = attack_displacement(
            row,
            "before_forces.csv",
            "after_forces.csv",
        )
        final_force_delta = attack_force_delta(
            row,
            "before_forces.csv",
            "after_forces.csv",
        )

        run_id = str(row.get("run_id", ""))

        rows.append({
            "material_slug": row.get("material_slug"),
            "calculator": row.get("calculator"),
            "attack_label": attack_label_from_row(row),
            "epsilon": pd.to_numeric(
                row.get("epsilon"),
                errors="coerce",
            ),
            "n_steps": pd.to_numeric(
                row.get("n_steps"),
                errors="coerce",
            ),
            "is_step_sweep": "_steps" in run_id,

            "after_attack_before_relaxation_median_displacement_a": (
                float(np.median(before_displacement))
                if before_displacement.size
                else np.nan
            ),
            "after_attack_before_relaxation_median_force_delta_ev_a": (
                float(np.median(before_force_delta))
                if before_force_delta.size
                else np.nan
            ),
            "after_attack_after_relaxation_median_displacement_a": (
                float(np.median(final_displacement))
                if final_displacement.size
                else np.nan
            ),
            "after_attack_after_relaxation_median_force_delta_ev_a": (
                float(np.median(final_force_delta))
                if final_force_delta.size
                else np.nan
            ),
            "after_attack_after_relaxation_force_angle_deg": (
                attack_force_angle(
                    row,
                    "before_forces.csv",
                    "after_forces.csv",
                )
            ),
            "after_attack_after_relaxation_steps": pd.to_numeric(
                row.get("after_relax_steps"),
                errors="coerce",
            ),
            "neighbor_jaccard_distance": pd.to_numeric(
                row.get("neighbor_jaccard_distance"),
                errors="coerce",
            ),
            "coordination_change_max": pd.to_numeric(
                row.get("coordination_change_max"),
                errors="coerce",
            ),
            "rdf_l1_distance": pd.to_numeric(
                row.get("rdf_l1_distance"),
                errors="coerce",
            ),

            # Compatibility with existing contour/attack plots.
            "attack_median_displacement_a": (
                float(np.median(before_displacement))
                if before_displacement.size
                else np.nan
            ),
            "attack_median_force_delta_ev_a": (
                float(np.median(before_force_delta))
                if before_force_delta.size
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def draw_attack_panels(
    fig,
    axes,
    data,
    x_col,
    x_label,
    calculator,
    contour_frames,
    title,
    attacks_to_plot,
):
    disp_stats = contour_frame_stats(
        contour_frames,
        "contour_median_displacement_a",
    )
    force_stats = contour_frame_stats(
        contour_frames,
        "contour_median_force_delta_ev_a",
    )
    color = CALC_COLORS.get(calculator, "#333333")

    for col, attack in enumerate(attacks_to_plot):
        subset = data[data["attack_label"] == attack].copy()
        subset = subset[np.isfinite(subset[x_col])]
        subset = subset.sort_values(x_col)

        ax_disp = axes[0, col]
        ax_force = axes[1, col]

        add_contour_band(ax_disp, disp_stats)
        add_contour_band(ax_force, force_stats)

        if not subset.empty:
            grouped = subset.groupby(x_col, as_index=False).agg({
                "attack_median_displacement_a": "median",
                "attack_median_force_delta_ev_a": "median",
            })

            ax_disp.plot(
                grouped[x_col],
                grouped["attack_median_displacement_a"],
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=color,
                label="attack median",
                zorder=3,
            )

            ax_force.plot(
                grouped[x_col],
                grouped["attack_median_force_delta_ev_a"],
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=color,
                label="attack median",
                zorder=3,
            )

        ax_disp.set_title(attack)
        ax_disp.set_ylabel(r"Displacement ($\AA$)" if col == 0 else "")
        ax_force.set_ylabel(r"$\Delta$ force (eV/$\AA$)" if col == 0 else "")
        ax_force.set_xlabel(x_label)

        for ax in [ax_disp, ax_force]:
            style_numeric_axis(ax)
            ax.grid(True, axis="y")
            ax.margins(x=0.04)

        if x_col in ["epsilon", "n_steps"]:
            x_values = data[x_col].to_numpy(dtype=float)

            for axis in [ax_disp, ax_force]:
                apply_log_decade_axis(axis, "x", x_values, x_label)

        disp_y_values = clean_axis_values(ax_disp, "y")
        force_y_values = clean_axis_values(ax_force, "y")

        if disp_stats is not None:
            disp_y_values = np.concatenate([
                disp_y_values,
                np.asarray([disp_stats["p05"], disp_stats["p95"]], dtype=float),
            ])

        if force_stats is not None:
            force_y_values = np.concatenate([
                force_y_values,
                np.asarray([force_stats["p05"], force_stats["p95"]], dtype=float),
            ])

        apply_log_decade_axis(
            ax_disp,
            "y",
            disp_y_values,
            r"Displacement ($\AA$)" if col == 0 else "",
        )
        apply_log_decade_axis(
            ax_force,
            "y",
            force_y_values,
            r"$\Delta$ force (eV/$\AA$)" if col == 0 else "",
        )

        if subset.empty:
            for ax in [ax_disp, ax_force]:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle(title, y=1.08, fontsize=10)


def plot_contour_vs_attack(
    material_slug,
    calculator,
    contour_rows,
    contour_frames,
    attacks,
    output_dir,
):
    attacks = attacks[
        (attacks["material_slug"] == material_slug)
        & (attacks["calculator"] == calculator)
    ].copy()

    if attacks.empty:
        return pd.DataFrame()

    table = attack_metric_table(attacks)

    material_dir = output_dir / material_slug
    material_dir.mkdir(parents=True, exist_ok=True)

    epsilon_data = table[~table["is_step_sweep"]].copy()
    if not epsilon_data.empty:
        fig, axes = plt.subplots(2, 3, figsize=(8.8, 4.8), sharex=False, sharey="row")
        draw_attack_panels(
            fig=fig,
            axes=axes,
            data=epsilon_data,
            x_col="epsilon",
            x_label=r"$\epsilon$ ($\AA$)",
            calculator=calculator,
            contour_frames=contour_frames,
            title=f"{material_slug} {calculator.upper()}: attacks vs contour baseline by epsilon",
            attacks_to_plot=ATTACK_ORDER,
        )
        label_axes(axes)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(
            material_dir / f"{calculator}_contour_vs_attack_by_epsilon.png",
            bbox_inches="tight",
        )
        plt.close(fig)

    step_data = table[table["is_step_sweep"]].copy()
    if not step_data.empty:
        fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=False, sharey="row")
        draw_attack_panels(
            fig=fig,
            axes=axes,
            data=step_data,
            x_col="n_steps",
            x_label="n_steps",
            calculator=calculator,
            contour_frames=contour_frames,
            title=f"{material_slug} {calculator.upper()}: attacks vs contour baseline by n_steps",
            attacks_to_plot=["I-FGSM", "PGD"],
        )
        label_axes(axes)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(
            material_dir / f"{calculator}_contour_vs_attack_by_n_steps.png",
            bbox_inches="tight",
        )
        plt.close(fig)

    disp_stats = contour_frame_stats(
        contour_frames,
        "contour_median_displacement_a",
    )
    force_stats = contour_frame_stats(
        contour_frames,
        "contour_median_force_delta_ev_a",
    )

    if (
        contour_frames.empty
        or "energy_deviation_mev_per_atom"
        not in contour_frames.columns
    ):
        energy_values = np.array([], dtype=float)
    else:
        energy_values = pd.to_numeric(
            contour_frames["energy_deviation_mev_per_atom"],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna().to_numpy(dtype=float)

    table["contour_displacement_p05_a"] = np.nan if disp_stats is None else disp_stats["p05"]
    table["contour_displacement_p95_a"] = np.nan if disp_stats is None else disp_stats["p95"]
    table["contour_force_delta_p05_ev_a"] = np.nan if force_stats is None else force_stats["p05"]
    table["contour_force_delta_p95_ev_a"] = np.nan if force_stats is None else force_stats["p95"]
    table["contour_abs_energy_deviation_p95_mev_atom"] = (
        np.nan if energy_values.size == 0 else float(np.percentile(np.abs(energy_values), 95))
    )

    return table


def plot_six_panel(material_slug, calculator, rows, output_dir):
    loaded = []
    for _, row in rows.sort_values("beta", ascending=False).iterrows():
        metrics = read_csv(row["metrics_csv"])
        if metrics is None:
            continue
        loaded.append((float(row["beta"]), metrics))

    if not loaded:
        return

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.2), sharex=False)

    for beta, metrics in loaded:
        color = color_for_beta(beta)
        label = format_beta(beta)
        step = metrics["step"]

        axes[0, 0].plot(step, metrics["energy_deviation_mev_per_atom"], lw=1.0, color=color, label=label)
        axes[0, 1].hist(
            metrics["energy_deviation_mev_per_atom"].dropna(),
            bins=40,
            histtype="step",
            linewidth=1.6,
            color=color,
            label=label,
        )
        axes[0, 2].plot(step, metrics["step_size_a"], lw=1.0, color=color, label=label)
        axes[1, 0].plot(step, metrics["separation_distance_a"], lw=1.0, color=color, label=label)
        axes[1, 1].plot(step, metrics["curvature_1_per_a"], lw=1.0, color=color, label=label)
        axes[1, 2].plot(step, metrics["out_of_plane_angle_deg"], lw=1.0, color=color, label=label)

    labels = [
        (axes[0, 0], "Iteration #", r"$E - E_{\mathrm{contour}}$ (meV/atom)"),
        (axes[0, 1], r"$E - E_{\mathrm{contour}}$ (meV/atom)", "Frequency"),
        (axes[0, 2], "Iteration #", r"Step size ($\AA$)"),
        (axes[1, 0], "Iteration #", r"Separation distance ($\AA$)"),
        (axes[1, 1], "Iteration #", r"Curvature, $\kappa$ (1/$\AA$)"),
        (axes[1, 2], "Iteration #", r"Out-of-plane angle ($^\circ$)"),
    ]

    for ax, xlabel, ylabel in labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)

    axes[1, 1].set_yscale("symlog", linthresh=1e-3)

    axes[0, 0].axhline(0, color="#222222", lw=0.8, alpha=0.65)
    axes[0, 1].legend(loc="upper right")
    axes[0, 2].legend(loc="upper right")
    axes[1, 2].legend(loc="upper right")

    label_axes(axes)
    fig.suptitle(f"{material_slug} {calculator.upper()} contour exploration", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    material_dir = output_dir / material_slug
    material_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(material_dir / f"{calculator}_six_panel.png", bbox_inches="tight")
    plt.close(fig)


def plot_mace_vs_uma(material_slug, all_rows, output_dir):
    rows = all_rows[all_rows["material_slug"] == material_slug].copy()
    if rows.empty or set(rows["calculator"]) != {"mace", "uma"}:
        return

    grouped = rows.groupby(["calculator", "beta"], as_index=False).agg({
        "mean_abs_energy_deviation_mev_per_atom": "mean",
        "mean_step_size_a": "mean",
        "mean_curvature_1_per_a": "mean",
        "max_displacement_from_initial_a": "mean",
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.8), sharex=True)
    metrics = [
        ("mean_abs_energy_deviation_mev_per_atom", r"Mean $|E-E_c|$ (meV/atom)"),
        ("mean_step_size_a", r"Mean step size ($\AA$)"),
        ("mean_curvature_1_per_a", r"Mean curvature (1/$\AA$)"),
        ("max_displacement_from_initial_a", r"Max displacement ($\AA$)"),
    ]

    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for calculator, color in CALC_COLORS.items():
            subset = grouped[grouped["calculator"] == calculator].sort_values("beta")
            ax.plot(
                subset["beta"],
                subset[metric],
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=color,
                label=calculator.upper(),
            )
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)

    axes[0, 0].legend(loc="best")
    label_axes(axes)
    fig.suptitle(f"{material_slug}: MACE vs UMA contour exploration", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    material_dir = output_dir / material_slug
    material_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(material_dir / "mace_vs_uma_contour.png", bbox_inches="tight")
    plt.close(fig)


def clean_xy(data, x_col, y_col):
    subset = data[[x_col, y_col, "calculator"]].copy()
    subset[x_col] = pd.to_numeric(subset[x_col], errors="coerce")
    subset[y_col] = pd.to_numeric(subset[y_col], errors="coerce")
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
    subset = subset[(subset[x_col] >= 0) & (subset[y_col] >= 0)]
    return subset


def compact_axis_limits(values, upper_percentile=99.0, pad=0.18):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values >= 0]

    if values.size == 0:
        return 0.0, 1.0

    upper = float(np.percentile(values, upper_percentile))
    max_value = float(np.max(values))

    if upper <= 0:
        upper = max_value

    if upper <= 0:
        return 0.0, 1.0

    # Keep extreme outliers visible if there are only a few points.
    if max_value > upper * 8:
        upper = upper * 1.8
    else:
        upper = max_value

    return 0.0, upper * (1.0 + pad)


def symlog_linthresh(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]

    if values.size == 0:
        return 1e-6

    return max(float(np.percentile(values, 10)) * 0.35, 1e-9)


def style_global_comparison_axis(ax, data, x_col, y_col):
    x_values = data[x_col].to_numpy(dtype=float)
    y_values = data[y_col].to_numpy(dtype=float)

    x_linthresh = symlog_linthresh(x_values)
    y_linthresh = symlog_linthresh(y_values)

    ax.set_xscale("symlog", linthresh=x_linthresh)
    ax.set_yscale("symlog", linthresh=y_linthresh)

    x_left, x_right = compact_axis_limits(x_values)
    y_bottom, y_top = compact_axis_limits(y_values)

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)

    line_max = min(x_right, y_top)
    if line_max > 0:
        ax.plot(
            [0, line_max],
            [0, line_max],
            color="#555555",
            lw=1.0,
            linestyle="--",
            label="1:1",
            zorder=2,
        )

    ax.grid(True, which="major", alpha=0.34)
    ax.grid(True, which="minor", alpha=0.12)


def plot_one_global_panel(ax, data, x_col, y_col, xlabel, ylabel, title):
    data = clean_xy(data, x_col, y_col)

    if data.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return

    for calculator, color in CALC_COLORS.items():
        subset = data[data["calculator"] == calculator]
        if subset.empty:
            continue

        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=28,
            color=color,
            alpha=0.68,
            edgecolor="white",
            linewidth=0.35,
            label=calculator.upper(),
            zorder=3,
        )

    style_global_comparison_axis(ax, data, x_col, y_col)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_global_relaxation_state(records, output_dir, displacement_col, force_col, title, output_name):
    displacement = records[
        records["contour_displacement_p95_a"].notna()
        & records[displacement_col].notna()
    ].copy()

    force = records[
        records["contour_force_delta_p95_ev_a"].notna()
        & records[force_col].notna()
    ].copy()

    if displacement.empty and force.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))

    plot_one_global_panel(
        ax=axes[0],
        data=displacement,
        x_col="contour_displacement_p95_a",
        y_col=displacement_col,
        xlabel=r"Contour p95 ($\AA$)",
        ylabel=r"Attack median ($\AA$)",
        title="Displacement",
    )

    plot_one_global_panel(
        ax=axes[1],
        data=force,
        x_col="contour_force_delta_p95_ev_a",
        y_col=force_col,
        xlabel=r"Contour p95 (eV/$\AA$)",
        ylabel=r"Attack median (eV/$\AA$)",
        title=r"$\Delta$ force",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.03),
            frameon=False,
        )

    label_axes(axes)
    fig.suptitle(f"{title} vs contour exploration", y=1.08, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_dir / output_name, bbox_inches="tight")
    plt.close(fig)


def plot_relaxation_attack_grid_panel(
    ax,
    data,
    x_col,
    y_col,
    attack_label,
    xlabel,
    ylabel,
    row_label,
):
    subset = data[
        (data["attack_label"] == attack_label)
        & data[x_col].notna()
        & data[y_col].notna()
    ].copy()

    subset = clean_xy(subset, x_col, y_col)

    if subset.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(attack_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if row_label else "")
        return

    for calculator, color in CALC_COLORS.items():
        calc_subset = subset[subset["calculator"] == calculator]
        if calc_subset.empty:
            continue

        ax.scatter(
            calc_subset[x_col],
            calc_subset[y_col],
            s=22,
            color=color,
            alpha=0.68,
            edgecolor="white",
            linewidth=0.32,
            label=calculator.upper(),
            zorder=3,
        )

    style_global_comparison_axis(ax, subset, x_col, y_col)

    ax.set_title(attack_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if row_label else "")


def plot_global_relaxation_attack_grid(
    records,
    output_dir,
    x_col,
    before_col,
    after_col,
    xlabel,
    ylabel,
    title,
    output_name,
):
    if records.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6), sharex=False, sharey=False)

    rows = [
        ("After attack, before relaxation", before_col),
        ("After attack, after relaxation", after_col),
    ]

    for row_index, (row_title, y_col) in enumerate(rows):
        for col_index, attack in enumerate(ATTACK_ORDER):
            ax = axes[row_index, col_index]

            plot_relaxation_attack_grid_panel(
                ax=ax,
                data=records,
                x_col=x_col,
                y_col=y_col,
                attack_label=attack,
                xlabel=xlabel,
                ylabel=ylabel,
                row_label=(col_index == 0),
            )

            if col_index == 0:
                ax.text(
                    -0.32,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
        )

    label_axes(axes)
    fig.suptitle(title, y=1.06, fontsize=11)
    fig.tight_layout(rect=[0.04, 0, 1, 0.98])
    fig.savefig(output_dir / output_name, bbox_inches="tight")
    plt.close(fig)


def plot_global(records, output_dir):
    if records.empty:
        return

    plot_global_relaxation_state(
        records=records,
        output_dir=output_dir,
        displacement_col="after_attack_before_relaxation_median_displacement_a",
        force_col="after_attack_before_relaxation_median_force_delta_ev_a",
        title="After attack, before relaxation",
        output_name="global_after_attack_before_relaxation_vs_contour_exploration.png",
    )

    plot_global_relaxation_state(
        records=records,
        output_dir=output_dir,
        displacement_col="after_attack_after_relaxation_median_displacement_a",
        force_col="after_attack_after_relaxation_median_force_delta_ev_a",
        title="After attack, after relaxation",
        output_name="global_after_attack_after_relaxation_vs_contour_exploration.png",
    )

    plot_global_relaxation_attack_grid(
        records=records,
        output_dir=output_dir,
        x_col="contour_displacement_p95_a",
        before_col="after_attack_before_relaxation_median_displacement_a",
        after_col="after_attack_after_relaxation_median_displacement_a",
        xlabel=r"Contour p95 ($\AA$)",
        ylabel=r"Attack median ($\AA$)",
        title="Relaxation vs contour exploration by attack type: displacement",
        output_name="global_relaxation_vs_contour_exploration_by_attack_type_displacement.png",
    )

    plot_global_relaxation_attack_grid(
        records=records,
        output_dir=output_dir,
        x_col="contour_force_delta_p95_ev_a",
        before_col="after_attack_before_relaxation_median_force_delta_ev_a",
        after_col="after_attack_after_relaxation_median_force_delta_ev_a",
        xlabel=r"Contour p95 (eV/$\AA$)",
        ylabel=r"Attack median (eV/$\AA$)",
        title=r"Relaxation vs contour exploration by attack type: $\Delta$ force",
        output_name="global_relaxation_vs_contour_exploration_by_attack_type_delta_force.png",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mace-contour-dir", default=BASE_DIR / "outputs_mace" / "contour", type=Path)
    parser.add_argument("--uma-contour-dir", default=BASE_DIR / "outputs_uma" / "contour", type=Path)
    parser.add_argument("--comprehensive-dir", default=BASE_DIR / "outputs_comprehensive", type=Path)
    parser.add_argument("--output-dir", default=BASE_DIR / "outputs_comprehensive" / "contour", type=Path)
    args = parser.parse_args()

    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for contour_dir in [args.mace_contour_dir, args.uma_contour_dir]:
        data = summary_rows(contour_dir)
        if not data.empty:
            summaries.append(data)

    if not summaries:
        print("No contour summaries found.")
        return

    all_rows = pd.concat(summaries, ignore_index=True)
    all_rows.to_csv(
        args.output_dir / "contour_summary_combined.csv",
        index=False,
    )

    contour_frames = build_contour_frame_dataset(all_rows)

    attacks = load_attack_dataset(args.comprehensive_dir)
    attack_metrics = (
        attack_metric_table(attacks)
        if not attacks.empty
        else pd.DataFrame()
    )

    if not contour_frames.empty:
        contour_frames.to_csv(
            args.output_dir / "contour_frame_metrics.csv",
            index=False,
        )

        make_contour_displacement_plots(
            contour_frames,
            attack_metrics,
            args.output_dir,
        )

        for material_slug, material_frames in (
            contour_frames.groupby("material_slug")
        ):
            if (
                attack_metrics.empty
                or "material_slug" not in attack_metrics.columns
            ):
                material_attacks = pd.DataFrame()
            else:
                material_attacks = attack_metrics[
                    attack_metrics["material_slug"]
                    == material_slug
                ].copy()

            make_contour_displacement_plots(
                material_frames,
                material_attacks,
                args.output_dir / str(material_slug),
            )
    else:
        print("No contour trajectory frames were available.")

    comparison_tables = []

    for (material_slug, calculator), rows in all_rows.groupby(["material_slug", "calculator"]):
        plot_six_panel(material_slug, calculator, rows, args.output_dir)

        if not attacks.empty:
            if (
                contour_frames.empty
                or "material_slug" not in contour_frames.columns
                or "calculator" not in contour_frames.columns
            ):
                matching_frames = pd.DataFrame()
            else:
                matching_frames = contour_frames[
                    (contour_frames["material_slug"] == material_slug)
                    & (contour_frames["calculator"] == calculator)
                ].copy()

            table = plot_contour_vs_attack(
                material_slug=material_slug,
                calculator=calculator,
                contour_rows=rows,
                contour_frames=matching_frames,
                attacks=attacks,
                output_dir=args.output_dir,
            )

            if not table.empty:
                comparison_tables.append(table)

    for material_slug in sorted(all_rows["material_slug"].unique()):
        plot_mace_vs_uma(material_slug, all_rows, args.output_dir)

    tables_dir = args.output_dir / "comparison_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if comparison_tables:
        comparison_df = pd.concat(comparison_tables, ignore_index=True)
    else:
        comparison_df = pd.DataFrame()

    comparison_df.to_csv(tables_dir / "contour_vs_attack_comparisons.csv", index=False)

    if not comparison_df.empty:
        plot_global(comparison_df, args.output_dir)

    print(f"Saved contour plots to {args.output_dir}")


if __name__ == "__main__":
    main()