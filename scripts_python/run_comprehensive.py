#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import math
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd
from ase.io import read as read_structure


BASE_DIR = Path(__file__).resolve().parent.parent

CALCULATOR_COLORS = {
    "mace": "#0072B2",
    "uma": "#D55E00",
}

ATTACK_ORDER = ["FGSM", "I-FGSM", "PGD"]
STEP_ATTACK_ORDER = ["I-FGSM", "PGD"]
ATTACK_FOLDER = {
    "FGSM": "fgsm",
    "I-FGSM": "ifgsm",
    "PGD": "pgd",
}

MODEL_OFFSETS = {
    "mace": -0.18,
    "uma": 0.18,
}


def apply_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "grid.color": "#D7D7D7",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "legend.frameon": False,
    })


def save_figure(fig, output_base):
    output_base = Path(output_base)
    tighten_axes_for_publication(fig)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")


def format_epsilon_label(value):
    return f"{float(value):g}"


def sparse_tick_indices(count, max_labels=6):
    if count <= max_labels:
        return set(range(count))
    return set(np.linspace(0, count - 1, max_labels, dtype=int).tolist())


def style_epsilon_tick_labels(ax, rotate=False, max_labels=6):
    labels = ax.get_xticklabels()
    keep = sparse_tick_indices(len(labels), max_labels=max_labels)

    for index, label in enumerate(labels):
        label.set_visible(index in keep)
        label.set_horizontalalignment("center")

    ax.tick_params(axis="x", labelrotation=0, pad=2)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return value


def as_float(value):
    value = clean_value(value)
    if value is None:
        return None
    return float(value)


def as_int(value):
    value = clean_value(value)
    if value is None:
        return None
    return int(float(value))


def normalized_run_id(run_id):
    return str(run_id).replace("_mace_", "_").replace("_uma_", "_")


def slug_text(value):
    text = str(value).strip().lower()
    chars = []
    previous_underscore = False

    for char in text:
        if char.isalnum():
            chars.append(char)
            previous_underscore = False
        elif not previous_underscore:
            chars.append("_")
            previous_underscore = True

    return "".join(chars).strip("_")


def material_file_slug(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def read_material_rows(path):
    path = Path(path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def structure_path_for_material(row, structures_dir):
    mpid = str(row["mpid"]).strip()
    label = material_file_slug(row["material_label"])
    return Path(structures_dir) / f"{mpid}_{label}.cif"


def make_structure_summary(materials_path, structures_dir, output_dir):
    material_rows = read_material_rows(materials_path)
    if not material_rows:
        return

    rows = []
    missing = []

    for material in material_rows:
        path = structure_path_for_material(material, structures_dir)
        if not path.exists():
            missing.append({
                "category": material.get("category"),
                "material_label": material.get("material_label"),
                "mpid": material.get("mpid"),
                "reason": f"Missing structure file: {path}",
            })
            continue

        atoms = read_structure(path)
        symbols = atoms.get_chemical_symbols()
        elements = sorted(set(symbols))
        cell_lengths = atoms.cell.lengths()
        cell_angles = atoms.cell.angles()
        volume = float(atoms.get_volume())
        n_atoms = len(atoms)

        rows.append({
            "category": material.get("category"),
            "material_label": material.get("material_label"),
            "formula": material.get("formula"),
            "mpid": material.get("mpid"),
            "n_atoms": n_atoms,
            "n_elements": len(elements),
            "elements": ";".join(elements),
            "volume_a3": volume,
            "volume_per_atom_a3": volume / n_atoms if n_atoms else np.nan,
            "cell_a": float(cell_lengths[0]),
            "cell_b": float(cell_lengths[1]),
            "cell_c": float(cell_lengths[2]),
            "cell_alpha": float(cell_angles[0]),
            "cell_beta": float(cell_angles[1]),
            "cell_gamma": float(cell_angles[2]),
        })

    output_dir = Path(output_dir)
    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(output_dir / "materials_summary_combined.csv", index=False)

        by_category = summary.groupby("category", as_index=False).agg({
            "material_label": "count",
            "n_atoms": ["median", "min", "max"],
            "n_elements": "median",
            "volume_per_atom_a3": ["median", "min", "max"],
        })

        by_category.columns = [
            "category",
            "n_materials",
            "median_atoms",
            "min_atoms",
            "max_atoms",
            "median_n_elements",
            "median_volume_per_atom_a3",
            "min_volume_per_atom_a3",
            "max_volume_per_atom_a3",
        ]

        by_category.to_csv(output_dir / "structure_summary_by_category.csv", index=False)

    if missing:
        pd.DataFrame(missing).to_csv(output_dir / "structure_summary_missing.csv", index=False)


def material_info(row, run_dir):
    material_label = clean_value(row.get("material_label"))
    material_slug = clean_value(row.get("material_slug"))

    if material_slug is None and run_dir.parent.name not in ["outputs_mace", "outputs_uma"]:
        material_slug = run_dir.parent.name

    if material_label is None:
        material_label = material_slug

    if material_slug is None:
        input_path = clean_value(row.get("input_path")) or "material"
        material_slug = slug_text(Path(str(input_path)).stem)

    return str(material_label), str(material_slug)


def attack_label(row):
    attack_type = str(row.get("attack_type", "")).strip().lower()
    run_id = str(row.get("run_id", "")).lower()
    n_steps = as_int(row.get("n_steps")) or 1

    if "_ifgsm_" in run_id:
        return "I-FGSM"
    if attack_type == "fgsm" and n_steps > 1:
        return "I-FGSM"
    if attack_type == "fgsm":
        return "FGSM"
    if attack_type == "pgd":
        return "PGD"
    return attack_type.upper()


def force_column(data):
    for column in data.columns:
        if column.startswith("Max Force"):
            return column
    return None


def relaxation_steps(path, relax_fmax):
    data = read_csv(path)
    if data is None or data.empty or "Step" not in data.columns:
        return None, None

    steps = int(data["Step"].iloc[-1])
    column = force_column(data)
    converged = None

    if column is not None and relax_fmax is not None:
        final_force = float(data[column].iloc[-1])
        converged = final_force <= relax_fmax

    return steps, converged


def resolve_run_dir(base_dir, row):
    run_id = str(row["run_id"])
    candidate = Path(base_dir) / run_id
    if candidate.exists():
        return candidate

    for column in [
        "before_force_csv",
        "perturbed_force_csv",
        "after_force_csv",
        "before_relax_traj",
        "after_attack_relax_traj",
        "history_file",
    ]:
        value = clean_value(row.get(column))
        if value is not None:
            path = Path(str(value))
            if path.exists():
                return path.parent

    return candidate


def load_summary(summary_path, base_dir, calculator):
    summary = read_csv(summary_path)
    if summary is None or summary.empty:
        return [], [f"Missing or empty summary: {summary_path}"]

    records = []
    missing = []

    for _, row in summary.iterrows():
        if str(row.get("status", "")).strip().lower() != "success":
            continue

        run_dir = resolve_run_dir(base_dir, row)
        relax_fmax = as_float(row.get("relax_fmax"))
        material_label, material_slug = material_info(row, run_dir)

        before_steps, before_converged = relaxation_steps(
            run_dir / "before_attack_relaxation_data.csv",
            relax_fmax,
        )
        after_steps, after_converged = relaxation_steps(
            run_dir / "after_attack_relaxation_data.csv",
            relax_fmax,
        )

        records.append({
            "run_id": str(row["run_id"]),
            "material_label": material_label,
            "material_slug": material_slug,
            "logical_run_id": normalized_run_id(row["run_id"]),
            "calculator": calculator,
            "attack_label": attack_label(row),
            "epsilon": as_float(row.get("epsilon")),
            "n_steps": as_int(row.get("n_steps")),
            "alpha": as_float(row.get("alpha")),
            "relax_fmax": relax_fmax,
            "run_dir": str(run_dir),
            "before_relax_steps": before_steps,
            "before_relax_converged": before_converged,
            "after_relax_steps": after_steps,
            "after_relax_converged": after_converged,
            "mean_displacement": as_float(row.get("mean_displacement")),
            "max_displacement": as_float(row.get("max_displacement")),
            "final_energy": as_float(row.get("final_energy")),
            "topology_edge_changes_csv": clean_value(row.get("topology_edge_changes_csv")),
            "neighbor_edges_before": as_float(row.get("neighbor_edges_before")),
            "neighbor_edges_after": as_float(row.get("neighbor_edges_after")),
            "neighbor_edges_added": as_float(row.get("neighbor_edges_added")),
            "neighbor_edges_removed": as_float(row.get("neighbor_edges_removed")),
            "neighbor_edge_change_count": as_float(row.get("neighbor_edge_change_count")),
            "neighbor_jaccard_distance": as_float(row.get("neighbor_jaccard_distance")),
            "coordination_change_mean": as_float(row.get("coordination_change_mean")),
            "coordination_change_max": as_float(row.get("coordination_change_max")),
            "rdf_l1_distance": as_float(row.get("rdf_l1_distance")),
        })

    return records, missing


def merge_atom_csvs(run_dir, before_name, after_name, required_columns):
    before_path = Path(run_dir) / before_name
    after_path = Path(run_dir) / after_name

    before = read_csv(before_path)
    after = read_csv(after_path)

    if before is None:
        return None, f"Missing {before_name}"
    if after is None:
        return None, f"Missing {after_name}"

    required = set(required_columns) | {"atom_index"}
    if not required.issubset(before.columns):
        missing = sorted(required - set(before.columns))
        return None, f"{before_name} missing columns: {missing}"
    if not required.issubset(after.columns):
        missing = sorted(required - set(after.columns))
        return None, f"{after_name} missing columns: {missing}"

    merged = before[["atom_index"] + required_columns].merge(
        after[["atom_index"] + required_columns],
        on="atom_index",
        suffixes=("_before", "_after"),
    )

    if merged.empty:
        return None, "No matching atom_index rows"

    return merged, None


def force_delta_values(run_dir, before_name, after_name):
    merged, reason = merge_atom_csvs(
        run_dir,
        before_name,
        after_name,
        ["fx", "fy", "fz"],
    )
    if merged is None:
        return None, reason

    before_forces = merged[["fx_before", "fy_before", "fz_before"]].to_numpy()
    after_forces = merged[["fx_after", "fy_after", "fz_after"]].to_numpy()
    return np.linalg.norm(after_forces - before_forces, axis=1), None


def displacement_values(run_dir, before_name, after_name):
    merged, reason = merge_atom_csvs(
        run_dir,
        before_name,
        after_name,
        ["x", "y", "z"],
    )
    if merged is None:
        return None, reason

    before_xyz = merged[["x_before", "y_before", "z_before"]].to_numpy()
    after_xyz = merged[["x_after", "y_after", "z_after"]].to_numpy()
    return np.linalg.norm(after_xyz - before_xyz, axis=1), None


def model_legend_handles():
    return [
        plt.Line2D(
            [0],
            [0],
            color=CALCULATOR_COLORS["mace"],
            lw=7,
            alpha=0.72,
            label="MACE",
        ),
        plt.Line2D(
            [0],
            [0],
            color=CALCULATOR_COLORS["uma"],
            lw=7,
            alpha=0.72,
            label="UMA",
        ),
    ]


def apply_shared_figure_header(fig, subtitle=None, left=0.05):
    if subtitle:
        fig.legend(
            handles=model_legend_handles(),
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.045),
            borderaxespad=0.0,
        )
        fig.suptitle(subtitle, y=1.005, fontsize=9)
        fig.tight_layout(rect=[left, 0.00, 1.00, 0.955])
    else:
        fig.legend(
            handles=model_legend_handles(),
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.015),
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=[left, 0.00, 1.00, 0.975])


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


def clean_numeric_array(values):
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    return values.to_numpy(dtype=float)


def variability_radius(values):
    values = clean_numeric_array(values)
    if len(values) < 2:
        return 0.0

    q25, q75 = np.percentile(values, [25, 75])
    return max(float((q75 - q25) / 2.0), 0.0)


PARAMETRIC_AXIS_PERCENTILE = 95


def percentile_axis_limit(values, percentile=PARAMETRIC_AXIS_PERCENTILE, pad=0.16):
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    values = values[np.isfinite(values)]

    if values.empty:
        return None

    data = values.to_numpy(dtype=float)

    if len(data) == 1:
        center = float(data[0])
        span = max(abs(center) * 0.20, 1e-9)
        if center >= 0 and center - span < 0:
            return 0.0, center + span
        return center - span, center + span

    lower = float(np.percentile(data, 2))
    upper = float(np.percentile(data, percentile))

    if not np.isfinite(lower) or not np.isfinite(upper):
        return None

    if upper <= lower:
        center = float(np.median(data))
        span = max(abs(center) * 0.20, float(np.std(data)) * 2.0, 1e-9)
        if center >= 0 and center - span < 0:
            return 0.0, center + span
        return center - span, center + span

    span = upper - lower
    low = lower - pad * span
    high = upper + pad * span

    if np.nanmin(data) >= 0 and low < 0:
        low = 0.0 if np.nanmin(data) < 0.08 * span else max(0.0, low)

    return low, high


def parametric_axis_limits(data, percentile=PARAMETRIC_AXIS_PERCENTILE):
    if data.empty:
        return None, None

    x_limits = percentile_axis_limit(data["x"], percentile=percentile)
    y_limits = percentile_axis_limit(data["y"], percentile=percentile)

    return x_limits, y_limits


def minimum_visible_radius(limits):
    if limits is None:
        return 0.0

    low, high = limits
    span = float(high - low)
    if not np.isfinite(span) or span <= 0:
        return 0.0

    return 0.004 * span


def style_numeric_axis(ax, xbins=4, ybins=5):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins, prune=None))

    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        axis.set_major_formatter(formatter)

    ax.tick_params(axis="both", labelsize=8, pad=2)


def _artist_values_for_axis(ax, axis_name):
    values = []

    for line in ax.lines:
        raw = line.get_xdata(orig=False) if axis_name == "x" else line.get_ydata(orig=False)
        try:
            values.extend(np.asarray(raw, dtype=float).ravel().tolist())
        except Exception:
            pass

    for collection in ax.collections:
        try:
            offsets = collection.get_offsets()
            if len(offsets):
                column = 0 if axis_name == "x" else 1
                values.extend(np.asarray(offsets[:, column], dtype=float).ravel().tolist())
        except Exception:
            pass

    cleaned = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return np.array([], dtype=float)
    return cleaned.to_numpy(dtype=float)


def _tight_limit(values, pad=0.14, lower_percentile=5, upper_percentile=95):
    values = clean_numeric_array(values)
    if len(values) == 0:
        return None

    if np.allclose(values, values[0]):
        center = float(values[0])
        span = max(abs(center) * 0.20, 1e-9)
        if center >= 0 and center - span < 0:
            return 0.0, center + span
        return center - span, center + span

    low = float(np.percentile(values, lower_percentile))
    high = float(np.percentile(values, upper_percentile))
    span = high - low

    if span <= 0 or not np.isfinite(span):
        return None

    low -= pad * span
    high += pad * span

    if np.nanmin(values) >= 0 and low < 0:
        low = 0.0 if np.nanmin(values) < 0.08 * span else max(0.0, low)

    return low, high


def tighten_axes_for_publication(fig):
    for ax in fig.axes:
        if not ax.has_data():
            continue

        y_limits = _tight_limit(_artist_values_for_axis(ax, "y"))
        if y_limits is not None:
            ax.set_ylim(*y_limits)

        xlabel = ax.get_xlabel().lower()
        if "displacement" in xlabel or "rdf" in xlabel:
            x_limits = _tight_limit(_artist_values_for_axis(ax, "x"))
            if x_limits is not None:
                ax.set_xlim(*x_limits)


def metric_distribution(row, getter):
    values, reason = getter(row)
    if values is None:
        return None, reason

    values = clean_numeric_array(values)
    if len(values) == 0:
        return None, "No finite values"

    return (float(np.median(values)), values), None


def scalar_distribution(row, column):
    value = row.get(column)
    if value is None or pd.isna(value):
        return None, f"Missing {column}"

    value = float(value)
    return (value, np.array([value], dtype=float)), None


def parametric_rows(records, x_getter, y_getter, bubble_col, missing_rows, figure_name):
    rows = []

    for _, row in records.iterrows():
        x_result, x_reason = x_getter(row)
        y_result, y_reason = y_getter(row)
        bubble_value = row.get(bubble_col)

        if x_result is None or y_result is None:
            missing_rows.append({
                "figure": figure_name,
                "run_id": row["run_id"],
                "reason": x_reason or y_reason,
            })
            continue

        if bubble_value is None or pd.isna(bubble_value):
            missing_rows.append({
                "figure": figure_name,
                "run_id": row["run_id"],
                "reason": f"Missing {bubble_col}",
            })
            continue

        x_center, x_values = x_result
        y_center, y_values = y_result

        rows.append({
            "run_id": row["run_id"],
            "calculator": row["calculator"],
            "attack_label": row["attack_label"],
            "bubble": float(bubble_value),
            "x": float(x_center),
            "y": float(y_center),
            "x_values": x_values,
            "y_values": y_values,
        })

    return pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "calculator",
            "attack_label",
            "bubble",
            "x",
            "y",
            "x_values",
            "y_values",
        ],
    )


def draw_parametric_panel(
    ax,
    data,
    attack,
    x_label,
    y_label,
    show_ylabel,
    x_limits=None,
    y_limits=None,
):
    subset = data[data["attack_label"] == attack].copy()

    if not subset.empty:
        panel_x_limits, panel_y_limits = parametric_axis_limits(subset)
        if x_limits is None:
            x_limits = panel_x_limits
        if y_limits is None:
            y_limits = panel_y_limits

    if subset.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(attack)
        ax.set_xlabel(x_label)
        if show_ylabel:
            ax.set_ylabel(y_label)
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        style_numeric_axis(ax)
        ax.grid(True, alpha=0.35)
        return

    min_x_radius = minimum_visible_radius(x_limits)
    min_y_radius = minimum_visible_radius(y_limits)

    x_span = None if x_limits is None else float(x_limits[1] - x_limits[0])
    y_span = None if y_limits is None else float(y_limits[1] - y_limits[0])

    for calculator, color in CALCULATOR_COLORS.items():
        calc_data = subset[subset["calculator"] == calculator].copy()
        if calc_data.empty:
            continue

        first_label = True

        for _, group in calc_data.groupby("bubble", sort=True):
            x_values = np.concatenate([
                clean_numeric_array(values)
                for values in group["x_values"]
            ])
            y_values = np.concatenate([
                clean_numeric_array(values)
                for values in group["y_values"]
            ])

            if len(x_values) == 0 or len(y_values) == 0:
                continue

            x_center = float(np.median(x_values))
            y_center = float(np.median(y_values))

            if len(group) >= 3:
                x_radius = max(variability_radius(x_values), min_x_radius)
                y_radius = max(variability_radius(y_values), min_y_radius)

                if x_span is not None and x_span > 0:
                    x_radius = min(x_radius, 0.055 * x_span)
                if y_span is not None and y_span > 0:
                    y_radius = min(y_radius, 0.055 * y_span)

                ellipse = Ellipse(
                    xy=(x_center, y_center),
                    width=2.0 * x_radius,
                    height=2.0 * y_radius,
                    angle=0.0,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.8,
                    alpha=0.11,
                    clip_on=True,
                    zorder=1,
                )
                ax.add_patch(ellipse)

            ax.scatter(
                [x_center],
                [y_center],
                s=24,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
                label=calculator.upper() if first_label else None,
            )

            first_label = False

    ax.set_title(attack)
    ax.set_xlabel(x_label)
    if show_ylabel:
        ax.set_ylabel(y_label)

    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    style_numeric_axis(ax)
    ax.grid(True, alpha=0.35)
    ax.margins(x=0.03, y=0.05)


def make_parametric_state_figure(
    records,
    output_dir,
    figure_name,
    title,
    x_label,
    y_label,
    bubble_label,
    attacks_to_plot,
    x_getters,
    y_getters,
):
    missing_rows = []
    rows = [
        ("After attack, before relaxation", x_getters[0], y_getters[0]),
        ("After attack, after relaxation", x_getters[1], y_getters[1]),
    ]

    n_cols = len(attacks_to_plot)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(5.2 * n_cols, 9.0),
        sharex=False,
        sharey=False,
    )

    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)

    panel_index = 0
    any_data = False

    for row_index, (row_title, x_getter, y_getter) in enumerate(rows):
        data = parametric_rows(
            records=records,
            x_getter=x_getter,
            y_getter=y_getter,
            bubble_col="epsilon" if "epsilon" in figure_name else "n_steps",
            missing_rows=missing_rows,
            figure_name=figure_name,
        )

        if not data.empty:
            any_data = True

        x_limits, y_limits = None, None

        for col_index, attack in enumerate(attacks_to_plot):
            ax = axes[row_index, col_index]
            draw_parametric_panel(
                ax=ax,
                data=data,
                attack=attack,
                x_label=x_label,
                y_label=y_label,
                show_ylabel=(col_index == 0),
                x_limits=x_limits,
                y_limits=y_limits,
            )

            ax.title.set_fontsize(13)
            ax.xaxis.label.set_fontsize(12)
            ax.yaxis.label.set_fontsize(12)
            style_numeric_axis(ax, xbins=4, ybins=5)

            if col_index == 0:
                ax.text(
                    -0.32,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )

            add_panel_label(ax, chr(ord("A") + panel_index))
            panel_index += 1

    if not any_data:
        plt.close(fig)
        return missing_rows

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 1.045),
            frameon=False,
            title=f"Grouped by {bubble_label}",
            fontsize=11,
            title_fontsize=11,
            handlelength=1.9,
            columnspacing=1.25,
            handletextpad=0.55,
        )

    fig.suptitle(title, y=1.095, fontsize=15)
    fig.text(
        0.995,
        0.008,
        f"Axes capped at p{PARAMETRIC_AXIS_PERCENTILE} for readability",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=[0.07, 0.04, 1.00, 0.94])
    save_figure(fig, output_dir / figure_name)
    plt.close(fig)

    return missing_rows


def make_parametric_figure_set(records, output_dir, suffix, attacks_to_plot, bubble_label):
    convergence_displacement_missing = make_parametric_state_figure(
        records=records,
        output_dir=output_dir,
        figure_name=f"figure_7_convergence_vs_displacement_by_{suffix}",
        title=f"Convergence vs displacement by {bubble_label}",
        x_label=r"Median displacement ($\AA$)",
        y_label="Relaxation steps",
        bubble_label=bubble_label,
        attacks_to_plot=attacks_to_plot,
        x_getters=[
            lambda row: metric_distribution(row, lambda item: displacement_values(
                item["run_dir"],
                "before_forces.csv",
                "perturbed_forces.csv",
            )),
            lambda row: metric_distribution(row, lambda item: displacement_values(
                item["run_dir"],
                "before_forces.csv",
                "after_forces.csv",
            )),
        ],
        y_getters=[
            lambda row: scalar_distribution(row, "after_relax_steps"),
            lambda row: scalar_distribution(row, "after_relax_steps"),
        ],
    )

    convergence_force_missing = make_parametric_state_figure(
        records=records,
        output_dir=output_dir,
        figure_name=f"figure_8_convergence_vs_delta_force_by_{suffix}",
        title=f"Convergence vs delta force by {bubble_label}",
        x_label=r"Median $\Delta$ force (eV/$\AA$)",
        y_label="Relaxation steps",
        bubble_label=bubble_label,
        attacks_to_plot=attacks_to_plot,
        x_getters=[
            lambda row: metric_distribution(row, lambda item: force_delta_values(
                item["run_dir"],
                "before_forces.csv",
                "perturbed_forces.csv",
            )),
            lambda row: metric_distribution(row, lambda item: force_delta_values(
                item["run_dir"],
                "before_forces.csv",
                "after_forces.csv",
            )),
        ],
        y_getters=[
            lambda row: scalar_distribution(row, "after_relax_steps"),
            lambda row: scalar_distribution(row, "after_relax_steps"),
        ],
    )

    force_displacement_missing = make_parametric_state_figure(
        records=records,
        output_dir=output_dir,
        figure_name=f"figure_9_delta_force_vs_displacement_by_{suffix}",
        title=f"Delta force vs displacement by {bubble_label}",
        x_label=r"Median displacement ($\AA$)",
        y_label=r"Median $\Delta$ force (eV/$\AA$)",
        bubble_label=bubble_label,
        attacks_to_plot=attacks_to_plot,
        x_getters=[
            lambda row: metric_distribution(row, lambda item: displacement_values(
                item["run_dir"],
                "before_forces.csv",
                "perturbed_forces.csv",
            )),
            lambda row: metric_distribution(row, lambda item: displacement_values(
                item["run_dir"],
                "before_forces.csv",
                "after_forces.csv",
            )),
        ],
        y_getters=[
            lambda row: metric_distribution(row, lambda item: force_delta_values(
                item["run_dir"],
                "before_forces.csv",
                "perturbed_forces.csv",
            )),
            lambda row: metric_distribution(row, lambda item: force_delta_values(
                item["run_dir"],
                "before_forces.csv",
                "after_forces.csv",
            )),
        ],
    )

    return (
        convergence_displacement_missing
        + convergence_force_missing
        + force_displacement_missing
    )


def collect_box_data(records, attack, value_getter, missing_rows):
    attack_records = records[records["attack_label"] == attack].copy()
    epsilons = sorted(attack_records["epsilon"].dropna().unique())

    positions = []
    values = []
    colors = []
    point_x = []
    point_y = []

    rng = np.random.default_rng(12345)

    for i, epsilon in enumerate(epsilons, start=1):
        for calculator in ["mace", "uma"]:
            rowset = attack_records[
                (attack_records["epsilon"] == epsilon)
                & (attack_records["calculator"] == calculator)
            ]

            box_values = []
            for _, row in rowset.iterrows():
                row_values, reason = value_getter(row)
                if row_values is None:
                    missing_rows.append({
                        "attack": attack,
                        "calculator": calculator,
                        "epsilon": epsilon,
                        "run_id": row["run_id"],
                        "reason": reason,
                    })
                else:
                    box_values.extend(row_values.tolist())

            if box_values:
                position = i + MODEL_OFFSETS[calculator]
                positions.append(position)
                values.append(box_values)
                colors.append(CALCULATOR_COLORS[calculator])

                jitter = rng.normal(loc=0.0, scale=0.010, size=len(box_values))
                point_x.extend((position + jitter).tolist())
                point_y.extend(box_values)

    return epsilons, positions, values, colors, point_x, point_y


def draw_grouped_boxplot(ax, records, attack, value_getter, ylabel, missing_rows):
    epsilons, positions, values, colors, point_x, point_y = collect_box_data(
        records,
        attack,
        value_getter,
        missing_rows,
    )

    if not values:
        ax.text(0.5, 0.5, "No plottable data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    if point_x and point_y:
        ax.scatter(
            point_x,
            point_y,
            s=6,
            color="#222222",
            alpha=0.14,
            linewidths=0,
            zorder=1,
        )

    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        zorder=2,
        medianprops={"color": "#111111", "linewidth": 1.5},
        capprops={"color": "#444444", "linewidth": 0.9},
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    tick_positions = list(range(1, len(epsilons) + 1))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([format_epsilon_label(epsilon) for epsilon in epsilons])
    style_epsilon_tick_labels(ax, rotate=len(epsilons) >= 6)
    ax.set_xlabel(r"$\epsilon$ ($\AA$)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(x=0.03)

    return True


def plot_convergence_panel(ax, records, attack, step_col, conv_col):
    attack_records = records[records["attack_label"] == attack].copy()
    if attack_records.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    epsilons = sorted(attack_records["epsilon"].dropna().unique())
    epsilon_positions = {epsilon: index + 1 for index, epsilon in enumerate(epsilons)}

    for calculator, color in CALCULATOR_COLORS.items():
        data = attack_records[
            (attack_records["calculator"] == calculator)
            & attack_records[step_col].notna()
        ].sort_values("epsilon")

        if data.empty:
            continue

        grouped = data.groupby("epsilon", as_index=False)[step_col].mean()
        grouped["epsilon_position"] = grouped["epsilon"].map(epsilon_positions)

        ax.plot(
            grouped["epsilon_position"],
            grouped[step_col],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=calculator.upper(),
        )

        not_converged = data[data[conv_col] == False].copy()
        if not not_converged.empty:
            not_converged["epsilon_position"] = not_converged["epsilon"].map(epsilon_positions)

            ax.scatter(
                not_converged["epsilon_position"],
                not_converged[step_col],
                s=45,
                facecolors="none",
                edgecolors=color,
                linewidths=1.4,
                zorder=3,
            )

    tick_positions = list(range(1, len(epsilons) + 1))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([format_epsilon_label(epsilon) for epsilon in epsilons])
    style_epsilon_tick_labels(ax, rotate=len(epsilons) >= 6)
    ax.set_xlabel(r"$\epsilon$ ($\AA$)")
    ax.set_ylabel("Relaxation steps")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(x=0.03)

    return True


def make_convergence_figure(records, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.0), sharex=False, sharey=False)

    rows = [
        ("before_relax_steps", "before_relax_converged", "Relaxation before attack"),
        ("after_relax_steps", "after_relax_converged", "Relaxation after attack"),
    ]

    panel_index = 0
    for row_index, (step_col, conv_col, row_title) in enumerate(rows):
        for col_index, attack in enumerate(ATTACK_ORDER):
            ax = axes[row_index, col_index]
            plot_convergence_panel(ax, records, attack, step_col, conv_col)

            if row_index == 0:
                ax.set_title(attack)

            if col_index == 0:
                ax.text(
                    -0.33,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

            add_panel_label(ax, chr(ord("A") + panel_index))
            panel_index += 1

    apply_shared_figure_header(fig, left=0.03)
    save_figure(fig, output_dir / "figure_1_convergence_by_epsilon")
    plt.close(fig)


def bootstrap_median_ci(values, confidence=95, n_bootstrap=1000, seed=12345):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    median = float(np.median(values))

    if len(values) == 1:
        return median, median, median

    rng = np.random.default_rng(seed)
    boot_medians = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_medians.append(np.median(sample))

    alpha = (100 - confidence) / 2
    lower = float(np.percentile(boot_medians, alpha))
    upper = float(np.percentile(boot_medians, 100 - alpha))

    return median, lower, upper


def draw_grouped_ci(ax, records, attack, value_getter, ylabel, missing_rows):
    epsilons, positions, values, colors, point_x, point_y = collect_box_data(
        records,
        attack,
        value_getter,
        missing_rows,
    )

    if not values:
        ax.text(0.5, 0.5, "No plottable data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    series = {
        "mace": {"x": [], "median": [], "lower": [], "upper": []},
        "uma": {"x": [], "median": [], "lower": [], "upper": []},
    }

    for position, box_values in zip(positions, values):
        ci = bootstrap_median_ci(box_values)
        if ci is None:
            continue

        median, lower, upper = ci
        center_position = round(position)
        offset = position - center_position
        calculator = min(
            MODEL_OFFSETS,
            key=lambda name: abs(offset - MODEL_OFFSETS[name]),
        )

        x_value = round(position - MODEL_OFFSETS[calculator])
        series[calculator]["x"].append(x_value)
        series[calculator]["median"].append(median)
        series[calculator]["lower"].append(lower)
        series[calculator]["upper"].append(upper)

    for calculator, data in series.items():
        if not data["x"]:
            continue

        x = np.asarray(data["x"], dtype=float)
        median = np.asarray(data["median"], dtype=float)
        lower = np.asarray(data["lower"], dtype=float)
        upper = np.asarray(data["upper"], dtype=float)
        color = CALCULATOR_COLORS[calculator]

        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

        ax.plot(
            x,
            median,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=calculator.upper(),
        )

    tick_positions = list(range(1, len(epsilons) + 1))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([format_epsilon_label(epsilon) for epsilon in epsilons])
    style_epsilon_tick_labels(ax, rotate=len(epsilons) >= 6)
    ax.set_xlabel(r"$\epsilon$ ($\AA$)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(x=0.03)

    return True


def make_ci_figure(records, output_dir, figure_name, ylabel, rows):
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2), sharex=False, sharey=False)

    all_missing = []
    panel_index = 0

    for row_index, (row_title, getter_factory) in enumerate(rows):
        for col_index, attack in enumerate(ATTACK_ORDER):
            ax = axes[row_index, col_index]
            attack_missing = []

            draw_grouped_ci(
                ax=ax,
                records=records,
                attack=attack,
                value_getter=getter_factory(),
                ylabel=ylabel,
                missing_rows=attack_missing,
            )

            for missing in attack_missing:
                missing["figure"] = figure_name
                missing["panel"] = f"{row_title} / {attack}"
            all_missing.extend(attack_missing)

            if row_index == 0:
                ax.set_title(attack)

            if col_index == 0:
                ax.text(
                    -0.33,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

            add_panel_label(ax, chr(ord("A") + panel_index))
            panel_index += 1

    apply_shared_figure_header(
        fig,
        subtitle="Line = median, shaded band = 95% CI",
        left=0.03,
    )
    save_figure(fig, output_dir / figure_name)
    plt.close(fig)

    return all_missing


def make_distribution_figure(records, output_dir, figure_name, ylabel, rows):
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2), sharex=False, sharey=False)

    all_missing = []
    panel_index = 0

    for row_index, (row_title, getter_factory) in enumerate(rows):
        for col_index, attack in enumerate(ATTACK_ORDER):
            ax = axes[row_index, col_index]
            attack_missing = []

            draw_grouped_boxplot(
                ax=ax,
                records=records,
                attack=attack,
                value_getter=getter_factory(),
                ylabel=ylabel,
                missing_rows=attack_missing,
            )
            all_missing.extend(attack_missing)

            if row_index == 0:
                ax.set_title(attack)

            if col_index == 0:
                ax.text(
                    -0.33,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

            add_panel_label(ax, chr(ord("A") + panel_index))
            panel_index += 1

    apply_shared_figure_header(fig, left=0.03)
    save_figure(fig, output_dir / figure_name)
    plt.close(fig)

    return all_missing


def collect_box_data_by_steps(records, attack, epsilon, value_getter, missing_rows):
    attack_records = records[
        (records["attack_label"] == attack)
        & (records["epsilon"] == float(epsilon))
    ].copy()
    steps = sorted(attack_records["n_steps"].dropna().unique())

    positions = []
    values = []
    colors = []
    point_x = []
    point_y = []

    rng = np.random.default_rng(12345)

    for i, n_steps in enumerate(steps, start=1):
        for calculator in ["mace", "uma"]:
            rowset = attack_records[
                (attack_records["n_steps"] == n_steps)
                & (attack_records["calculator"] == calculator)
            ]

            box_values = []
            for _, row in rowset.iterrows():
                row_values, reason = value_getter(row)
                if row_values is None:
                    missing_rows.append({
                        "attack": attack,
                        "calculator": calculator,
                        "epsilon": epsilon,
                        "n_steps": n_steps,
                        "run_id": row["run_id"],
                        "reason": reason,
                    })
                else:
                    box_values.extend(row_values.tolist())

            if box_values:
                position = i + MODEL_OFFSETS[calculator]
                positions.append(position)
                values.append(box_values)
                colors.append(CALCULATOR_COLORS[calculator])

                jitter = rng.normal(loc=0.0, scale=0.010, size=len(box_values))
                point_x.extend((position + jitter).tolist())
                point_y.extend(box_values)

    return steps, positions, values, colors, point_x, point_y


def draw_grouped_boxplot_by_steps(ax, records, attack, epsilon, value_getter, ylabel, missing_rows):
    steps, positions, values, colors, point_x, point_y = collect_box_data_by_steps(
        records,
        attack,
        epsilon,
        value_getter,
        missing_rows,
    )

    if not values:
        ax.text(0.5, 0.5, "No plottable data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    if point_x and point_y:
        ax.scatter(
            point_x,
            point_y,
            s=6,
            color="#222222",
            alpha=0.14,
            linewidths=0,
            zorder=1,
        )

    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        zorder=2,
        medianprops={"color": "#111111", "linewidth": 1.5},
        capprops={"color": "#444444", "linewidth": 0.9},
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    ax.set_xticks(list(range(1, len(steps) + 1)))
    ax.set_xticklabels([str(int(step)) for step in steps])
    ax.tick_params(axis="x", labelrotation=35, pad=2)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    ax.set_xlabel("n_steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(x=0.03)

    return True


def tukey_whisker_span(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    if len(values) == 1:
        return 0.0

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    inlier_values = values[(values >= lower_fence) & (values <= upper_fence)]
    if len(inlier_values) == 0:
        return None

    lower_whisker = float(np.min(inlier_values))
    upper_whisker = float(np.max(inlier_values))

    return upper_whisker - lower_whisker


def collect_whisker_span_data(records, attack, value_getter, missing_rows):
    attack_records = records[records["attack_label"] == attack].copy()
    epsilons = sorted(attack_records["epsilon"].dropna().unique())

    points = []

    for i, epsilon in enumerate(epsilons, start=1):
        for calculator in ["mace", "uma"]:
            rowset = attack_records[
                (attack_records["epsilon"] == epsilon)
                & (attack_records["calculator"] == calculator)
            ]

            values = []
            for _, row in rowset.iterrows():
                row_values, reason = value_getter(row)
                if row_values is None:
                    missing_rows.append({
                        "attack": attack,
                        "calculator": calculator,
                        "epsilon": epsilon,
                        "run_id": row["run_id"],
                        "reason": reason,
                    })
                else:
                    values.extend(row_values.tolist())

            span = tukey_whisker_span(values)
            if span is not None:
                points.append({
                    "x": i,
                    "y": span,
                    "calculator": calculator,
                })

    return epsilons, points


def draw_whisker_span(ax, records, attack, value_getter, ylabel, missing_rows):
    epsilons, points = collect_whisker_span_data(
        records,
        attack,
        value_getter,
        missing_rows,
    )

    if not points:
        ax.text(0.5, 0.5, "No plottable data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    for calculator, color in CALCULATOR_COLORS.items():
        calc_points = [point for point in points if point["calculator"] == calculator]
        if not calc_points:
            continue

        calc_points = sorted(calc_points, key=lambda point: point["x"])
        x_values = [point["x"] for point in calc_points]
        y_values = [point["y"] for point in calc_points]

        ax.plot(
            x_values,
            y_values,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=calculator.upper(),
            zorder=3,
        )

    tick_positions = list(range(1, len(epsilons) + 1))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([format_epsilon_label(epsilon) for epsilon in epsilons])
    style_epsilon_tick_labels(ax, rotate=len(epsilons) >= 6)
    ax.set_xlabel(r"$\epsilon$ ($\AA$)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(x=0.05, y=0.12)

    return True


def make_whisker_span_figure(records, output_dir, figure_name, ylabel, rows):
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2), sharex=False, sharey=False)

    all_missing = []
    panel_index = 0

    for row_index, (row_title, getter_factory) in enumerate(rows):
        for col_index, attack in enumerate(ATTACK_ORDER):
            ax = axes[row_index, col_index]
            attack_missing = []

            draw_whisker_span(
                ax=ax,
                records=records,
                attack=attack,
                value_getter=getter_factory(),
                ylabel=ylabel,
                missing_rows=attack_missing,
            )

            for missing in attack_missing:
                missing["figure"] = figure_name
                missing["panel"] = f"{row_title} / {attack}"
            all_missing.extend(attack_missing)

            if row_index == 0:
                ax.set_title(attack)

            if col_index == 0:
                ax.text(
                    -0.33,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

            add_panel_label(ax, chr(ord("A") + panel_index))
            panel_index += 1

    apply_shared_figure_header(
        fig,
        subtitle="Each dot = upper whisker - lower whisker",
        left=0.03,
    )
    save_figure(fig, output_dir / figure_name)
    plt.close(fig)

    return all_missing


def collect_whisker_span_data_by_steps(records, attack, epsilon, value_getter, missing_rows):
    attack_records = records[
        (records["attack_label"] == attack)
        & (records["epsilon"] == float(epsilon))
    ].copy()
    steps = sorted(attack_records["n_steps"].dropna().unique())

    points = []

    for i, n_steps in enumerate(steps, start=1):
        for calculator in ["mace", "uma"]:
            rowset = attack_records[
                (attack_records["n_steps"] == n_steps)
                & (attack_records["calculator"] == calculator)
            ]

            values = []
            for _, row in rowset.iterrows():
                row_values, reason = value_getter(row)
                if row_values is None:
                    missing_rows.append({
                        "attack": attack,
                        "calculator": calculator,
                        "epsilon": epsilon,
                        "n_steps": n_steps,
                        "run_id": row["run_id"],
                        "reason": reason,
                    })
                else:
                    values.extend(row_values.tolist())

            span = tukey_whisker_span(values)
            if span is not None:
                points.append({
                    "x": i,
                    "y": span,
                    "calculator": calculator,
                })

    return steps, points


def draw_whisker_span_by_steps(ax, records, attack, epsilon, value_getter, ylabel, missing_rows):
    steps, points = collect_whisker_span_data_by_steps(
        records,
        attack,
        epsilon,
        value_getter,
        missing_rows,
    )

    if not points:
        ax.text(0.5, 0.5, "No plottable data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    for calculator, color in CALCULATOR_COLORS.items():
        calc_points = [point for point in points if point["calculator"] == calculator]
        if not calc_points:
            continue

        calc_points = sorted(calc_points, key=lambda point: point["x"])
        x_values = [point["x"] for point in calc_points]
        y_values = [point["y"] for point in calc_points]

        ax.plot(
            x_values,
            y_values,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=calculator.upper(),
            zorder=3,
        )

    tick_positions = list(range(1, len(steps) + 1))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(step)) for step in steps])
    ax.tick_params(axis="x", labelrotation=35, pad=2)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    ax.set_xlabel("n_steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(x=0.05, y=0.12)

    return True


def make_whisker_span_by_steps_figure(records, output_dir, figure_name, ylabel, rows, epsilon=0.1):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=False, sharey=False)

    all_missing = []
    panel_index = 0

    for row_index, (row_title, getter_factory) in enumerate(rows):
        for col_index, attack in enumerate(STEP_ATTACK_ORDER):
            ax = axes[row_index, col_index]
            attack_missing = []

            draw_whisker_span_by_steps(
                ax=ax,
                records=records,
                attack=attack,
                epsilon=epsilon,
                value_getter=getter_factory(),
                ylabel=ylabel,
                missing_rows=attack_missing,
            )

            for missing in attack_missing:
                missing["figure"] = figure_name
                missing["panel"] = f"{row_title} / {attack}"
            all_missing.extend(attack_missing)

            if row_index == 0:
                ax.set_title(attack)

            if col_index == 0:
                ax.text(
                    -0.33,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

            add_panel_label(ax, chr(ord("A") + panel_index))
            panel_index += 1

    apply_shared_figure_header(
        fig,
        subtitle=rf"Fixed $\epsilon$ = {epsilon:g} $\AA$; each dot = upper whisker - lower whisker",
        left=0.05,
    )
    save_figure(fig, output_dir / figure_name)
    plt.close(fig)

    return all_missing


def plot_convergence_panel_by_steps(ax, records, attack, epsilon, step_col, conv_col):
    attack_records = records[
        (records["attack_label"] == attack)
        & (records["epsilon"] == float(epsilon))
    ].copy()

    if attack_records.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    steps = sorted(attack_records["n_steps"].dropna().unique())
    step_positions = {step: index + 1 for index, step in enumerate(steps)}

    for calculator, color in CALCULATOR_COLORS.items():
        data = attack_records[
            (attack_records["calculator"] == calculator)
            & attack_records[step_col].notna()
        ].sort_values("n_steps")

        if data.empty:
            continue

        grouped = data.groupby("n_steps", as_index=False)[step_col].mean()
        grouped["step_position"] = grouped["n_steps"].map(step_positions)

        ax.plot(
            grouped["step_position"],
            grouped[step_col],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=calculator.upper(),
        )

        not_converged = data[data[conv_col] == False].copy()
        if not not_converged.empty:
            not_converged["step_position"] = not_converged["n_steps"].map(step_positions)

            ax.scatter(
                not_converged["step_position"],
                not_converged[step_col],
                s=45,
                facecolors="none",
                edgecolors=color,
                linewidths=1.4,
                zorder=3,
            )

    tick_positions = list(range(1, len(steps) + 1))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(step)) for step in steps])
    ax.tick_params(axis="x", labelrotation=35, pad=2)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    ax.set_xlabel("n_steps")
    ax.set_ylabel("Relaxation steps")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(x=0.03)

    return True


def make_convergence_by_steps_figure(records, output_dir, epsilon=0.1):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=False, sharey=False)

    rows = [
        ("before_relax_steps", "before_relax_converged", "Relaxation before attack"),
        ("after_relax_steps", "after_relax_converged", "Relaxation after attack"),
    ]

    panel_index = 0
    for row_index, (step_col, conv_col, row_title) in enumerate(rows):
        for col_index, attack in enumerate(STEP_ATTACK_ORDER):
            ax = axes[row_index, col_index]
            plot_convergence_panel_by_steps(ax, records, attack, epsilon, step_col, conv_col)

            if row_index == 0:
                ax.set_title(attack)

            if col_index == 0:
                ax.text(
                    -0.33,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

            add_panel_label(ax, chr(ord("A") + panel_index))
            panel_index += 1

    apply_shared_figure_header(
        fig,
        subtitle=rf"Fixed $\epsilon$ = {epsilon:g} $\AA$",
        left=0.05,
    )
    save_figure(fig, output_dir / "figure_4_convergence_by_n_steps")
    plt.close(fig)


def draw_grouped_ci_by_steps(ax, records, attack, epsilon, value_getter, ylabel, missing_rows):
    steps, positions, values, colors, point_x, point_y = collect_box_data_by_steps(
        records,
        attack,
        epsilon,
        value_getter,
        missing_rows,
    )

    if not values:
        ax.text(0.5, 0.5, "No plottable data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    series = {
        "mace": {"x": [], "median": [], "lower": [], "upper": []},
        "uma": {"x": [], "median": [], "lower": [], "upper": []},
    }

    for position, box_values in zip(positions, values):
        ci = bootstrap_median_ci(box_values)
        if ci is None:
            continue

        median, lower, upper = ci
        center_position = round(position)
        offset = position - center_position
        calculator = min(
            MODEL_OFFSETS,
            key=lambda name: abs(offset - MODEL_OFFSETS[name]),
        )

        x_value = round(position - MODEL_OFFSETS[calculator])
        series[calculator]["x"].append(x_value)
        series[calculator]["median"].append(median)
        series[calculator]["lower"].append(lower)
        series[calculator]["upper"].append(upper)

    for calculator, data in series.items():
        if not data["x"]:
            continue

        x = np.asarray(data["x"], dtype=float)
        median = np.asarray(data["median"], dtype=float)
        lower = np.asarray(data["lower"], dtype=float)
        upper = np.asarray(data["upper"], dtype=float)
        color = CALCULATOR_COLORS[calculator]

        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

        ax.plot(
            x,
            median,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=calculator.upper(),
        )

    tick_positions = list(range(1, len(steps) + 1))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(step)) for step in steps])
    ax.tick_params(axis="x", labelrotation=35, pad=2)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    ax.set_xlabel("n_steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(x=0.03)

    return True


def make_ci_by_steps_figure(records, output_dir, figure_name, ylabel, rows, epsilon=0.1):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=False, sharey=False)

    all_missing = []
    panel_index = 0

    for row_index, (row_title, getter_factory) in enumerate(rows):
        for col_index, attack in enumerate(STEP_ATTACK_ORDER):
            ax = axes[row_index, col_index]
            attack_missing = []

            draw_grouped_ci_by_steps(
                ax=ax,
                records=records,
                attack=attack,
                epsilon=epsilon,
                value_getter=getter_factory(),
                ylabel=ylabel,
                missing_rows=attack_missing,
            )

            for missing in attack_missing:
                missing["figure"] = figure_name
                missing["panel"] = f"{row_title} / {attack}"
            all_missing.extend(attack_missing)

            if row_index == 0:
                ax.set_title(attack)

            if col_index == 0:
                ax.text(
                    -0.33,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

            add_panel_label(ax, chr(ord("A") + panel_index))
            panel_index += 1

    apply_shared_figure_header(
        fig,
        subtitle=rf"Fixed $\epsilon$ = {epsilon:g} $\AA$; line = median, shaded band = 95% CI",
        left=0.05,
    )
    save_figure(fig, output_dir / figure_name)
    plt.close(fig)

    return all_missing


def make_distribution_by_steps_figure(records, output_dir, figure_name, ylabel, rows, epsilon=0.1):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=False, sharey=False)

    all_missing = []
    panel_index = 0

    for row_index, (row_title, getter_factory) in enumerate(rows):
        for col_index, attack in enumerate(STEP_ATTACK_ORDER):
            ax = axes[row_index, col_index]
            attack_missing = []

            draw_grouped_boxplot_by_steps(
                ax=ax,
                records=records,
                attack=attack,
                epsilon=epsilon,
                value_getter=getter_factory(),
                ylabel=ylabel,
                missing_rows=attack_missing,
            )
            all_missing.extend(attack_missing)

            if row_index == 0:
                ax.set_title(attack)

            if col_index == 0:
                ax.text(
                    -0.33,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

            add_panel_label(ax, chr(ord("A") + panel_index))
            panel_index += 1

    apply_shared_figure_header(
        fig,
        subtitle=rf"Fixed $\epsilon$ = {epsilon:g} $\AA$",
        left=0.05,
    )
    save_figure(fig, output_dir / figure_name)
    plt.close(fig)

    return all_missing


TOPOLOGY_METRICS = [
    "neighbor_jaccard_distance",
    "coordination_change_mean",
    "coordination_change_max",
    "rdf_l1_distance",
]


def topology_ready(records):
    required = ["neighbor_jaccard_distance", "coordination_change_max", "rdf_l1_distance"]
    return all(column in records.columns for column in required)


def save_topology_summary(records, output_dir):
    rows = []

    for (calculator, attack_label), group in records.groupby(["calculator", "attack_label"]):
        clean = group.replace([np.inf, -np.inf], np.nan)

        rows.append({
            "calculator": calculator,
            "attack_label": attack_label,
            "n_runs": int(len(clean)),
            "mean_neighbor_jaccard_distance": float(clean["neighbor_jaccard_distance"].mean()),
            "median_neighbor_jaccard_distance": float(clean["neighbor_jaccard_distance"].median()),
            "max_neighbor_jaccard_distance": float(clean["neighbor_jaccard_distance"].max()),
            "mean_coordination_change_max": float(clean["coordination_change_max"].mean()),
            "max_coordination_change_max": float(clean["coordination_change_max"].max()),
            "mean_rdf_l1_distance": float(clean["rdf_l1_distance"].mean()),
            "max_rdf_l1_distance": float(clean["rdf_l1_distance"].max()),
        })

    pd.DataFrame(rows).to_csv(output_dir / "topology_summary.csv", index=False)


def topology_scatter(ax, data, x_col, y_col, xlabel, ylabel, title):
    plotted = False

    for calculator, color in CALCULATOR_COLORS.items():
        subset = data[data["calculator"] == calculator].copy()
        subset = subset[[x_col, y_col, "attack_label"]].replace([np.inf, -np.inf], np.nan).dropna()
        if subset.empty:
            continue

        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=18,
            alpha=0.28,
            color=color,
            edgecolor="none",
            label=f"{calculator.upper()} runs",
        )
        plotted = True

        if len(subset) >= 6 and subset[x_col].nunique() >= 3:
            ordered = subset.sort_values(x_col)
            bins = min(8, max(3, int(np.sqrt(len(ordered)))))
            ordered["_bin"] = pd.qcut(ordered[x_col], q=bins, duplicates="drop")
            trend = ordered.groupby("_bin", observed=True).agg(
                x=(x_col, "median"),
                y=(y_col, "median"),
            ).dropna()

            if len(trend) >= 2:
                ax.plot(
                    trend["x"],
                    trend["y"],
                    color=color,
                    linewidth=2.0,
                    marker="o",
                    markersize=4,
                    label=f"{calculator.upper()} median trend",
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.28)

    y_values = _artist_values_for_axis(ax, "y")
    if len(y_values) and np.nanmax(np.abs(y_values)) <= 1e-12:
        ax.set_ylim(-0.0005, 0.0005)
        ax.text(
            0.5,
            0.88,
            "No measurable topology change in this subset",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#555555",
        )

    if not plotted:
        ax.text(0.5, 0.5, "No topology data", transform=ax.transAxes, ha="center", va="center")


def make_topology_vs_displacement(records, output_dir):
    data = records[
        records["neighbor_jaccard_distance"].notna()
        & records["mean_displacement"].notna()
    ].copy()

    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    topology_scatter(
        ax=ax,
        data=data,
        x_col="mean_displacement",
        y_col="neighbor_jaccard_distance",
        xlabel=r"Mean displacement ($\AA$)",
        ylabel="Neighbor-graph Jaccard distance",
        title="Topology change vs displacement",
    )
    ax.legend(frameon=False, ncol=2, fontsize=7)
    tighten_axes_for_publication(fig)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_topology_vs_displacement.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_topology_by_attack_type(records, output_dir):
    data = records[
        records["attack_label"].notna()
        & records["neighbor_jaccard_distance"].notna()
        & records["coordination_change_max"].notna()
        & records["rdf_l1_distance"].notna()
    ].copy()

    if data.empty:
        return

    data["has_topology_change"] = (
        (data["neighbor_jaccard_distance"].abs() > 1e-12)
        | (data["coordination_change_max"].abs() > 1e-12)
        | (data["rdf_l1_distance"].abs() > 1e-12)
    )

    attacks = [attack for attack in ATTACK_ORDER if attack in set(data["attack_label"])]
    positions = np.arange(len(attacks))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.34

    for offset, calculator in [(-width / 2, "mace"), (width / 2, "uma")]:
        calc_data = data[data["calculator"] == calculator]
        rates = []

        for attack in attacks:
            group = calc_data[calc_data["attack_label"] == attack]
            rates.append(np.nan if group.empty else 100.0 * float(group["has_topology_change"].mean()))

        ax.bar(
            positions + offset,
            rates,
            width=width,
            color=CALCULATOR_COLORS[calculator],
            alpha=0.78,
            label=calculator.upper(),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(attacks)
    ax.set_xlabel("Attack type")
    ax.set_ylabel("Runs with topology change (%)")
    ax.set_title("Topology-change rate by attack type")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=False, ncol=2)

    if data["has_topology_change"].sum() == 0:
        ax.set_ylim(0, 5)
        ax.text(
            0.5,
            0.82,
            "No neighbor, coordination, or RDF topology changes detected",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#555555",
        )
    else:
        max_rate = data.groupby(["calculator", "attack_label"])["has_topology_change"].mean().max() * 100.0
        ax.set_ylim(0, min(100.0, max(5.0, max_rate * 1.25)))

    fig.tight_layout()
    fig.savefig(output_dir / "figure_topology_by_attack_type.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_rdf_vs_coordination_change(records, output_dir):
    data = records[
        records["rdf_l1_distance"].notna()
        & records["coordination_change_max"].notna()
    ].copy()

    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    topology_scatter(
        ax=ax,
        data=data,
        x_col="rdf_l1_distance",
        y_col="coordination_change_max",
        xlabel="RDF L1 distance",
        ylabel="Max coordination change",
        title="RDF change vs coordination change",
    )
    ax.legend(frameon=False, ncol=2, fontsize=7)
    tighten_axes_for_publication(fig)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_rdf_vs_coordination_change.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_topology_figures(records, output_dir):
    if records.empty or not topology_ready(records):
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = records.copy()
    for column in TOPOLOGY_METRICS + ["mean_displacement"]:
        if column in clean.columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")

    save_topology_summary(clean, output_dir)
    make_topology_vs_displacement(clean, output_dir)
    make_topology_by_attack_type(clean, output_dir)
    make_rdf_vs_coordination_change(clean, output_dir)

    for material_slug, material_records in clean.groupby("material_slug"):
        material_output_dir = output_dir / str(material_slug)
        material_output_dir.mkdir(parents=True, exist_ok=True)
        save_topology_summary(material_records, material_output_dir)
        make_topology_vs_displacement(material_records, material_output_dir)
        make_topology_by_attack_type(material_records, material_output_dir)
        make_rdf_vs_coordination_change(material_records, material_output_dir)


def main():
    apply_plot_style()

    parser = argparse.ArgumentParser(
        description="Create publication-quality comprehensive MACE vs UMA plots."
    )
    parser.add_argument("--mace-dir", default=BASE_DIR / "outputs_mace", type=Path)
    parser.add_argument("--uma-dir", default=BASE_DIR / "outputs_uma", type=Path)
    parser.add_argument("--output-dir", default=BASE_DIR / "outputs_comprehensive", type=Path)
    parser.add_argument("--materials", default=BASE_DIR / "tests_materials.csv", type=Path)
    parser.add_argument("--structures-dir", default=BASE_DIR / "mp_structures", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    make_structure_summary(args.materials, args.structures_dir, args.output_dir)

    mace_records, mace_missing = load_summary(
        args.mace_dir / "summary.csv",
        args.mace_dir,
        "mace",
    )
    uma_records, uma_missing = load_summary(
        args.uma_dir / "summary.csv",
        args.uma_dir,
        "uma",
    )

    records = pd.DataFrame(mace_records + uma_records)
    missing_rows = [{"reason": item} for item in mace_missing + uma_missing]

    if records.empty:
        pd.DataFrame(missing_rows).to_csv(
            args.output_dir / "missing_data_report.csv",
            index=False,
        )
        raise SystemExit("No successful runs found in outputs_mace or outputs_uma.")

    records.to_csv(args.output_dir / "combined_dataset.csv", index=False)

    make_topology_figures(records, args.output_dir / "topology")

    epsilon_records = records[
        ~records["run_id"].str.contains("_steps", regex=False)
    ].copy()

    n_step_records = records[
        records["run_id"].str.contains("_steps", regex=False)
    ].copy()

    make_convergence_figure(epsilon_records, args.output_dir)

    force_missing = make_distribution_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_2_delta_force_by_epsilon",
        ylabel=r"$\Delta$ force (eV/$\AA$)",
        rows=[
            (
                "After attack, before relaxation",
                lambda: (lambda row: force_delta_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "perturbed_forces.csv",
                )),
            ),
            (
                "After attack, after relaxation",
                lambda: (lambda row: force_delta_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "after_forces.csv",
                )),
            ),
        ],
    )

    force_by_epsilon_missing = make_ci_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_2_delta_force_ci_by_epsilon",
        ylabel=r"Median $\Delta$ force with 95% CI (eV/$\AA$)",
        rows=[
            (
                "After attack, before relaxation",
                lambda: (lambda row: force_delta_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "perturbed_forces.csv",
                )),
            ),
            (
                "After attack, after relaxation",
                lambda: (lambda row: force_delta_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "after_forces.csv",
                )),
            ),
        ],
    )

    displacement_missing = make_distribution_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_3_displacement_by_epsilon",
        ylabel=r"Displacement ($\AA$)",
        rows=[
            (
                "After attack, before relaxation",
                lambda: (lambda row: displacement_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "perturbed_forces.csv",
                )),
            ),
            (
                "After attack, after relaxation",
                lambda: (lambda row: displacement_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "after_forces.csv",
                )),
            ),
        ],
    )

    displacement_by_epsilon_missing = make_ci_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_3_displacement_ci_by_epsilon",
        ylabel=r"Median displacement with 95% CI ($\AA$)",
        rows=[
            (
                "After attack, before relaxation",
                lambda: (lambda row: displacement_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "perturbed_forces.csv",
                )),
            ),
            (
                "After attack, after relaxation",
                lambda: (lambda row: displacement_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "after_forces.csv",
                )),
            ),
        ],
    )

    parametric_by_epsilon_missing = make_parametric_figure_set(
        records=epsilon_records,
        output_dir=args.output_dir,
        suffix="epsilon",
        attacks_to_plot=ATTACK_ORDER,
        bubble_label="epsilon",
    )

    parametric_by_steps_missing = make_parametric_figure_set(
        records=n_step_records,
        output_dir=args.output_dir,
        suffix="n_steps",
        attacks_to_plot=STEP_ATTACK_ORDER,
        bubble_label="n_steps",
    )

    make_convergence_by_steps_figure(n_step_records, args.output_dir, epsilon=0.1)

    force_by_steps_missing = make_distribution_by_steps_figure(
        records=n_step_records,
        output_dir=args.output_dir,
        figure_name="figure_5_delta_force_by_n_steps",
        ylabel=r"$\Delta$ force (eV/$\AA$)",
        epsilon=0.1,
        rows=[
            (
                "After attack, before relaxation",
                lambda: (lambda row: force_delta_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "perturbed_forces.csv",
                )),
            ),
            (
                "After attack, after relaxation",
                lambda: (lambda row: force_delta_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "after_forces.csv",
                )),
            ),
        ],
    )

    force_by_steps_ci_missing = make_ci_by_steps_figure(
        records=n_step_records,
        output_dir=args.output_dir,
        figure_name="figure_5_delta_force_ci_by_n_steps",
        ylabel=r"Median $\Delta$ force with 95% CI (eV/$\AA$)",
        epsilon=0.1,
        rows=[
            (
                "After attack, before relaxation",
                lambda: (lambda row: force_delta_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "perturbed_forces.csv",
                )),
            ),
            (
                "After attack, after relaxation",
                lambda: (lambda row: force_delta_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "after_forces.csv",
                )),
            ),
        ],
    )

    displacement_by_steps_missing = make_distribution_by_steps_figure(
        records=n_step_records,
        output_dir=args.output_dir,
        figure_name="figure_6_displacement_by_n_steps",
        ylabel=r"Displacement ($\AA$)",
        epsilon=0.1,
        rows=[
            (
                "After attack, before relaxation",
                lambda: (lambda row: displacement_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "perturbed_forces.csv",
                )),
            ),
            (
                "After attack, after relaxation",
                lambda: (lambda row: displacement_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "after_forces.csv",
                )),
            ),
        ],
    )

    displacement_by_steps_ci_missing = make_ci_by_steps_figure(
        records=n_step_records,
        output_dir=args.output_dir,
        figure_name="figure_6_displacement_ci_by_n_steps",
        ylabel=r"Median displacement with 95% CI ($\AA$)",
        epsilon=0.1,
        rows=[
            (
                "After attack, before relaxation",
                lambda: (lambda row: displacement_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "perturbed_forces.csv",
                )),
            ),
            (
                "After attack, after relaxation",
                lambda: (lambda row: displacement_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "after_forces.csv",
                )),
            ),
        ],
    )

    force_whisker_span_missing = make_whisker_span_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_2_delta_force_whisker_span_by_epsilon",
        ylabel=r"$\Delta$ force whisker span (eV/$\AA$)",
        rows=[
            (
                "After attack, before relaxation",
                lambda: (lambda row: force_delta_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "perturbed_forces.csv",
                )),
            ),
            (
                "After attack, after relaxation",
                lambda: (lambda row: force_delta_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "after_forces.csv",
                )),
            ),
        ],
    )

    displacement_whisker_span_missing = make_whisker_span_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_3_displacement_whisker_span_by_epsilon",
        ylabel=r"Displacement whisker span ($\AA$)",
        rows=[
            (
                "After attack, before relaxation",
                lambda: (lambda row: displacement_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "perturbed_forces.csv",
                )),
            ),
            (
                "After attack, after relaxation",
                lambda: (lambda row: displacement_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "after_forces.csv",
                )),
            ),
        ],
    )

    force_by_steps_whisker_span_missing = make_whisker_span_by_steps_figure(
        records=n_step_records,
        output_dir=args.output_dir,
        figure_name="figure_5_delta_force_whisker_span_by_n_steps",
        ylabel=r"$\Delta$ force whisker span (eV/$\AA$)",
        epsilon=0.1,
        rows=[
            (
                "After attack, before relaxation",
                lambda: (lambda row: force_delta_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "perturbed_forces.csv",
                )),
            ),
            (
                "After attack, after relaxation",
                lambda: (lambda row: force_delta_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "after_forces.csv",
                )),
            ),
        ],
    )

    displacement_by_steps_whisker_span_missing = make_whisker_span_by_steps_figure(
        records=n_step_records,
        output_dir=args.output_dir,
        figure_name="figure_6_displacement_whisker_span_by_n_steps",
        ylabel=r"Displacement whisker span ($\AA$)",
        epsilon=0.1,
        rows=[
            (
                "After attack, before relaxation",
                lambda: (lambda row: displacement_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "perturbed_forces.csv",
                )),
            ),
            (
                "After attack, after relaxation",
                lambda: (lambda row: displacement_values(
                    row["run_dir"],
                    "before_forces.csv",
                    "after_forces.csv",
                )),
            ),
        ],
    )

    for material_slug, material_records in records.groupby("material_slug"):
        material_output_dir = args.output_dir / str(material_slug)
        material_output_dir.mkdir(parents=True, exist_ok=True)

        material_epsilon_records = material_records[
            ~material_records["run_id"].str.contains("_steps", regex=False)
        ].copy()

        material_n_step_records = material_records[
            material_records["run_id"].str.contains("_steps", regex=False)
        ].copy()

        if not material_epsilon_records.empty:
            make_convergence_figure(material_epsilon_records, material_output_dir)

            make_distribution_figure(
                records=material_epsilon_records,
                output_dir=material_output_dir,
                figure_name="figure_2_delta_force_by_epsilon",
                ylabel=r"$\Delta$ force (eV/$\AA$)",
                rows=[
                    (
                        "After attack, before relaxation",
                        lambda: (lambda row: force_delta_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "perturbed_forces.csv",
                        )),
                    ),
                    (
                        "After attack, after relaxation",
                        lambda: (lambda row: force_delta_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "after_forces.csv",
                        )),
                    ),
                ],
            )

            make_ci_figure(
                records=material_epsilon_records,
                output_dir=material_output_dir,
                figure_name="figure_2_delta_force_ci_by_epsilon",
                ylabel=r"Median $\Delta$ force with 95% CI (eV/$\AA$)",
                rows=[
                    (
                        "After attack, before relaxation",
                        lambda: (lambda row: force_delta_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "perturbed_forces.csv",
                        )),
                    ),
                    (
                        "After attack, after relaxation",
                        lambda: (lambda row: force_delta_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "after_forces.csv",
                        )),
                    ),
                ],
            )

            make_distribution_figure(
                records=material_epsilon_records,
                output_dir=material_output_dir,
                figure_name="figure_3_displacement_by_epsilon",
                ylabel=r"Displacement ($\AA$)",
                rows=[
                    (
                        "After attack, before relaxation",
                        lambda: (lambda row: displacement_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "perturbed_forces.csv",
                        )),
                    ),
                    (
                        "After attack, after relaxation",
                        lambda: (lambda row: displacement_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "after_forces.csv",
                        )),
                    ),
                ],
            )

            make_ci_figure(
                records=material_epsilon_records,
                output_dir=material_output_dir,
                figure_name="figure_3_displacement_ci_by_epsilon",
                ylabel=r"Median displacement with 95% CI ($\AA$)",
                rows=[
                    (
                        "After attack, before relaxation",
                        lambda: (lambda row: displacement_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "perturbed_forces.csv",
                        )),
                    ),
                    (
                        "After attack, after relaxation",
                        lambda: (lambda row: displacement_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "after_forces.csv",
                        )),
                    ),
                ],
            )

            make_whisker_span_figure(
                records=material_epsilon_records,
                output_dir=material_output_dir,
                figure_name="figure_2_delta_force_whisker_span_by_epsilon",
                ylabel=r"$\Delta$ force whisker span (eV/$\AA$)",
                rows=[
                    (
                        "After attack, before relaxation",
                        lambda: (lambda row: force_delta_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "perturbed_forces.csv",
                        )),
                    ),
                    (
                        "After attack, after relaxation",
                        lambda: (lambda row: force_delta_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "after_forces.csv",
                        )),
                    ),
                ],
            )

            make_whisker_span_figure(
                records=material_epsilon_records,
                output_dir=material_output_dir,
                figure_name="figure_3_displacement_whisker_span_by_epsilon",
                ylabel=r"Displacement whisker span ($\AA$)",
                rows=[
                    (
                        "After attack, before relaxation",
                        lambda: (lambda row: displacement_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "perturbed_forces.csv",
                        )),
                    ),
                    (
                        "After attack, after relaxation",
                        lambda: (lambda row: displacement_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "after_forces.csv",
                        )),
                    ),
                ],
            )

            make_parametric_figure_set(
                records=material_epsilon_records,
                output_dir=material_output_dir,
                suffix="epsilon",
                attacks_to_plot=ATTACK_ORDER,
                bubble_label="epsilon",
            )

        if not material_n_step_records.empty:
            make_convergence_by_steps_figure(
                material_n_step_records,
                material_output_dir,
                epsilon=0.1,
            )

            make_distribution_by_steps_figure(
                records=material_n_step_records,
                output_dir=material_output_dir,
                figure_name="figure_5_delta_force_by_n_steps",
                ylabel=r"$\Delta$ force (eV/$\AA$)",
                epsilon=0.1,
                rows=[
                    (
                        "After attack, before relaxation",
                        lambda: (lambda row: force_delta_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "perturbed_forces.csv",
                        )),
                    ),
                    (
                        "After attack, after relaxation",
                        lambda: (lambda row: force_delta_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "after_forces.csv",
                        )),
                    ),
                ],
            )

            make_ci_by_steps_figure(
                records=material_n_step_records,
                output_dir=material_output_dir,
                figure_name="figure_5_delta_force_ci_by_n_steps",
                ylabel=r"Median $\Delta$ force with 95% CI (eV/$\AA$)",
                epsilon=0.1,
                rows=[
                    (
                        "After attack, before relaxation",
                        lambda: (lambda row: force_delta_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "perturbed_forces.csv",
                        )),
                    ),
                    (
                        "After attack, after relaxation",
                        lambda: (lambda row: force_delta_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "after_forces.csv",
                        )),
                    ),
                ],
            )

            make_distribution_by_steps_figure(
                records=material_n_step_records,
                output_dir=material_output_dir,
                figure_name="figure_6_displacement_by_n_steps",
                ylabel=r"Displacement ($\AA$)",
                epsilon=0.1,
                rows=[
                    (
                        "After attack, before relaxation",
                        lambda: (lambda row: displacement_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "perturbed_forces.csv",
                        )),
                    ),
                    (
                        "After attack, after relaxation",
                        lambda: (lambda row: displacement_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "after_forces.csv",
                        )),
                    ),
                ],
            )

            make_ci_by_steps_figure(
                records=material_n_step_records,
                output_dir=material_output_dir,
                figure_name="figure_6_displacement_ci_by_n_steps",
                ylabel=r"Median displacement with 95% CI ($\AA$)",
                epsilon=0.1,
                rows=[
                    (
                        "After attack, before relaxation",
                        lambda: (lambda row: displacement_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "perturbed_forces.csv",
                        )),
                    ),
                    (
                        "After attack, after relaxation",
                        lambda: (lambda row: displacement_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "after_forces.csv",
                        )),
                    ),
                ],
            )

            make_whisker_span_by_steps_figure(
                records=material_n_step_records,
                output_dir=material_output_dir,
                figure_name="figure_5_delta_force_whisker_span_by_n_steps",
                ylabel=r"$\Delta$ force whisker span (eV/$\AA$)",
                epsilon=0.1,
                rows=[
                    (
                        "After attack, before relaxation",
                        lambda: (lambda row: force_delta_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "perturbed_forces.csv",
                        )),
                    ),
                    (
                        "After attack, after relaxation",
                        lambda: (lambda row: force_delta_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "after_forces.csv",
                        )),
                    ),
                ],
            )

            make_whisker_span_by_steps_figure(
                records=material_n_step_records,
                output_dir=material_output_dir,
                figure_name="figure_6_displacement_whisker_span_by_n_steps",
                ylabel=r"displacement whisker span ($\AA$)",
                epsilon=0.1,
                rows=[
                    (
                        "After attack, before relaxation",
                        lambda: (lambda row: displacement_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "perturbed_forces.csv",
                        )),
                    ),
                    (
                        "After attack, after relaxation",
                        lambda: (lambda row: displacement_values(
                            row["run_dir"],
                            "before_forces.csv",
                            "after_forces.csv",
                        )),
                    ),
                ],
            )

            make_parametric_figure_set(
                records=material_n_step_records,
                output_dir=material_output_dir,
                suffix="n_steps",
                attacks_to_plot=STEP_ATTACK_ORDER,
                bubble_label="n_steps",
            )

    missing_rows.extend(force_missing)
    missing_rows.extend(force_whisker_span_missing)
    missing_rows.extend(force_by_epsilon_missing)
    missing_rows.extend(displacement_missing)
    missing_rows.extend(displacement_whisker_span_missing)
    missing_rows.extend(displacement_by_epsilon_missing)
    missing_rows.extend(parametric_by_epsilon_missing)
    missing_rows.extend(force_by_steps_missing)
    missing_rows.extend(force_by_steps_whisker_span_missing)
    missing_rows.extend(force_by_steps_ci_missing)
    missing_rows.extend(displacement_by_steps_missing)
    missing_rows.extend(displacement_by_steps_whisker_span_missing)
    missing_rows.extend(displacement_by_steps_ci_missing)
    missing_rows.extend(parametric_by_steps_missing)

    pd.DataFrame(missing_rows).to_csv(
        args.output_dir / "missing_data_report.csv",
        index=False,
    )

    print(f"Saved comprehensive plots to {args.output_dir}")
    print(f"Saved combined dataset to {args.output_dir / 'combined_dataset.csv'}")
    print(f"Saved structure summary to {args.output_dir / 'materials_summary_combined.csv'}")
    print("Main publication figures:")
    print(f"  {args.output_dir / 'figure_1_convergence_by_epsilon.png'}")
    print(f"  {args.output_dir / 'figure_2_delta_force_by_epsilon.png'}")
    print(f"  {args.output_dir / 'figure_2_delta_force_whisker_span_by_epsilon.png'}")
    print(f"  {args.output_dir / 'figure_2_delta_force_ci_by_epsilon.png'}")
    print(f"  {args.output_dir / 'figure_3_displacement_by_epsilon.png'}")
    print(f"  {args.output_dir / 'figure_3_displacement_whisker_span_by_epsilon.png'}")
    print(f"  {args.output_dir / 'figure_3_displacement_ci_by_epsilon.png'}")
    print(f"  {args.output_dir / 'figure_4_convergence_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_5_delta_force_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_5_delta_force_whisker_span_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_5_delta_force_ci_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_6_displacement_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_6_displacement_whisker_span_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_6_displacement_ci_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_7_convergence_vs_displacement_by_epsilon.png'}")
    print(f"  {args.output_dir / 'figure_7_convergence_vs_displacement_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_8_convergence_vs_delta_force_by_epsilon.png'}")
    print(f"  {args.output_dir / 'figure_8_convergence_vs_delta_force_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_9_delta_force_vs_displacement_by_epsilon.png'}")
    print(f"  {args.output_dir / 'figure_9_delta_force_vs_displacement_by_n_steps.png'}")

if __name__ == "__main__":
    main()
