#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import math
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter, FixedLocator, NullLocator
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

EPSILON_POSITION_FACTORS = {
    "mace": 10 ** (-0.035),
    "uma": 10 ** (0.035),
}

EPSILON_BOX_WIDTH_LOG10 = 0.020

EPSILON_PERCENT_SUFFIX = "_percent_displacement"
EPSILON_AXIS_RAW = "epsilon"
EPSILON_AXIS_PERCENT = "percent_displacement"


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


def format_power_tick(value):
    value = float(value)
    if value <= 0 or not np.isfinite(value):
        return ""

    power = int(round(np.log10(value)))
    decade = 10.0 ** power

    if not np.isclose(value, decade, rtol=1e-8, atol=0.0):
        return ""

    if power >= 0:
        return f"{decade:g}"

    decimals = abs(power)
    return f"{decade:.{decimals}f}"


def positive_finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def decade_ticks(values):
    values = positive_finite_values(values)
    if len(values) == 0:
        return []

    min_power = int(np.floor(np.log10(np.min(values))))
    max_power = int(np.ceil(np.log10(np.max(values))))

    return [10.0 ** power for power in range(min_power, max_power + 1)]


def apply_decade_ticks(axis, values):
    ticks = decade_ticks(values)
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda value, _: format_power_tick(value)))


def epsilon_plot_position(epsilon, calculator=None):
    epsilon = float(epsilon)
    if calculator is None:
        return epsilon
    return epsilon * EPSILON_POSITION_FACTORS[calculator]


def epsilon_box_widths(positions):
    widths = []
    for position in positions:
        lower = position / (10 ** EPSILON_BOX_WIDTH_LOG10)
        upper = position * (10 ** EPSILON_BOX_WIDTH_LOG10)
        widths.append(upper - lower)
    return widths


def apply_epsilon_axis(ax, x_values, plotted_positions=None, axis_mode=EPSILON_AXIS_RAW):
    ticks = decade_ticks(x_values)
    ax.set_xscale("log")

    limit_values = list(positive_finite_values(x_values))
    if plotted_positions is not None:
        limit_values.extend(positive_finite_values(plotted_positions).tolist())

    if ticks:
        apply_decade_ticks(ax.xaxis, x_values)
        finite_limits = positive_finite_values(limit_values)
        left = min(ticks[0], float(np.min(finite_limits))) / 1.18
        right = max(ticks[-1], float(np.max(finite_limits))) * 1.18
        ax.set_xlim(left, right)

    ax.tick_params(axis="x", labelrotation=0, pad=2)
    if axis_mode == EPSILON_AXIS_PERCENT:
        ax.set_xlabel("Epsilon (% min lattice)")
    else:
        ax.set_xlabel(r"$\epsilon$ ($\AA$)")


def percent_displacement_plot_x(value):
    if value is None or not np.isfinite(value) or value <= 0:
        return np.nan
    return 10 ** (round(np.log10(float(value)) * 4) / 4)


STEP_POSITION_FACTORS = {
    "mace": 10 ** (-0.035),
    "uma": 10 ** (0.035),
}

STEP_BOX_WIDTH_LOG10 = 0.020


def step_plot_position(n_steps, calculator=None):
    n_steps = float(n_steps)
    if calculator is None:
        return n_steps
    return n_steps * STEP_POSITION_FACTORS[calculator]


def step_box_widths(positions):
    widths = []
    for position in positions:
        lower = position / (10 ** STEP_BOX_WIDTH_LOG10)
        upper = position * (10 ** STEP_BOX_WIDTH_LOG10)
        widths.append(upper - lower)
    return widths


def apply_step_axis(ax, steps, plotted_positions=None):
    ticks = decade_ticks(steps)
    ax.set_xscale("log")

    limit_values = list(positive_finite_values(steps))
    if plotted_positions is not None:
        limit_values.extend(positive_finite_values(plotted_positions).tolist())

    if ticks:
        apply_decade_ticks(ax.xaxis, steps)

        finite_limits = positive_finite_values(limit_values)
        if len(finite_limits):
            left = min(ticks[0], float(np.min(finite_limits))) / 1.18
            right = max(ticks[-1], float(np.max(finite_limits))) * 1.18
        else:
            left = ticks[0] / 1.18
            right = ticks[-1] * 1.18

        ax.set_xlim(left, right)

    ax.tick_params(axis="x", labelrotation=0, pad=2)
    ax.set_xlabel("n_steps")


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


def epsilon_reference_length_from_summary_row(row):
    input_path = clean_value(row.get("input_path"))
    if input_path is None:
        return np.nan, "Missing input_path"

    path = Path(str(input_path))
    if not path.is_absolute():
        path = BASE_DIR / path

    try:
        atoms = read_structure(path)
        lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    except Exception as exc:
        return np.nan, f"Could not read structure with ASE: {exc}"

    lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
    if len(lengths) == 0:
        return np.nan, "No positive lattice lengths"

    return float(np.min(lengths)), None


def percent_displacement_from_epsilon(epsilon, reference_length_a):
    epsilon = as_float(epsilon)
    reference_length_a = as_float(reference_length_a)
    if epsilon is None or reference_length_a is None or reference_length_a <= 0:
        return np.nan
    return 100.0 * epsilon / reference_length_a


def percent_figure_name(figure_name):
    return f"{figure_name}{EPSILON_PERCENT_SUFFIX}"


def has_percent_displacement_axis(records):
    if "epsilon_percent_displacement" not in records.columns:
        return False
    values = positive_finite_values(records["epsilon_percent_displacement"].dropna())
    return len(values) > 0


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

        epsilon_value = as_float(row.get("epsilon"))
        input_path = clean_value(row.get("input_path"))
        epsilon_reference_length_a, epsilon_reference_reason = epsilon_reference_length_from_summary_row(row)
        epsilon_percent_displacement = percent_displacement_from_epsilon(
            epsilon_value,
            epsilon_reference_length_a,
        )

        records.append({
            "run_id": str(row["run_id"]),
            "material_label": material_label,
            "material_slug": material_slug,
            "logical_run_id": normalized_run_id(row["run_id"]),
            "calculator": calculator,
            "attack_label": attack_label(row),
            "epsilon": epsilon_value,
            "input_path": input_path,
            "epsilon_reference_length_a": epsilon_reference_length_a,
            "epsilon_reference_reason": epsilon_reference_reason,
            "epsilon_percent_displacement": epsilon_percent_displacement,
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


def limits_from_bubble_extents(extents, pad=0.12, nonnegative=True):
    if not extents:
        return None, None

    xmins = np.asarray([item[0] for item in extents], dtype=float)
    xmaxs = np.asarray([item[1] for item in extents], dtype=float)
    ymins = np.asarray([item[2] for item in extents], dtype=float)
    ymaxs = np.asarray([item[3] for item in extents], dtype=float)

    finite = (
        np.isfinite(xmins)
        & np.isfinite(xmaxs)
        & np.isfinite(ymins)
        & np.isfinite(ymaxs)
    )
    if not np.any(finite):
        return None, None

    xmin = float(np.min(xmins[finite]))
    xmax = float(np.max(xmaxs[finite]))
    ymin = float(np.min(ymins[finite]))
    ymax = float(np.max(ymaxs[finite]))

    xspan = xmax - xmin
    yspan = ymax - ymin

    if xspan <= 0 or not np.isfinite(xspan):
        xspan = max(abs(xmax) * 0.20, 1e-9)
        xmin -= 0.5 * xspan
        xmax += 0.5 * xspan

    if yspan <= 0 or not np.isfinite(yspan):
        yspan = max(abs(ymax) * 0.20, 1e-9)
        ymin -= 0.5 * yspan
        ymax += 0.5 * yspan

    xpad = max(pad * (xmax - xmin), 1e-9)
    ypad = max(pad * (ymax - ymin), 1e-9)

    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    if nonnegative:
        if xmin < 0 and np.min(xmins[finite]) >= 0:
            xmin = 0.0
        if ymin < 0 and np.min(ymins[finite]) >= 0:
            ymin = 0.0

    return (xmin, xmax), (ymin, ymax)


def style_numeric_axis(ax, xbins=5, ybins=5):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))

    formatter_x = ScalarFormatter(useMathText=True, useOffset=False)
    formatter_x.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(formatter_x)

    style_y_axis_no_offset(ax, ybins=ybins)

    ax.tick_params(axis="both", labelsize=8, pad=2)


def style_y_axis_no_offset(ax, ybins=5):
    y_values = clean_numeric_array(_artist_values_for_axis(ax, "y"))

    if len(y_values):
        ymin, ymax = ax.get_ylim()
        span = float(ymax - ymin)
        center = float(np.nanmedian(y_values))

        if np.isfinite(span) and np.isfinite(center) and abs(center) > 0:
            relative_span = abs(span / center)

            if relative_span < 1e-5:
                pad = max(abs(center) * 0.01, 1e-8)
                ax.set_ylim(center - pad, center + pad)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.5g}"))
                ax.tick_params(axis="y", labelsize=8, pad=2)
                return

    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins, prune=None))

    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)

    ax.tick_params(axis="y", labelsize=8, pad=2)


def style_relaxation_steps_axis(ax, log_scale=False, tight_linear=False):
    y_values = clean_numeric_array(_artist_values_for_axis(ax, "y"))

    if len(y_values) == 0:
        return

    if log_scale:
        positive = positive_finite_values(y_values)
        if len(positive) == 0:
            return

        ax.set_yscale("log")
        ticks = decade_ticks(positive)
        if ticks:
            apply_decade_ticks(ax.yaxis, positive)

        ax.set_ylim(float(np.min(positive)) / 1.35, float(np.max(positive)) * 1.35)
        ax.tick_params(axis="y", labelsize=8, pad=2)
        return

    finite = y_values[np.isfinite(y_values)]
    if len(finite) == 0:
        ax.set_ylim(-0.03, 1.0)
        ax.set_yticks([0, 1])
    elif tight_linear:
        y_min = float(np.nanmin(finite))
        y_max = float(np.nanmax(finite))
        span = y_max - y_min
        pad = max(span * 0.35, abs(y_max) * 0.035, 0.05)

        ax.set_ylim(y_min - pad, y_max + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False, prune=None))
    else:
        y_max = float(np.nanmax(finite))
        if not np.isfinite(y_max) or y_max <= 0:
            ax.set_ylim(-0.03, 1.0)
            ax.set_yticks([0, 1])
        else:
            ax.set_ylim(0, y_max * 1.12)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, prune=None))

    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis="y", labelsize=8, pad=2)


def apply_displacement_symlog_axis(ax, linthresh=0.05):
    xlabel = ax.get_xlabel().lower()
    if "displacement" not in xlabel:
        return

    ax.set_xscale("symlog", linthresh=linthresh)

    ticks = [0, 0.01, 0.1, 1, 10]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "0.01", "0.1", "1", "10"])
    ax.tick_params(axis="x", labelrotation=0, pad=2)


def apply_positive_log_axis(ax, axis_name):
    values = positive_finite_values(_artist_values_for_axis(ax, axis_name))
    if len(values) == 0:
        return

    lower = float(np.min(values)) / 1.35
    upper = float(np.max(values)) * 1.35

    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper <= 0:
        return

    if lower == upper:
        lower /= 1.35
        upper *= 1.35

    log_span = np.log10(upper) - np.log10(lower)

    if log_span < 1.0:
        if axis_name == "x":
            ax.set_xscale("linear")
            ax.set_xlim(lower, upper)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
            formatter = ScalarFormatter(useMathText=True, useOffset=False)
            formatter.set_powerlimits((-3, 3))
            ax.xaxis.set_major_formatter(formatter)
        else:
            ax.set_yscale("linear")
            ax.set_ylim(lower, upper)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
            formatter = ScalarFormatter(useMathText=True, useOffset=False)
            formatter.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(formatter)

        ax.tick_params(axis=axis_name, labelsize=8, pad=2)
        return

    ticks = decade_ticks(values)

    def format_log_tick(value, _):
        if value <= 0 or not np.isfinite(value):
            return ""

        if value >= 100:
            return f"{value:.0f}"
        if value >= 10:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        if value >= 1:
            return f"{value:.2f}".rstrip("0").rstrip(".")
        if value >= 0.01:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.1e}"

    if axis_name == "x":
        ax.set_xscale("log")
        ax.set_xlim(lower, upper)
        if ticks:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(format_log_tick))
    else:
        ax.set_yscale("log")
        ax.set_ylim(lower, upper)
        if ticks:
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(FuncFormatter(format_log_tick))

    ax.tick_params(axis=axis_name, labelsize=8, pad=2)


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

        if (
            getattr(ax, "_preserve_parametric_limits", False)
            or getattr(ax, "_preserve_manual_limits", False)
        ):
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
    ax._preserve_parametric_limits = True

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

    bubble_extents = []

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

            x_radius_for_extent = max(min_x_radius, 1e-9)
            y_radius_for_extent = max(min_y_radius, 1e-9)

            if len(group) >= 3:
                x_radius = max(variability_radius(x_values), min_x_radius)
                y_radius = max(variability_radius(y_values), min_y_radius)

                if x_span is not None and x_span > 0:
                    x_radius = min(x_radius, 0.055 * x_span)
                if y_span is not None and y_span > 0:
                    y_radius = min(y_radius, 0.055 * y_span)

                x_radius_for_extent = max(x_radius, x_radius_for_extent)
                y_radius_for_extent = max(y_radius, y_radius_for_extent)

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

            bubble_extents.append((
                x_center - x_radius_for_extent,
                x_center + x_radius_for_extent,
                y_center - y_radius_for_extent,
                y_center + y_radius_for_extent,
            ))

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

    fitted_x_limits, fitted_y_limits = limits_from_bubble_extents(
        bubble_extents,
        pad=0.12,
        nonnegative=True,
    )

    if fitted_x_limits is not None:
        ax.set_xlim(*fitted_x_limits)
    elif x_limits is not None:
        ax.set_xlim(*x_limits)

    if fitted_y_limits is not None:
        ax.set_ylim(*fitted_y_limits)
    elif y_limits is not None:
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
    x_log=False,
    y_log=False,
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
            if x_log:
                apply_positive_log_axis(ax, "x")
            if y_log:
                apply_positive_log_axis(ax, "y")

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
        x_log=True,
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
        x_log=True,
        y_log=True,
    )

    return (
        convergence_displacement_missing
        + convergence_force_missing
        + force_displacement_missing
    )


def collect_box_data(records, attack, value_getter, missing_rows, x_col="epsilon"):
    attack_records = records[records["attack_label"] == attack].copy()

    plot_x_col = x_col
    if x_col == "epsilon_percent_displacement":
        plot_x_col = "_epsilon_percent_displacement_plot"
        attack_records[plot_x_col] = attack_records[x_col].map(percent_displacement_plot_x)

    x_values = sorted(attack_records[plot_x_col].dropna().unique())

    positions = []
    values = []
    colors = []
    calculators = []
    point_x = []
    point_y = []

    rng = np.random.default_rng(12345)

    for x_value in x_values:
        for calculator in ["mace", "uma"]:
            rowset = attack_records[
                (attack_records[plot_x_col] == x_value)
                & (attack_records["calculator"] == calculator)
            ]

            box_values = []
            for _, row in rowset.iterrows():
                row_values, reason = value_getter(row)
                if row_values is None:
                    missing_rows.append({
                        "attack": attack,
                        "calculator": calculator,
                        "epsilon": row.get("epsilon"),
                        "x_col": x_col,
                        "x_value": row.get(x_col),
                        "run_id": row["run_id"],
                        "reason": reason,
                    })
                else:
                    box_values.extend(row_values.tolist())

            if box_values:
                position = epsilon_plot_position(x_value, calculator)
                positions.append(position)
                values.append(box_values)
                colors.append(CALCULATOR_COLORS[calculator])
                calculators.append(calculator)

                inlier_values = tukey_inlier_values(box_values)

                if len(inlier_values):
                    jitter = 10 ** rng.normal(loc=0.0, scale=0.004, size=len(inlier_values))
                    point_x.extend((position * jitter).tolist())
                    point_y.extend(inlier_values.tolist())

    return x_values, positions, values, colors, calculators, point_x, point_y


def draw_grouped_boxplot(
    ax,
    records,
    attack,
    value_getter,
    ylabel,
    missing_rows,
    x_col="epsilon",
    axis_mode=EPSILON_AXIS_RAW,
):
    x_values, positions, values, colors, calculators, point_x, point_y = collect_box_data(
        records,
        attack,
        value_getter,
        missing_rows,
        x_col=x_col,
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
        widths=epsilon_box_widths(positions),
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

    ax._preserve_manual_limits = True
    apply_epsilon_axis(ax, x_values, positions, axis_mode=axis_mode)
    ax.set_ylabel(ylabel)
    style_y_axis_no_offset(ax)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(x=0.03)

    return True


def plot_convergence_panel(
    ax,
    records,
    attack,
    step_col,
    conv_col,
    log_steps=False,
    x_col="epsilon",
    axis_mode=EPSILON_AXIS_RAW,
):
    attack_records = records[records["attack_label"] == attack].copy()
    if attack_records.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    plot_x_col = x_col
    if x_col == "epsilon_percent_displacement":
        plot_x_col = "_epsilon_percent_displacement_plot"
        attack_records[plot_x_col] = attack_records[x_col].map(percent_displacement_plot_x)

    x_values = sorted(attack_records[plot_x_col].dropna().unique())

    for calculator, color in CALCULATOR_COLORS.items():
        data = attack_records[
            (attack_records["calculator"] == calculator)
            & attack_records[step_col].notna()
            & attack_records[plot_x_col].notna()
        ].sort_values(plot_x_col)

        if data.empty:
            continue

        grouped = data.groupby(plot_x_col, as_index=False)[step_col].mean()

        ax.plot(
            grouped[plot_x_col],
            grouped[step_col],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=calculator.upper(),
        )

    ax._preserve_manual_limits = True
    apply_epsilon_axis(ax, x_values, axis_mode=axis_mode)
    ax.set_ylabel("Relaxation steps")
    style_relaxation_steps_axis(ax, log_scale=log_steps, tight_linear=not log_steps)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(x=0.03)

    return True


def make_convergence_figure(records, output_dir):
    axis_specs = [
        ("epsilon", EPSILON_AXIS_RAW, "figure_1_convergence_by_epsilon"),
    ]
    if has_percent_displacement_axis(records):
        axis_specs.append((
            "epsilon_percent_displacement",
            EPSILON_AXIS_PERCENT,
            "figure_1_convergence_by_epsilon_percent_displacement",
        ))

    for x_col, axis_mode, figure_name in axis_specs:
        fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.0), sharex=False, sharey=False)

        rows = [
            ("before_relax_steps", "before_relax_converged", "Relaxation before attack"),
            ("after_relax_steps", "after_relax_converged", "Relaxation after attack"),
        ]

        panel_index = 0
        for row_index, (step_col, conv_col, row_title) in enumerate(rows):
            for col_index, attack in enumerate(ATTACK_ORDER):
                ax = axes[row_index, col_index]
                plot_convergence_panel(
                    ax,
                    records,
                    attack,
                    step_col,
                    conv_col,
                    log_steps=(step_col == "after_relax_steps"),
                    x_col=x_col,
                    axis_mode=axis_mode,
                )

                if row_index == 0:
                    ax.set_title(attack)

                if col_index == 0:
                    ax.text(
                        -0.48,
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

        apply_shared_figure_header(fig, left=0.11)
        save_figure(fig, output_dir / figure_name)
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


def draw_grouped_ci(
    ax,
    records,
    attack,
    value_getter,
    ylabel,
    missing_rows,
    x_col="epsilon",
    axis_mode=EPSILON_AXIS_RAW,
):
    x_values, positions, values, colors, calculators, point_x, point_y = collect_box_data(
        records,
        attack,
        value_getter,
        missing_rows,
        x_col=x_col,
    )

    if not values:
        ax.text(0.5, 0.5, "No plottable data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    series = {
        "mace": {"x": [], "median": [], "lower": [], "upper": []},
        "uma": {"x": [], "median": [], "lower": [], "upper": []},
    }

    for position, box_values, calculator in zip(positions, values, calculators):
        ci = bootstrap_median_ci(box_values)
        if ci is None:
            continue

        median, lower, upper = ci
        series[calculator]["x"].append(position)
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

        ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)
        ax.plot(
            x,
            median,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=calculator.upper(),
        )

    ax._preserve_manual_limits = True
    apply_epsilon_axis(ax, x_values, positions, axis_mode=axis_mode)
    ax.set_ylabel(ylabel)
    style_y_axis_no_offset(ax)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(y=0.08)

    return True


def epsilon_axis_specs(records, figure_name):
    specs = [(EPSILON_AXIS_RAW, "epsilon", figure_name)]
    if has_percent_displacement_axis(records):
        specs.append((
            EPSILON_AXIS_PERCENT,
            "epsilon_percent_displacement",
            percent_figure_name(figure_name),
        ))
    return specs


def make_ci_figure(records, output_dir, figure_name, ylabel, rows):
    all_missing = []

    for axis_mode, x_col, output_figure_name in epsilon_axis_specs(records, figure_name):
        fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2), sharex=False, sharey=False)

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
                    x_col=x_col,
                    axis_mode=axis_mode,
                )

                for missing in attack_missing:
                    missing["figure"] = output_figure_name
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
        save_figure(fig, output_dir / output_figure_name)
        plt.close(fig)

    return all_missing


def make_distribution_figure(records, output_dir, figure_name, ylabel, rows):
    all_missing = []

    for axis_mode, x_col, output_figure_name in epsilon_axis_specs(records, figure_name):
        fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2), sharex=False, sharey=False)

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
                    x_col=x_col,
                    axis_mode=axis_mode,
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
        save_figure(fig, output_dir / output_figure_name)
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
    calculators = []
    point_x = []
    point_y = []

    rng = np.random.default_rng(12345)

    for n_steps in steps:
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
                position = step_plot_position(n_steps, calculator)
                positions.append(position)
                values.append(box_values)
                colors.append(CALCULATOR_COLORS[calculator])
                calculators.append(calculator)

                inlier_values = tukey_inlier_values(box_values)

                if len(inlier_values):
                    jitter = 10 ** rng.normal(loc=0.0, scale=0.004, size=len(inlier_values))
                    point_x.extend((position * jitter).tolist())
                    point_y.extend(inlier_values.tolist())

    return steps, positions, values, colors, calculators, point_x, point_y


def draw_grouped_boxplot_by_steps(ax, records, attack, epsilon, value_getter, ylabel, missing_rows):
    steps, positions, values, colors, calculators, point_x, point_y = collect_box_data_by_steps(
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
        widths=step_box_widths(positions),
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

    ax._preserve_manual_limits = True
    apply_step_axis(ax, steps, positions)
    ax.set_ylabel(ylabel)
    style_y_axis_no_offset(ax)
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


def tukey_inlier_values(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 4:
        return values

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    if not np.isfinite(iqr) or iqr <= 0:
        return values

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return values[(values >= lower) & (values <= upper)]


def collect_whisker_span_data(records, attack, value_getter, missing_rows, x_col="epsilon"):
    attack_records = records[records["attack_label"] == attack].copy()
    x_values = sorted(attack_records[x_col].dropna().unique())

    points = []

    for x_value in x_values:
        for calculator in ["mace", "uma"]:
            rowset = attack_records[
                (attack_records[x_col] == x_value)
                & (attack_records["calculator"] == calculator)
            ]

            values = []
            for _, row in rowset.iterrows():
                row_values, reason = value_getter(row)
                if row_values is None:
                    missing_rows.append({
                        "attack": attack,
                        "calculator": calculator,
                        "epsilon": row.get("epsilon"),
                        "x_col": x_col,
                        "x_value": x_value,
                        "run_id": row["run_id"],
                        "reason": reason,
                    })
                else:
                    values.extend(row_values.tolist())

            span = tukey_whisker_span(values)
            if span is not None:
                points.append({
                    "x": epsilon_plot_position(x_value, calculator),
                    "y": span,
                    "calculator": calculator,
                })

    return x_values, points


def draw_whisker_span(
    ax,
    records,
    attack,
    value_getter,
    ylabel,
    missing_rows,
    x_col="epsilon",
    axis_mode=EPSILON_AXIS_RAW,
):
    x_values, points = collect_whisker_span_data(
        records,
        attack,
        value_getter,
        missing_rows,
        x_col=x_col,
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
        x_plot_values = [point["x"] for point in calc_points]
        y_values = [point["y"] for point in calc_points]

        ax.plot(
            x_plot_values,
            y_values,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=calculator.upper(),
            zorder=3,
        )

    ax._preserve_manual_limits = True
    apply_epsilon_axis(ax, x_values, [point["x"] for point in points], axis_mode=axis_mode)
    ax.set_ylabel(ylabel)
    style_y_axis_no_offset(ax)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(x=0.05, y=0.12)

    return True


def make_whisker_span_figure(records, output_dir, figure_name, ylabel, rows):
    all_missing = []

    for axis_mode, x_col, output_figure_name in epsilon_axis_specs(records, figure_name):
        fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2), sharex=False, sharey=False)

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
                    x_col=x_col,
                    axis_mode=axis_mode,
                )

                for missing in attack_missing:
                    missing["figure"] = output_figure_name
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
        save_figure(fig, output_dir / output_figure_name)
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
                    "x": step_plot_position(n_steps, calculator),
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

    ax._preserve_manual_limits = True
    apply_step_axis(ax, steps, [point["x"] for point in points])
    ax.set_ylabel(ylabel)
    style_y_axis_no_offset(ax)
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
        left=0.12,
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

    for calculator, color in CALCULATOR_COLORS.items():
        data = attack_records[
            (attack_records["calculator"] == calculator)
            & attack_records[step_col].notna()
        ].sort_values("n_steps")

        if data.empty:
            continue

        grouped = data.groupby("n_steps", as_index=False)[step_col].mean()

        ax.plot(
            grouped["n_steps"],
            grouped[step_col],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=calculator.upper(),
        )

    ax._preserve_manual_limits = True
    apply_step_axis(ax, steps)
    ax.set_ylabel("Relaxation steps")
    style_relaxation_steps_axis(ax, tight_linear=True)
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
                    -0.48,
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
        left=0.12,
    )
    save_figure(fig, output_dir / "figure_4_convergence_by_n_steps")
    plt.close(fig)


def draw_grouped_ci_by_steps(ax, records, attack, epsilon, value_getter, ylabel, missing_rows):
    steps, positions, values, colors, calculators, point_x, point_y = collect_box_data_by_steps(
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

    for position, box_values, calculator in zip(positions, values, calculators):
        ci = bootstrap_median_ci(box_values)
        if ci is None:
            continue

        median, lower, upper = ci
        series[calculator]["x"].append(position)
        series[calculator]["median"].append(median)
        series[calculator]["lower"].append(lower)
        series[calculator]["upper"].append(upper)

    for calculator, data in series.items():
        if not data["x"]:
            continue

        order = np.argsort(np.asarray(data["x"], dtype=float))
        x = np.asarray(data["x"], dtype=float)[order]
        median = np.asarray(data["median"], dtype=float)[order]
        lower = np.asarray(data["lower"], dtype=float)[order]
        upper = np.asarray(data["upper"], dtype=float)[order]
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

    ax._preserve_manual_limits = True
    apply_step_axis(ax, steps, positions)
    ax.set_ylabel(ylabel)
    style_y_axis_no_offset(ax)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(y=0.08)

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
        left=0.12,
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
        left=0.12,
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


def normalized_topology_data(records):
    data = records.copy()
    for column in TOPOLOGY_METRICS + ["mean_displacement", "epsilon", "n_steps"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    components = {
        "jaccard_norm": "neighbor_jaccard_distance",
        "rdf_norm": "rdf_l1_distance",
        "coord_norm": "coordination_change_max",
    }

    for norm_col, raw_col in components.items():
        values = data[raw_col].replace([np.inf, -np.inf], np.nan)
        max_value = values.max(skipna=True)
        if pd.isna(max_value) or max_value <= 0:
            data[norm_col] = 0.0
        else:
            data[norm_col] = values / max_value

    data["topology_score"] = data[
        ["jaccard_norm", "rdf_norm", "coord_norm"]
    ].mean(axis=1)

    return data


def topology_discovery_data(records):
    data = normalized_topology_data(records)
    required = [
        "material_slug",
        "calculator",
        "attack_label",
        "neighbor_jaccard_distance",
        "rdf_l1_distance",
        "coordination_change_max",
        "jaccard_norm",
        "rdf_norm",
        "coord_norm",
        "topology_score",
        "epsilon",
        "n_steps",
    ]

    available = [column for column in required if column in data.columns]
    data = data[available].replace([np.inf, -np.inf], np.nan)

    metric_cols = [
        "neighbor_jaccard_distance",
        "rdf_l1_distance",
        "coordination_change_max",
        "jaccard_norm",
        "rdf_norm",
        "coord_norm",
        "topology_score",
    ]

    for column in metric_cols + ["epsilon", "n_steps"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.dropna(subset=[
        "material_slug",
        "calculator",
        "attack_label",
        "neighbor_jaccard_distance",
        "rdf_l1_distance",
        "coordination_change_max",
    ]).copy()


def topology_metric_axes(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.28)


def finite_metric_data(data, columns):
    clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    return clean.copy()


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
    fig.savefig(output_dir / "topology_by_attack_type.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_topology_mechanism_map(records, output_dir):
    data = finite_metric_data(
        normalized_topology_data(records),
        [
            "neighbor_jaccard_distance",
            "rdf_l1_distance",
            "coordination_change_max",
            "attack_label",
            "calculator",
        ],
    )

    if data.empty:
        return

    attacks = [attack for attack in ATTACK_ORDER if attack in set(data["attack_label"])]
    if not attacks:
        return

    fig, axes = plt.subplots(1, len(attacks), figsize=(4.2 * len(attacks), 3.8), sharex=False, sharey=False)
    axes = np.atleast_1d(axes)

    size_max = data["coordination_change_max"].max()
    if pd.isna(size_max) or size_max <= 0:
        data["_size"] = 28.0
    else:
        data["_size"] = 28.0 + 170.0 * (data["coordination_change_max"] / size_max)

    for ax, attack in zip(axes, attacks):
        subset = data[data["attack_label"] == attack]

        for calculator, color in CALCULATOR_COLORS.items():
            calc_subset = subset[subset["calculator"] == calculator]
            if calc_subset.empty:
                continue

            ax.scatter(
                calc_subset["neighbor_jaccard_distance"],
                calc_subset["rdf_l1_distance"],
                s=calc_subset["_size"],
                color=color,
                alpha=0.52,
                edgecolor="white",
                linewidth=0.45,
                label=calculator.upper(),
            )

        topology_metric_axes(
            ax,
            xlabel="Neighbor-graph Jaccard distance",
            ylabel="RDF L1 distance",
            title=attack,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.suptitle("Topology mechanism map: size = max coordination change", y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "topology_mechanism_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_topology_material_ranking(records, output_dir, max_materials=20):
    data = finite_metric_data(
        normalized_topology_data(records),
        ["material_slug", "jaccard_norm", "rdf_norm", "coord_norm", "topology_score"],
    )

    if data.empty:
        return

    ranked = (
        data.groupby("material_slug", as_index=False)
        .agg(
            jaccard_norm=("jaccard_norm", "mean"),
            rdf_norm=("rdf_norm", "mean"),
            coord_norm=("coord_norm", "mean"),
            topology_score=("topology_score", "mean"),
        )
        .sort_values("topology_score", ascending=False)
        .head(max_materials)
        .sort_values("topology_score", ascending=True)
    )

    if ranked.empty:
        return

    fig_height = max(4.0, 0.30 * len(ranked) + 1.5)
    fig, ax = plt.subplots(figsize=(7.4, fig_height))

    y = np.arange(len(ranked))
    left = np.zeros(len(ranked))

    components = [
        ("jaccard_norm", "Jaccard", "#0072B2"),
        ("rdf_norm", "RDF", "#009E73"),
        ("coord_norm", "Coordination", "#D55E00"),
    ]

    for column, label, color in components:
        values = ranked[column].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=color, alpha=0.78, label=label)
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(ranked["material_slug"])
    topology_metric_axes(
        ax,
        xlabel="Mean normalized topology contribution",
        ylabel="Material",
        title="Most topology-sensitive materials",
    )
    ax.legend(frameon=False, ncol=3, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "topology_material_ranking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_topology_model_disagreement(records, output_dir, max_rows=24):
    data = finite_metric_data(
        normalized_topology_data(records),
        ["material_slug", "attack_label", "calculator", "topology_score"],
    )

    if data.empty:
        return

    grouped = (
        data.groupby(["material_slug", "attack_label", "calculator"], as_index=False)
        .agg(topology_score=("topology_score", "mean"))
    )

    pivot = grouped.pivot_table(
        index=["material_slug", "attack_label"],
        columns="calculator",
        values="topology_score",
        aggfunc="mean",
    ).reset_index()

    if "mace" not in pivot.columns or "uma" not in pivot.columns:
        return

    pivot = pivot.dropna(subset=["mace", "uma"]).copy()
    if pivot.empty:
        return

    pivot["disagreement"] = (pivot["mace"] - pivot["uma"]).abs()
    pivot = pivot.sort_values("disagreement", ascending=False).head(max_rows)
    pivot = pivot.sort_values("disagreement", ascending=True)
    pivot["label"] = pivot["material_slug"].astype(str) + " / " + pivot["attack_label"].astype(str)

    fig_height = max(4.0, 0.32 * len(pivot) + 1.5)
    fig, ax = plt.subplots(figsize=(7.4, fig_height))

    y = np.arange(len(pivot))

    for index, row in enumerate(pivot.itertuples()):
        ax.plot(
            [row.mace, row.uma],
            [index, index],
            color="#999999",
            linewidth=1.2,
            zorder=1,
        )

    ax.scatter(pivot["mace"], y, color=CALCULATOR_COLORS["mace"], s=34, label="MACE", zorder=3)
    ax.scatter(pivot["uma"], y, color=CALCULATOR_COLORS["uma"], s=34, label="UMA", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(pivot["label"])
    topology_metric_axes(
        ax,
        xlabel="Mean normalized topology score",
        ylabel="Material / attack",
        title="Largest MACE-UMA topology-score disagreements",
    )
    ax.legend(frameon=False, ncol=2, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "topology_model_disagreement.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_topology_metric_coupling(records, output_dir):
    columns = [
        "neighbor_jaccard_distance",
        "rdf_l1_distance",
        "coordination_change_max",
        "mean_displacement",
        "epsilon",
        "n_steps",
    ]

    available = [column for column in columns if column in records.columns]
    data = finite_metric_data(records, available)

    if data.empty or len(available) < 2:
        return

    corr = data[available].corr(method="spearman")

    labels = {
        "neighbor_jaccard_distance": "Jaccard",
        "rdf_l1_distance": "RDF L1",
        "coordination_change_max": "Max coord.",
        "mean_displacement": "Mean disp.",
        "epsilon": "epsilon",
        "n_steps": "n_steps",
    }

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(np.arange(len(available)))
    ax.set_yticks(np.arange(len(available)))
    ax.set_xticklabels([labels[column] for column in available], rotation=35, ha="right")
    ax.set_yticklabels([labels[column] for column in available])

    for i in range(len(available)):
        for j in range(len(available)):
            value = corr.iloc[i, j]
            if pd.isna(value):
                text = "NA"
            else:
                text = f"{value:.2f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if pd.notna(value) and abs(value) > 0.55 else "#222222",
            )

    ax.set_title("Spearman coupling among topology and attack metrics")
    fig.colorbar(image, ax=ax, label="Spearman rho", shrink=0.82)

    fig.tight_layout()
    fig.savefig(output_dir / "topology_metric_coupling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_topology_synergy_outliers(records, output_dir, max_points=18):
    data = topology_discovery_data(records)
    if data.empty:
        return

    data["synergy_score"] = (
        data["jaccard_norm"] * data["rdf_norm"] * data["coord_norm"]
    ) ** (1.0 / 3.0)

    ranked = (
        data.groupby(["material_slug", "calculator", "attack_label"], as_index=False)
        .agg(
            synergy_score=("synergy_score", "mean"),
            jaccard=("neighbor_jaccard_distance", "mean"),
            rdf=("rdf_l1_distance", "mean"),
            coordination=("coordination_change_max", "mean"),
        )
        .sort_values("synergy_score", ascending=False)
        .head(max_points)
        .sort_values("synergy_score", ascending=True)
    )

    if ranked.empty:
        return

    labels = (
        ranked["material_slug"].astype(str)
        + " / "
        + ranked["calculator"].str.upper()
        + " / "
        + ranked["attack_label"].astype(str)
    )

    fig_height = max(4.2, 0.34 * len(ranked) + 1.4)
    fig, ax = plt.subplots(figsize=(7.6, fig_height))

    y = np.arange(len(ranked))
    ax.barh(y, ranked["synergy_score"], color="#7B3294", alpha=0.78)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Topology synergy score: geometric mean of normalized Jaccard, RDF, and coordination")
    ax.set_ylabel("Material / model / attack")
    ax.set_title("Candidate coupled-topology failure modes")

    ax.grid(True, axis="x", alpha=0.28)

    fig.tight_layout()
    fig.savefig(output_dir / "topology_synergy_outliers.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_topology_mechanism_triangle(records, output_dir):
    data = topology_discovery_data(records)
    if data.empty:
        return

    components = data[["jaccard_norm", "rdf_norm", "coord_norm"]].clip(lower=0)
    total = components.sum(axis=1)

    data = data[total > 0].copy()
    components = components.loc[data.index]
    total = total.loc[data.index]

    if data.empty:
        return

    data["jaccard_fraction"] = components["jaccard_norm"] / total
    data["rdf_fraction"] = components["rdf_norm"] / total
    data["coord_fraction"] = components["coord_norm"] / total

    data["x"] = data["rdf_fraction"] + 0.5 * data["coord_fraction"]
    data["y"] = (np.sqrt(3.0) / 2.0) * data["coord_fraction"]

    fig, ax = plt.subplots(figsize=(6.2, 5.6))

    triangle_x = [0.0, 1.0, 0.5, 0.0]
    triangle_y = [0.0, 0.0, np.sqrt(3.0) / 2.0, 0.0]
    ax.plot(triangle_x, triangle_y, color="#222222", linewidth=1.0)

    for calculator, color in CALCULATOR_COLORS.items():
        subset = data[data["calculator"] == calculator]
        if subset.empty:
            continue

        ax.scatter(
            subset["x"],
            subset["y"],
            s=24 + 180 * subset["topology_score"].fillna(0),
            color=color,
            alpha=0.42,
            edgecolor="white",
            linewidth=0.35,
            label=calculator.upper(),
        )

    ax.text(-0.04, -0.04, "Jaccard-dominant", ha="right", va="top", fontsize=8)
    ax.text(1.04, -0.04, "RDF-dominant", ha="left", va="top", fontsize=8)
    ax.text(0.5, np.sqrt(3.0) / 2.0 + 0.04, "Coordination-dominant", ha="center", va="bottom", fontsize=8)

    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.08, np.sqrt(3.0) / 2.0 + 0.12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Topology mechanism simplex: which topology signal dominates?")
    ax.legend(frameon=False, ncol=2, loc="lower center")

    fig.tight_layout()
    fig.savefig(output_dir / "topology_mechanism_simplex.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


COMPONENTS = ["x", "y", "z"]

def vector_component_values(run_dir, before_name, after_name, columns, component, scale=1.0, absolute=False):
    merged, reason = merge_atom_csvs(run_dir, before_name, after_name, columns)
    if merged is None:
        return None, reason

    before = merged[[f"{col}_before" for col in columns]].to_numpy(dtype=float)
    after = merged[[f"{col}_after" for col in columns]].to_numpy(dtype=float)
    delta = after - before
    index = columns.index(component)
    values = delta[:, index] * float(scale)

    if absolute:
        values = np.abs(values)

    return values, None


def displacement_component_values(row, before_name, after_name, component, absolute=False):
    reference = as_float(row.get("epsilon_reference_length_a"))
    if reference is None or reference <= 0:
        return None, "Missing positive epsilon_reference_length_a"

    return vector_component_values(
        row["run_dir"],
        before_name,
        after_name,
        ["x", "y", "z"],
        component,
        scale=100.0 / reference,
        absolute=absolute,
    )


def force_component_values(run_dir, before_name, after_name, component, absolute=False):
    force_component = {
        "x": "fx",
        "y": "fy",
        "z": "fz",
    }.get(component)

    if force_component is None:
        return None, f"Unknown force component: {component}"

    return vector_component_values(
        run_dir,
        before_name,
        after_name,
        ["fx", "fy", "fz"],
        force_component,
        scale=1.0,
        absolute=absolute,
    )


def force_angle_values(run_dir, before_name, after_name):
    merged, reason = merge_atom_csvs(run_dir, before_name, after_name, ["fx", "fy", "fz"])
    if merged is None:
        return None, reason

    before = merged[["fx_before", "fy_before", "fz_before"]].to_numpy(dtype=float)
    after = merged[["fx_after", "fy_after", "fz_after"]].to_numpy(dtype=float)

    before_norm = np.linalg.norm(before, axis=1)
    after_norm = np.linalg.norm(after, axis=1)
    denom = before_norm * after_norm

    angles = np.full(len(denom), np.nan, dtype=float)
    valid = denom > 0
    cos_theta = np.clip(np.sum(before[valid] * after[valid], axis=1) / denom[valid], -1.0, 1.0)
    angles[valid] = np.degrees(np.arccos(cos_theta))

    return angles[np.isfinite(angles)], None


def make_component_figures(epsilon_records, n_step_records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    states = [
        ("After attack, before relaxation", "before_forces.csv", "perturbed_forces.csv"),
        ("After attack, after relaxation", "before_forces.csv", "after_forces.csv"),
    ]

    for component in COMPONENTS:
        for absolute in [True]:
            suffix = "magnitude"

            displacement_rows = [
                (
                    label,
                    lambda before=before, after=after, comp=component, abs_val=absolute:
                        (lambda row: displacement_component_values(row, before, after, comp, abs_val)),
                )
                for label, before, after in states
            ]

            force_rows = [
                (
                    label,
                    lambda before=before, after=after, comp=component, abs_val=absolute:
                        (lambda row: force_component_values(row["run_dir"], before, after, comp, abs_val)),
                )
                for label, before, after in states
            ]

            make_distribution_figure(
                epsilon_records,
                output_dir,
                f"components_displacement_{component}_{suffix}_by_epsilon",
                f"{component} displacement (% min lattice)",
                displacement_rows,
            )
            make_ci_figure(
                epsilon_records,
                output_dir,
                f"components_displacement_{component}_{suffix}_ci_by_epsilon",
                f"Median {component} displacement (% min lattice)",
                displacement_rows,
            )
            make_distribution_figure(
                epsilon_records,
                output_dir,
                f"components_delta_force_{component}_{suffix}_by_epsilon",
                rf"{component} $\Delta$ force (eV/$\AA$)",
                force_rows,
            )
            make_ci_figure(
                epsilon_records,
                output_dir,
                f"components_delta_force_{component}_{suffix}_ci_by_epsilon",
                rf"Median {component} $\Delta$ force (eV/$\AA$)",
                force_rows,
            )

            make_distribution_by_steps_figure(
                n_step_records,
                output_dir,
                f"components_displacement_{component}_{suffix}_by_n_steps",
                f"{component} displacement (% min lattice)",
                displacement_rows,
            )
            make_ci_by_steps_figure(
                n_step_records,
                output_dir,
                f"components_displacement_{component}_{suffix}_ci_by_n_steps",
                f"Median {component} displacement (% min lattice)",
                displacement_rows,
            )
            make_distribution_by_steps_figure(
                n_step_records,
                output_dir,
                f"components_delta_force_{component}_{suffix}_by_n_steps",
                rf"{component} $\Delta$ force (eV/$\AA$)",
                force_rows,
            )
            make_ci_by_steps_figure(
                n_step_records,
                output_dir,
                f"components_delta_force_{component}_{suffix}_ci_by_n_steps",
                rf"Median {component} $\Delta$ force (eV/$\AA$)",
                force_rows,
            )

    angle_rows = [
        (
            label,
            lambda before=before, after=after:
                (lambda row: force_angle_values(row["run_dir"], before, after)),
        )
        for label, before, after in states
    ]

    make_distribution_figure(
        epsilon_records,
        output_dir,
        "components_delta_force_angle_by_epsilon",
        "Force-vector angle (deg)",
        angle_rows,
    )
    make_ci_figure(
        epsilon_records,
        output_dir,
        "components_delta_force_angle_ci_by_epsilon",
        "Median force-vector angle (deg)",
        angle_rows,
    )
    make_distribution_by_steps_figure(
        n_step_records,
        output_dir,
        "components_delta_force_angle_by_n_steps",
        "Force-vector angle (deg)",
        angle_rows,
    )
    make_ci_by_steps_figure(
        n_step_records,
        output_dir,
        "components_delta_force_angle_ci_by_n_steps",
        "Median force-vector angle (deg)",
        angle_rows,
    )


def make_topology_metric_figure_set(epsilon_records, n_step_records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("neighbor_jaccard_distance", "Neighbor Jaccard distance"),
        ("rdf_l1_distance", "RDF L1 distance"),
        ("coordination_change_max", "Max coordination change"),
    ]

    for column, label in metrics:
        rows = [
            ("Topology change", lambda col=column: (lambda row: scalar_distribution(row, col))),
            ("Topology change", lambda col=column: (lambda row: scalar_distribution(row, col))),
        ]

        make_distribution_figure(
            epsilon_records,
            output_dir,
            f"topology_{column}_by_epsilon",
            label,
            rows,
        )
        make_ci_figure(
            epsilon_records,
            output_dir,
            f"topology_{column}_ci_by_epsilon",
            f"Median {label}",
            rows,
        )
        make_distribution_by_steps_figure(
            n_step_records,
            output_dir,
            f"topology_{column}_by_n_steps",
            label,
            rows,
        )
        make_ci_by_steps_figure(
            n_step_records,
            output_dir,
            f"topology_{column}_ci_by_n_steps",
            f"Median {label}",
            rows,
        )


def recommended_repeat_tuple(n_atoms, cell_lengths, target_atoms=64):
    repeats = [1, 1, 1]
    lengths = np.asarray(cell_lengths, dtype=float)

    if n_atoms <= 0 or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        return (1, 1, 1)

    while n_atoms * repeats[0] * repeats[1] * repeats[2] < target_atoms:
        scaled_lengths = lengths * np.asarray(repeats, dtype=float)
        index = int(np.argmin(scaled_lengths))
        repeats[index] += 1

    return tuple(repeats)


def make_supercell_metadata(records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()

    for _, row in records.iterrows():
        key = (row.get("material_slug"), row.get("input_path"))
        if key in seen:
            continue
        seen.add(key)

        input_path = clean_value(row.get("input_path"))
        if input_path is None:
            continue

        path = Path(str(input_path))
        if not path.is_absolute():
            path = BASE_DIR / path

        try:
            atoms = read_structure(path)
            lengths = atoms.cell.lengths()
            repeats = recommended_repeat_tuple(len(atoms), lengths)
            rows.append({
                "material_slug": row.get("material_slug"),
                "material_label": row.get("material_label"),
                "input_path": str(input_path),
                "unit_cell_atoms": len(atoms),
                "cell_a": float(lengths[0]),
                "cell_b": float(lengths[1]),
                "cell_c": float(lengths[2]),
                "min_lattice_a": float(np.min(lengths)),
                "repeat_tuple": f"{repeats[0]}x{repeats[1]}x{repeats[2]}",
                "supercell_atoms": int(len(atoms) * repeats[0] * repeats[1] * repeats[2]),
            })
        except Exception as exc:
            rows.append({
                "material_slug": row.get("material_slug"),
                "material_label": row.get("material_label"),
                "input_path": str(input_path),
                "error": str(exc),
            })

    pd.DataFrame(rows).to_csv(output_dir / "supercell_metadata.csv", index=False)


def tukey_outlier_rows(data, group_cols, value_col):
    rows = []
    clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col]).copy()

    if clean.empty:
        return rows

    for group_key, group in clean.groupby(group_cols, dropna=False):
        values = pd.to_numeric(group[value_col], errors="coerce").dropna()
        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = group[(group[value_col] < lower) | (group[value_col] > upper)].copy()

        for _, row in outliers.iterrows():
            item = row.to_dict()
            item["outlier_metric"] = value_col
            item["outlier_lower_bound"] = float(lower)
            item["outlier_upper_bound"] = float(upper)
            rows.append(item)

    return rows


def make_outlier_reports(records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        "epsilon",
        "epsilon_percent_displacement",
        "mean_displacement",
        "max_displacement",
        "before_relax_steps",
        "after_relax_steps",
        "neighbor_jaccard_distance",
        "coordination_change_mean",
        "coordination_change_max",
        "rdf_l1_distance",
    ]

    all_rows = []
    summary_rows = []

    group_cols = ["calculator", "attack_label"]
    for column in metric_cols:
        if column not in records.columns:
            continue

        data = records.copy()
        data[column] = pd.to_numeric(data[column], errors="coerce")
        rows = tukey_outlier_rows(data, group_cols, column)

        pd.DataFrame(rows).to_csv(output_dir / f"outliers_{column}.csv", index=False)
        summary_rows.append({
            "metric": column,
            "n_outliers": len(rows),
        })
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(output_dir / "outliers_all_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "outliers_summary.csv", index=False)


def make_topology_figures(records, output_dir):
    if records.empty or not topology_ready(records):
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = records.copy()
    for column in TOPOLOGY_METRICS + ["mean_displacement", "epsilon", "n_steps"]:
        if column in clean.columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")

    save_topology_summary(clean, output_dir)
    make_topology_by_attack_type(clean, output_dir)
    make_topology_mechanism_map(clean, output_dir)
    make_topology_material_ranking(clean, output_dir)
    make_topology_model_disagreement(clean, output_dir)
    make_topology_metric_coupling(clean, output_dir)
    make_topology_synergy_outliers(clean, output_dir)
    make_topology_mechanism_triangle(clean, output_dir)

    for material_slug, material_records in clean.groupby("material_slug"):
        material_output_dir = output_dir / str(material_slug)
        material_output_dir.mkdir(parents=True, exist_ok=True)
        save_topology_summary(material_records, material_output_dir)
        make_topology_by_attack_type(material_records, material_output_dir)
        make_topology_mechanism_map(material_records, material_output_dir)
        make_topology_model_disagreement(material_records, material_output_dir)
        make_topology_metric_coupling(material_records, material_output_dir)
        make_topology_synergy_outliers(material_records, material_output_dir)
        make_topology_mechanism_triangle(material_records, material_output_dir)


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
    make_supercell_metadata(records, args.output_dir / "supercells")

    epsilon_records = records[
        ~records["run_id"].str.contains("_steps", regex=False)
    ].copy()

    n_step_records = records[
        records["run_id"].str.contains("_steps", regex=False)
    ].copy()

    make_convergence_figure(epsilon_records, args.output_dir)
    make_component_figures(epsilon_records, n_step_records, args.output_dir / "components")
    make_topology_metric_figure_set(epsilon_records, n_step_records, args.output_dir / "Topology")
    make_outlier_reports(records, args.output_dir / "Outliers")

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
