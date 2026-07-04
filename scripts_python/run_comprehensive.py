#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import math
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
EPSILON_AXIS_PERCENT = "percent_min_lattice"
EPSILON_AXIS_PERCENT_X = "percent_x_lattice"
EPSILON_AXIS_PERCENT_Y = "percent_y_lattice"
EPSILON_AXIS_PERCENT_Z = "percent_z_lattice"

EPSILON_PERCENT_AXIS_LABELS = {
    EPSILON_AXIS_PERCENT: "Epsilon (% min lattice)",
    EPSILON_AXIS_PERCENT_X: "Epsilon (% x lattice)",
    EPSILON_AXIS_PERCENT_Y: "Epsilon (% y lattice)",
    EPSILON_AXIS_PERCENT_Z: "Epsilon (% z lattice)",
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

    if axis_mode in EPSILON_PERCENT_AXIS_LABELS:
        ax.set_xlabel(EPSILON_PERCENT_AXIS_LABELS[axis_mode])
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


def epsilon_lattice_lengths_from_summary_row(row):
    input_path = clean_value(row.get("input_path"))
    if input_path is None:
        return {
            "min": np.nan,
            "x": np.nan,
            "y": np.nan,
            "z": np.nan,
            "reason": "Missing input_path",
        }

    path = Path(str(input_path))
    if not path.is_absolute():
        path = BASE_DIR / path

    try:
        atoms = read_structure(path)
        lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    except Exception as exc:
        return {
            "min": np.nan,
            "x": np.nan,
            "y": np.nan,
            "z": np.nan,
            "reason": f"Could not read structure with ASE: {exc}",
        }

    if len(lengths) < 3 or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        return {
            "min": np.nan,
            "x": np.nan,
            "y": np.nan,
            "z": np.nan,
            "reason": "Missing positive lattice lengths",
        }

    return {
        "min": float(np.min(lengths)),
        "x": float(lengths[0]),
        "y": float(lengths[1]),
        "z": float(lengths[2]),
        "reason": None,
    }


def percent_displacement_from_epsilon(epsilon, reference_length_a):
    epsilon = as_float(epsilon)
    reference_length_a = as_float(reference_length_a)
    if epsilon is None or reference_length_a is None or reference_length_a <= 0:
        return np.nan
    return 100.0 * epsilon / reference_length_a


def epsilon_percent_axis_specs(records):
    specs = []

    candidates = [
        (
            EPSILON_AXIS_PERCENT,
            "epsilon_percent_displacement",
            "percent_displacement",
        ),
        (
            EPSILON_AXIS_PERCENT_X,
            "epsilon_percent_displacement_x_lattice",
            "percent_x_lattice",
        ),
        (
            EPSILON_AXIS_PERCENT_Y,
            "epsilon_percent_displacement_y_lattice",
            "percent_y_lattice",
        ),
        (
            EPSILON_AXIS_PERCENT_Z,
            "epsilon_percent_displacement_z_lattice",
            "percent_z_lattice",
        ),
    ]

    for axis_mode, column, suffix in candidates:
        if has_percent_displacement_axis(records, column):
            specs.append((axis_mode, column, suffix))

    return specs


def has_percent_displacement_axis(records, column="epsilon_percent_displacement"):
    if column not in records.columns:
        return False
    values = positive_finite_values(records[column].dropna())
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
        epsilon_lattice_lengths = epsilon_lattice_lengths_from_summary_row(row)
        epsilon_reference_length_a = epsilon_lattice_lengths["min"]
        epsilon_reference_reason = epsilon_lattice_lengths["reason"]

        epsilon_percent_displacement = percent_displacement_from_epsilon(
            epsilon_value,
            epsilon_lattice_lengths["min"],
        )
        epsilon_percent_displacement_x_lattice = percent_displacement_from_epsilon(
            epsilon_value,
            epsilon_lattice_lengths["x"],
        )
        epsilon_percent_displacement_y_lattice = percent_displacement_from_epsilon(
            epsilon_value,
            epsilon_lattice_lengths["y"],
        )
        epsilon_percent_displacement_z_lattice = percent_displacement_from_epsilon(
            epsilon_value,
            epsilon_lattice_lengths["z"],
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
            "epsilon_lattice_x_a": epsilon_lattice_lengths["x"],
            "epsilon_lattice_y_a": epsilon_lattice_lengths["y"],
            "epsilon_lattice_z_a": epsilon_lattice_lengths["z"],
            "epsilon_percent_displacement_x_lattice": epsilon_percent_displacement_x_lattice,
            "epsilon_percent_displacement_y_lattice": epsilon_percent_displacement_y_lattice,
            "epsilon_percent_displacement_z_lattice": epsilon_percent_displacement_z_lattice,
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
            "perturbed_topology_edge_changes_csv": clean_value(
                row.get("perturbed_topology_edge_changes_csv")
            ),
            "perturbed_neighbor_edges_before": as_float(
                row.get("perturbed_neighbor_edges_before")
            ),
            "perturbed_neighbor_edges_after": as_float(
                row.get("perturbed_neighbor_edges_after")
            ),
            "perturbed_neighbor_edges_added": as_float(
                row.get("perturbed_neighbor_edges_added")
            ),
            "perturbed_neighbor_edges_removed": as_float(
                row.get("perturbed_neighbor_edges_removed")
            ),
            "perturbed_neighbor_edge_change_count": as_float(
                row.get("perturbed_neighbor_edge_change_count")
            ),
            "perturbed_neighbor_jaccard_distance": as_float(
                row.get("perturbed_neighbor_jaccard_distance")
            ),
            "perturbed_coordination_change_mean": as_float(
                row.get("perturbed_coordination_change_mean")
            ),
            "perturbed_coordination_change_max": as_float(
                row.get("perturbed_coordination_change_max")
            ),
            "perturbed_rdf_l1_distance": as_float(
                row.get("perturbed_rdf_l1_distance")
            ),
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
    def safe_limits(ax, axis_name):
        values = _artist_values_for_axis(ax, axis_name)
        values = clean_numeric_array(values)

        if len(values) == 0:
            return None

        scale = (
            ax.get_xscale()
            if axis_name == "x"
            else ax.get_yscale()
        )

        if scale == "log":
            values = values[values > 0]

            if len(values) == 0:
                return None

        limits = _tight_limit(values)

        if limits is None:
            return None

        low, high = limits

        if scale == "log":
            smallest_positive = float(np.min(values))

            if not np.isfinite(low) or low <= 0:
                low = smallest_positive * 0.8

            if not np.isfinite(high) or high <= low:
                high = max(
                    float(np.max(values)) * 1.2,
                    low * 1.2,
                )

        if not np.isfinite(low) or not np.isfinite(high):
            return None

        if high <= low:
            return None

        return low, high

    for ax in fig.axes:
        if not ax.has_data():
            continue

        if (
            getattr(ax, "_preserve_parametric_limits", False)
            or getattr(ax, "_preserve_manual_limits", False)
        ):
            continue

        y_limits = safe_limits(ax, "y")

        if y_limits is not None:
            ax.set_ylim(*y_limits)

        xlabel = ax.get_xlabel().lower()

        if "displacement" in xlabel or "rdf" in xlabel:
            x_limits = safe_limits(ax, "x")

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


def paired_relaxation_rows(
    records,
    x_getters,
    y_getters,
):
    rows = []

    for _, row in records.iterrows():
        x_before, _ = x_getters[0](row)
        x_after, _ = x_getters[1](row)
        y_before, _ = y_getters[0](row)
        y_after, _ = y_getters[1](row)

        if any(
            value is None
            for value in [
                x_before,
                x_after,
                y_before,
                y_after,
            ]
        ):
            continue

        epsilon = as_float(row.get("epsilon"))

        if epsilon is None or epsilon <= 0:
            continue

        values = [
            float(x_before[0]),
            float(x_after[0]),
            float(y_before[0]),
            float(y_after[0]),
        ]

        if not np.all(np.isfinite(values)):
            continue

        rows.append({
            "run_id": row.get("run_id"),
            "material_slug": row.get("material_slug"),
            "calculator": row.get("calculator"),
            "attack_label": row.get("attack_label"),
            "epsilon": float(epsilon),
            "x_before": values[0],
            "x_after": values[1],
            "y_before": values[2],
            "y_after": values[3],
        })

    columns = [
        "run_id",
        "material_slug",
        "calculator",
        "attack_label",
        "epsilon",
        "x_before",
        "x_after",
        "y_before",
        "y_after",
    ]

    return pd.DataFrame(rows, columns=columns)


def grouped_relaxation_vectors(data):
    return (
        data.groupby(
            [
                "attack_label",
                "calculator",
                "epsilon",
            ],
            as_index=False,
        )
        .agg({
            "x_before": "median",
            "x_after": "median",
            "y_before": "median",
            "y_after": "median",
        })
    )


def make_paired_relaxation_figure(
    records,
    output_dir,
    figure_name,
    title,
    x_label,
    y_label,
    x_getters,
    y_getters,
    attacks_to_plot=ATTACK_ORDER,
    x_log=False,
    y_log=False,
):
    # Relaxation differences may be zero or negative.
    # Therefore, logarithmic axes are intentionally not used.
    _ = x_log, y_log

    paired = paired_relaxation_rows(
        records,
        x_getters,
        y_getters,
    )

    if paired.empty:
        return

    paired["delta_x"] = (
        paired["x_after"] - paired["x_before"]
    )
    paired["delta_y"] = (
        paired["y_after"] - paired["y_before"]
    )

    grouped = (
        paired.groupby(
            [
                "attack_label",
                "calculator",
                "epsilon",
            ],
            as_index=False,
        )
        .agg({
            "delta_x": "median",
            "delta_y": "median",
        })
    )

    fig, axes = plt.subplots(
        1,
        len(attacks_to_plot),
        figsize=(5.0 * len(attacks_to_plot), 4.8),
        squeeze=False,
    )
    axes = axes.ravel()

    def full_limits(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return -1.0, 1.0

        minimum = min(float(np.min(values)), 0.0)
        maximum = max(float(np.max(values)), 0.0)
        span = maximum - minimum

        if not np.isfinite(span) or span <= 0:
            span = max(
                abs(minimum),
                abs(maximum),
                1.0,
            )

        padding = 0.10 * span
        return minimum - padding, maximum + padding

    for column, attack in enumerate(attacks_to_plot):
        ax = axes[column]

        attack_data = grouped[
            grouped["attack_label"] == attack
        ].copy()

        if attack_data.empty:
            ax.text(
                0.5,
                0.5,
                "No paired data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        else:
            for calculator in ["mace", "uma"]:
                calculator_data = attack_data[
                    attack_data["calculator"] == calculator
                ]

                if calculator_data.empty:
                    continue

                ax.scatter(
                    calculator_data["delta_x"],
                    calculator_data["delta_y"],
                    s=48,
                    marker="o",
                    color=CALCULATOR_COLORS[calculator],
                    edgecolor="white",
                    linewidth=0.7,
                    alpha=0.90,
                    zorder=3,
                )

            ax.set_xlim(
                full_limits(attack_data["delta_x"])
            )
            ax.set_ylim(
                full_limits(attack_data["delta_y"])
            )

        ax.axvline(
            0.0,
            color="#666666",
            linestyle=":",
            linewidth=0.9,
            zorder=1,
        )
        ax.axhline(
            0.0,
            color="#666666",
            linestyle=":",
            linewidth=0.9,
            zorder=1,
        )

        # Preserve the complete range when save_figure() is called.
        ax._preserve_manual_limits = True

        ax.set_title(attack)
        ax.set_xlabel(
            f"Relaxation change in {x_label}"
        )
        ax.set_ylabel(
            f"Relaxation change in {y_label}"
        )
        ax.grid(True, alpha=0.22)

        add_panel_label(
            ax,
            chr(ord("A") + column),
        )

    # Exactly two legend entries.
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=CALCULATOR_COLORS["mace"],
            markeredgecolor="white",
            label="MACE",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=CALCULATOR_COLORS["uma"],
            markeredgecolor="white",
            label="UMA",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
    )

    fig.suptitle(
        title,
        y=1.03,
        fontsize=14,
    )

    fig.text(
        0.5,
        0.012,
        (
            "Each circle = median(after perturbation and relaxation "
            "- after perturbation before relaxation) "
            "across materials at one epsilon"
        ),
        ha="center",
        fontsize=8.5,
        color="#555555",
    )

    fig.tight_layout(
        rect=[0.03, 0.06, 1.0, 0.89]
    )

    save_figure(
        fig,
        Path(output_dir) / figure_name,
    )
    plt.close(fig)


def make_parametric_figure_set(
    records,
    output_dir,
    suffix,
    attacks_to_plot,
    bubble_label,
):
    displacement_getters = displacement_metric_getters()
    force_getters = delta_force_metric_getters()
    plot_convergence_getters = [
        lambda row: scalar_distribution(
            row,
            "after_relax_steps",
        ),
        lambda row: scalar_distribution(
            row,
            "after_relax_steps",
        ),
    ]

    paired_convergence_getters = [
        lambda row: scalar_distribution(
            row,
            "before_relax_steps",
        ),
        lambda row: scalar_distribution(
            row,
            "after_relax_steps",
        ),
    ]

    convergence_displacement_missing = (
        make_parametric_state_figure(
            records=records,
            output_dir=output_dir,
            figure_name=(
                f"figure_7_convergence_vs_displacement_by_{suffix}"
            ),
            title=(
                f"Convergence vs displacement by {bubble_label}"
            ),
            x_label=r"Median displacement ($\AA$)",
            y_label="Relaxation steps",
            bubble_label=bubble_label,
            attacks_to_plot=attacks_to_plot,
            x_getters=displacement_getters,
            y_getters=plot_convergence_getters,
        )
    )

    convergence_force_missing = (
        make_parametric_state_figure(
            records=records,
            output_dir=output_dir,
            figure_name=(
                f"figure_8_convergence_vs_delta_force_by_{suffix}"
            ),
            title=(
                f"Convergence vs delta force by {bubble_label}"
            ),
            x_label=r"Median $\Delta$ force (eV/$\AA$)",
            y_label="Relaxation steps",
            bubble_label=bubble_label,
            attacks_to_plot=attacks_to_plot,
            x_getters=force_getters,
            y_getters=plot_convergence_getters,
            x_log=True,
        )
    )

    force_displacement_missing = (
        make_parametric_state_figure(
            records=records,
            output_dir=output_dir,
            figure_name=(
                f"figure_9_delta_force_vs_displacement_by_{suffix}"
            ),
            title=(
                f"Delta force vs displacement by {bubble_label}"
            ),
            x_label=r"Median displacement ($\AA$)",
            y_label=r"Median $\Delta$ force (eV/$\AA$)",
            bubble_label=bubble_label,
            attacks_to_plot=attacks_to_plot,
            x_getters=displacement_getters,
            y_getters=force_getters,
            x_log=True,
            y_log=True,
        )
    )

    if suffix == "epsilon":
        make_paired_relaxation_figure(
            records=records,
            output_dir=output_dir,
            figure_name=(
                "figure_7_1_convergence_vs_displacement_"
                "relaxation_by_epsilon"
            ),
            title=(
                "Relaxation change: convergence vs displacement"
            ),
            x_label=r"Median displacement ($\AA$)",
            y_label="Relaxation steps",
            x_getters=displacement_getters,
            y_getters=paired_convergence_getters,
        )

        make_paired_relaxation_figure(
            records=records,
            output_dir=output_dir,
            figure_name=(
                "figure_8_1_convergence_vs_delta_force_"
                "relaxation_by_epsilon"
            ),
            title=(
                "Relaxation change: convergence vs delta force"
            ),
            x_label=r"Median $\Delta$ force (eV/$\AA$)",
            y_label="Relaxation steps",
            x_getters=force_getters,
            y_getters=paired_convergence_getters,
            x_log=True,
        )

        make_paired_relaxation_figure(
            records=records,
            output_dir=output_dir,
            figure_name=(
                "figure_9_1_delta_force_vs_displacement_"
                "relaxation_by_epsilon"
            ),
            title=(
                "Relaxation change: delta force vs displacement"
            ),
            x_label=r"Median displacement ($\AA$)",
            y_label=r"Median $\Delta$ force (eV/$\AA$)",
            x_getters=displacement_getters,
            y_getters=force_getters,
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
    if str(x_col).startswith("epsilon_percent_displacement"):
        plot_x_col = f"_{x_col}_plot"
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
    if str(x_col).startswith("epsilon_percent_displacement"):
        plot_x_col = f"_{x_col}_plot"
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


def make_convergence_figure(records, output_dir, axis_specs=None):
    if axis_specs is None:
        axis_specs = epsilon_axis_specs(records, "figure_1_convergence_by_epsilon")

    for axis_mode, x_col, figure_name in axis_specs:
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


def epsilon_axis_specs(records, figure_name, include_raw=True, include_min=True, include_xyz=False):
    specs = []

    if include_raw:
        specs.append((EPSILON_AXIS_RAW, "epsilon", figure_name))

    for axis_mode, x_col, suffix in epsilon_percent_axis_specs(records):
        is_min_lattice = x_col == "epsilon_percent_displacement"
        is_xyz_lattice = x_col in {
            "epsilon_percent_displacement_x_lattice",
            "epsilon_percent_displacement_y_lattice",
            "epsilon_percent_displacement_z_lattice",
        }

        if is_min_lattice and include_min:
            specs.append((axis_mode, x_col, f"{figure_name}_{suffix}"))

        if is_xyz_lattice and include_xyz:
            specs.append((axis_mode, x_col, f"{figure_name}_{suffix}"))

    return specs


def epsilon_component_axis_specs(records, figure_name):
    return epsilon_axis_specs(
        records,
        figure_name,
        include_raw=False,
        include_min=False,
        include_xyz=True,
    )


def make_ci_figure(records, output_dir, figure_name, ylabel, rows, axis_specs=None):
    all_missing = []

    if axis_specs is None:
        axis_specs = epsilon_axis_specs(records, figure_name)

    for axis_mode, x_col, output_figure_name in axis_specs:
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


def make_distribution_figure(records, output_dir, figure_name, ylabel, rows, axis_specs=None):
    all_missing = []

    if axis_specs is None:
        axis_specs = epsilon_axis_specs(records, figure_name)

    for axis_mode, x_col, output_figure_name in axis_specs:
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


def make_whisker_span_figure(records, output_dir, figure_name, ylabel, rows, axis_specs=None):
    all_missing = []

    if axis_specs is None:
        axis_specs = epsilon_axis_specs(records, figure_name)

    for axis_mode, x_col, output_figure_name in axis_specs:
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


COMPONENTS = ["x", "y", "z"]


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


def make_lattice_axis_component_figures(epsilon_records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    make_convergence_figure(
        epsilon_records,
        output_dir,
        axis_specs=epsilon_component_axis_specs(
            epsilon_records,
            "figure_1_convergence_by_epsilon",
        ),
    )

    force_rows = [
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
    ]

    force_angle_rows = [
        (
            "After attack, before relaxation",
            lambda: (lambda row: force_angle_values(
                row["run_dir"],
                "before_forces.csv",
                "perturbed_forces.csv",
            )),
        ),
        (
            "After attack, after relaxation",
            lambda: (lambda row: force_angle_values(
                row["run_dir"],
                "before_forces.csv",
                "after_forces.csv",
            )),
        ),
    ]

    displacement_rows = [
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
    ]

    make_distribution_figure(
        epsilon_records,
        output_dir,
        "figure_2_delta_force_by_epsilon",
        r"$\Delta$ force (eV/$\AA$)",
        force_rows,
        axis_specs=epsilon_component_axis_specs(
            epsilon_records,
            "figure_2_delta_force_by_epsilon",
        ),
    )

    make_ci_figure(
        epsilon_records,
        output_dir,
        "figure_2_delta_force_ci_by_epsilon",
        r"Median $\Delta$ force with 95% CI (eV/$\AA$)",
        force_rows,
        axis_specs=epsilon_component_axis_specs(
            epsilon_records,
            "figure_2_delta_force_ci_by_epsilon",
        ),
    )

    make_whisker_span_figure(
        epsilon_records,
        output_dir,
        "figure_2_delta_force_whisker_span_by_epsilon",
        r"$\Delta$ force whisker span (eV/$\AA$)",
        force_rows,
        axis_specs=epsilon_component_axis_specs(
            epsilon_records,
            "figure_2_delta_force_whisker_span_by_epsilon",
        ),
    )

    make_distribution_figure(
        epsilon_records,
        output_dir,
        "figure_2_delta_force_angle_by_epsilon",
        "Force-vector angle (deg)",
        force_angle_rows,
        axis_specs=epsilon_component_axis_specs(
            epsilon_records,
            "figure_2_delta_force_angle_by_epsilon",
        ),
    )

    make_ci_figure(
        epsilon_records,
        output_dir,
        "figure_2_delta_force_angle_ci_by_epsilon",
        "Median force-vector angle (deg)",
        force_angle_rows,
        axis_specs=epsilon_component_axis_specs(
            epsilon_records,
            "figure_2_delta_force_angle_ci_by_epsilon",
        ),
    )

    make_whisker_span_figure(
        epsilon_records,
        output_dir,
        "figure_2_delta_force_angle_whisker_span_by_epsilon",
        "Force-vector angle whisker span (deg)",
        force_angle_rows,
        axis_specs=epsilon_component_axis_specs(
            epsilon_records,
            "figure_2_delta_force_angle_whisker_span_by_epsilon",
        ),
    )

    make_distribution_figure(
        epsilon_records,
        output_dir,
        "figure_3_displacement_by_epsilon",
        r"Displacement ($\AA$)",
        displacement_rows,
        axis_specs=epsilon_component_axis_specs(
            epsilon_records,
            "figure_3_displacement_by_epsilon",
        ),
    )

    make_ci_figure(
        epsilon_records,
        output_dir,
        "figure_3_displacement_ci_by_epsilon",
        r"Median displacement with 95% CI ($\AA$)",
        displacement_rows,
        axis_specs=epsilon_component_axis_specs(
            epsilon_records,
            "figure_3_displacement_ci_by_epsilon",
        ),
    )

    make_whisker_span_figure(
        epsilon_records,
        output_dir,
        "figure_3_displacement_whisker_span_by_epsilon",
        r"Displacement whisker span ($\AA$)",
        displacement_rows,
        axis_specs=epsilon_component_axis_specs(
            epsilon_records,
            "figure_3_displacement_whisker_span_by_epsilon",
        ),
    )


def topology_scalar_values(row, column):
    value = row.get(column)
    if value is None or pd.isna(value):
        return None, f"Missing {column}"

    value = float(value)
    if not np.isfinite(value):
        return None, f"Nonfinite {column}"

    return np.array([value], dtype=float), None


def force_angle_rows():
    return [
        (
            "After attack, before relaxation",
            lambda: (lambda row: force_angle_values(
                row["run_dir"],
                "before_forces.csv",
                "perturbed_forces.csv",
            )),
        ),
        (
            "After attack, after relaxation",
            lambda: (lambda row: force_angle_values(
                row["run_dir"],
                "before_forces.csv",
                "after_forces.csv",
            )),
        ),
    ]


def displacement_rows():
    return [
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
    ]


def topology_stage_rows(column):
    return [
        (
            "After attack, before relaxation",
            lambda col=f"perturbed_{column}": (
                lambda row: topology_scalar_values(row, col)
            ),
        ),
        (
            "After attack, after relaxation",
            lambda col=column: (
                lambda row: topology_scalar_values(row, col)
            ),
        ),
    ]


def topology_stage_getters(column):
    return [
        topology_metric_getter(f"perturbed_{column}"),
        topology_metric_getter(column),
    ]


def force_angle_metric_getters():
    return [
        lambda row: metric_distribution(row, lambda item: force_angle_values(
            item["run_dir"],
            "before_forces.csv",
            "perturbed_forces.csv",
        )),
        lambda row: metric_distribution(row, lambda item: force_angle_values(
            item["run_dir"],
            "before_forces.csv",
            "after_forces.csv",
        )),
    ]


def delta_force_metric_getters():
    return [
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
    ]


def displacement_metric_getters():
    return [
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
    ]


def topology_metric_getter(column):
    return lambda row, col=column: scalar_distribution(row, col)


def make_delta_force_angle_figure_set(epsilon_records, n_step_records, output_dir):
    missing = []

    rows = force_angle_rows()

    missing.extend(make_distribution_figure(
        records=epsilon_records,
        output_dir=output_dir,
        figure_name="figure_2_delta_force_angle_by_epsilon",
        ylabel="Force-vector angle (deg)",
        rows=rows,
    ))

    missing.extend(make_ci_figure(
        records=epsilon_records,
        output_dir=output_dir,
        figure_name="figure_2_delta_force_angle_ci_by_epsilon",
        ylabel="Median force-vector angle with 95% CI (deg)",
        rows=rows,
    ))

    missing.extend(make_whisker_span_figure(
        records=epsilon_records,
        output_dir=output_dir,
        figure_name="figure_2_delta_force_angle_whisker_span_by_epsilon",
        ylabel="Force-vector angle whisker span (deg)",
        rows=rows,
    ))

    missing.extend(make_distribution_by_steps_figure(
        records=n_step_records,
        output_dir=output_dir,
        figure_name="figure_5_delta_force_angle_by_n_steps",
        ylabel="Force-vector angle (deg)",
        epsilon=0.1,
        rows=rows,
    ))

    missing.extend(make_ci_by_steps_figure(
        records=n_step_records,
        output_dir=output_dir,
        figure_name="figure_5_delta_force_angle_ci_by_n_steps",
        ylabel="Median force-vector angle with 95% CI (deg)",
        epsilon=0.1,
        rows=rows,
    ))

    missing.extend(make_whisker_span_by_steps_figure(
        records=n_step_records,
        output_dir=output_dir,
        figure_name="figure_5_delta_force_angle_whisker_span_by_n_steps",
        ylabel="Force-vector angle whisker span (deg)",
        epsilon=0.1,
        rows=rows,
    ))

    angle_getters = force_angle_metric_getters()
    displacement_getters = displacement_metric_getters()

    missing.extend(make_parametric_state_figure(
        records=epsilon_records,
        output_dir=output_dir,
        figure_name="figure_8_convergence_vs_delta_force_angle_by_epsilon",
        title="Convergence vs delta-force angle by epsilon",
        x_label="Median force-vector angle (deg)",
        y_label="Relaxation steps",
        bubble_label="epsilon",
        attacks_to_plot=ATTACK_ORDER,
        x_getters=angle_getters,
        y_getters=[
            lambda row: scalar_distribution(row, "after_relax_steps"),
            lambda row: scalar_distribution(row, "after_relax_steps"),
        ],
    ))

    missing.extend(make_parametric_state_figure(
        records=n_step_records,
        output_dir=output_dir,
        figure_name="figure_8_convergence_vs_delta_force_angle_by_n_steps",
        title="Convergence vs delta-force angle by n_steps",
        x_label="Median force-vector angle (deg)",
        y_label="Relaxation steps",
        bubble_label="n_steps",
        attacks_to_plot=STEP_ATTACK_ORDER,
        x_getters=angle_getters,
        y_getters=[
            lambda row: scalar_distribution(row, "after_relax_steps"),
            lambda row: scalar_distribution(row, "after_relax_steps"),
        ],
    ))

    missing.extend(make_parametric_state_figure(
        records=epsilon_records,
        output_dir=output_dir,
        figure_name="figure_9_delta_force_angle_vs_displacement_by_epsilon",
        title="Delta-force angle vs displacement by epsilon",
        x_label=r"Median displacement ($\AA$)",
        y_label="Median force-vector angle (deg)",
        bubble_label="epsilon",
        attacks_to_plot=ATTACK_ORDER,
        x_getters=displacement_getters,
        y_getters=angle_getters,
    ))

    missing.extend(make_parametric_state_figure(
        records=n_step_records,
        output_dir=output_dir,
        figure_name="figure_9_delta_force_angle_vs_displacement_by_n_steps",
        title="Delta-force angle vs displacement by n_steps",
        x_label=r"Median displacement ($\AA$)",
        y_label="Median force-vector angle (deg)",
        bubble_label="n_steps",
        attacks_to_plot=STEP_ATTACK_ORDER,
        x_getters=displacement_getters,
        y_getters=angle_getters,
    ))

    convergence_getters = [
        lambda row: scalar_distribution(
            row,
            "before_relax_steps",
        ),
        lambda row: scalar_distribution(
            row,
            "after_relax_steps",
        ),
    ]

    make_paired_relaxation_figure(
        records=epsilon_records,
        output_dir=output_dir,
        figure_name=(
            "figure_8_1_convergence_vs_delta_force_angle_"
            "relaxation_by_epsilon"
        ),
        title=(
            "Relaxation change: convergence vs force-vector angle"
        ),
        x_label="Median force-vector angle (deg)",
        y_label="Relaxation steps",
        x_getters=angle_getters,
        y_getters=convergence_getters,
    )

    make_paired_relaxation_figure(
        records=epsilon_records,
        output_dir=output_dir,
        figure_name=(
            "figure_9_1_delta_force_angle_vs_displacement_"
            "relaxation_by_epsilon"
        ),
        title=(
            "Relaxation change: force-vector angle vs displacement"
        ),
        x_label=r"Median displacement ($\AA$)",
        y_label="Median force-vector angle (deg)",
        x_getters=displacement_getters,
        y_getters=angle_getters,
    )

    return missing


def make_topology_metric_figure_set(
    epsilon_records,
    n_step_records,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
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
            "Max coordination change",
        ),
    ]

    displacement_getters = displacement_metric_getters()
    delta_force_getters = delta_force_metric_getters()

    for column, label in metrics:
        rows = topology_stage_rows(column)
        metric_getters = topology_stage_getters(column)

        make_distribution_figure(
            records=epsilon_records,
            output_dir=output_dir,
            figure_name=f"{column}_by_epsilon",
            ylabel=label,
            rows=rows,
        )

        make_ci_figure(
            records=epsilon_records,
            output_dir=output_dir,
            figure_name=f"{column}_ci_by_epsilon",
            ylabel=f"Median {label}",
            rows=rows,
        )

        make_whisker_span_figure(
            records=epsilon_records,
            output_dir=output_dir,
            figure_name=f"{column}_whisker_span_by_epsilon",
            ylabel=f"{label} whisker span",
            rows=rows,
        )

        make_distribution_by_steps_figure(
            records=n_step_records,
            output_dir=output_dir,
            figure_name=f"{column}_by_n_steps",
            ylabel=label,
            rows=rows,
            epsilon=0.1,
        )

        make_ci_by_steps_figure(
            records=n_step_records,
            output_dir=output_dir,
            figure_name=f"{column}_ci_by_n_steps",
            ylabel=f"Median {label}",
            rows=rows,
            epsilon=0.1,
        )

        make_whisker_span_by_steps_figure(
            records=n_step_records,
            output_dir=output_dir,
            figure_name=f"{column}_whisker_span_by_n_steps",
            ylabel=f"{label} whisker span",
            rows=rows,
            epsilon=0.1,
        )

        plot_convergence_getters = [
            lambda row: scalar_distribution(
                row,
                "after_relax_steps",
            ),
            lambda row: scalar_distribution(
                row,
                "after_relax_steps",
            ),
        ]

        paired_convergence_getters = [
            lambda row: scalar_distribution(
                row,
                "before_relax_steps",
            ),
            lambda row: scalar_distribution(
                row,
                "after_relax_steps",
            ),
        ]

        for suffix, records, attacks, bubble_label in [
            (
                "epsilon",
                epsilon_records,
                ATTACK_ORDER,
                "epsilon",
            ),
            (
                "n_steps",
                n_step_records,
                STEP_ATTACK_ORDER,
                "n_steps",
            ),
        ]:
            make_parametric_state_figure(
                records=records,
                output_dir=output_dir,
                figure_name=(
                    f"convergence_vs_{column}_by_{suffix}"
                ),
                title=(
                    f"Convergence vs {label} by {bubble_label}"
                ),
                x_label=f"Median {label}",
                y_label="Relaxation steps",
                bubble_label=bubble_label,
                attacks_to_plot=attacks,
                x_getters=metric_getters,
                y_getters=plot_convergence_getters,
            )

            make_parametric_state_figure(
                records=records,
                output_dir=output_dir,
                figure_name=(
                    f"{column}_vs_displacement_by_{suffix}"
                ),
                title=(
                    f"{label} vs displacement by {bubble_label}"
                ),
                x_label=r"Median displacement ($\AA$)",
                y_label=f"Median {label}",
                bubble_label=bubble_label,
                attacks_to_plot=attacks,
                x_getters=displacement_getters,
                y_getters=metric_getters,
            )

            make_parametric_state_figure(
                records=records,
                output_dir=output_dir,
                figure_name=(
                    f"{column}_vs_delta_force_by_{suffix}"
                ),
                title=(
                    f"{label} vs delta force by {bubble_label}"
                ),
                x_label=r"Median $\Delta$ force (eV/$\AA$)",
                y_label=f"Median {label}",
                bubble_label=bubble_label,
                attacks_to_plot=attacks,
                x_getters=delta_force_getters,
                y_getters=metric_getters,
            )

            if suffix == "epsilon":
                make_paired_relaxation_figure(
                    records=records,
                    output_dir=output_dir,
                    figure_name=(
                        f"convergence_vs_{column}_"
                        "relaxation_by_epsilon"
                    ),
                    title=(
                        f"Relaxation change: convergence vs {label}"
                    ),
                    x_label=f"Median {label}",
                    y_label="Relaxation steps",
                    x_getters=metric_getters,
                    y_getters=paired_convergence_getters,
                )

                make_paired_relaxation_figure(
                    records=records,
                    output_dir=output_dir,
                    figure_name=(
                        f"{column}_vs_displacement_"
                        "relaxation_by_epsilon"
                    ),
                    title=(
                        f"Relaxation change: {label} vs displacement"
                    ),
                    x_label=r"Median displacement ($\AA$)",
                    y_label=f"Median {label}",
                    x_getters=displacement_getters,
                    y_getters=metric_getters,
                )

                make_paired_relaxation_figure(
                    records=records,
                    output_dir=output_dir,
                    figure_name=(
                        f"{column}_vs_delta_force_"
                        "relaxation_by_epsilon"
                    ),
                    title=(
                        f"Relaxation change: {label} vs delta force"
                    ),
                    x_label=r"Median $\Delta$ force (eV/$\AA$)",
                    y_label=f"Median {label}",
                    x_getters=delta_force_getters,
                    y_getters=metric_getters,
                    x_log=True,
                )


def make_topology_lattice_axis_component_figures(
    epsilon_records,
    output_dir,
):
    output_dir = Path(output_dir) / "components"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
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
            "Max coordination change",
        ),
    ]

    for column, label in metrics:
        rows = topology_stage_rows(column)

        make_distribution_figure(
            records=epsilon_records,
            output_dir=output_dir,
            figure_name=f"{column}_by_epsilon",
            ylabel=label,
            rows=rows,
            axis_specs=epsilon_component_axis_specs(
                epsilon_records,
                f"{column}_by_epsilon",
            ),
        )

        make_ci_figure(
            records=epsilon_records,
            output_dir=output_dir,
            figure_name=f"{column}_ci_by_epsilon",
            ylabel=f"Median {label}",
            rows=rows,
            axis_specs=epsilon_component_axis_specs(
                epsilon_records,
                f"{column}_ci_by_epsilon",
            ),
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
    make_topology_material_ranking(clean, output_dir)

    for material_slug, material_records in clean.groupby("material_slug"):
        material_output_dir = output_dir / str(material_slug)
        material_output_dir.mkdir(parents=True, exist_ok=True)

        save_topology_summary(material_records, material_output_dir)
        make_topology_by_attack_type(material_records, material_output_dir)

        material_epsilon_records = material_records[
            ~material_records["run_id"].str.contains(
                "_steps",
                regex=False,
                na=False,
            )
        ].copy()

        material_n_step_records = material_records[
            material_records["run_id"].str.contains(
                "_steps",
                regex=False,
                na=False,
            )
        ].copy()

        make_topology_metric_figure_set(
            material_epsilon_records,
            material_n_step_records,
            material_output_dir,
        )

        if not material_epsilon_records.empty:
            make_topology_lattice_axis_component_figures(
                material_epsilon_records,
                material_output_dir,
            )


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
    make_lattice_axis_component_figures(
        epsilon_records,
        args.output_dir / "components",
    )
    make_topology_metric_figure_set(epsilon_records, n_step_records, args.output_dir / "topology")
    make_topology_lattice_axis_component_figures(
        epsilon_records,
        args.output_dir / "topology",
    )
    make_outlier_reports(records, args.output_dir / "outliers")

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

    force_angle_missing = make_delta_force_angle_figure_set(
        epsilon_records,
        n_step_records,
        args.output_dir,
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

            if not material_epsilon_records.empty and not material_n_step_records.empty:
                make_delta_force_angle_figure_set(
                    material_epsilon_records,
                    material_n_step_records,
                    material_output_dir,
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
    missing_rows.extend(force_angle_missing)
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
