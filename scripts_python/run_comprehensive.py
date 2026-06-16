#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import math
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")


def format_epsilon_label(value):
    return f"{float(value):g}"


def style_epsilon_tick_labels(ax, rotate=False):
    if rotate:
        ax.tick_params(axis="x", labelrotation=35, pad=2)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    else:
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


def finite_xy(data, x_col, y_col):
    clean = data[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    return clean[x_col].to_numpy(dtype=float), clean[y_col].to_numpy(dtype=float)


def add_std_ellipse(ax, x, y, color, n_std, label=None):
    if len(x) < 3 or len(y) < 3:
        return

    covariance = np.cov(x, y)
    if not np.isfinite(covariance).all():
        return

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0):
        return

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        xy=(float(np.mean(x)), float(np.mean(y))),
        width=float(width),
        height=float(height),
        angle=float(angle),
        fill=False,
        edgecolor=color,
        linewidth=1.0 if n_std == 1 else 0.8,
        linestyle="-" if n_std == 1 else "--",
        alpha=0.85 if n_std == 1 else 0.55,
        label=label,
    )
    ax.add_patch(ellipse)


def epsilon_bubble_sizes(values):
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return np.array([])

    minimum = float(values.min())
    maximum = float(values.max())

    if maximum <= minimum:
        return np.full(len(values), 70.0)

    scaled = (values.to_numpy(dtype=float) - minimum) / (maximum - minimum)
    return 35.0 + 115.0 * scaled


def convergence_displacement_rows(records, displacement_getter, step_col, missing_rows, figure_name):
    rows = []

    for _, row in records.iterrows():
        displacements, reason = displacement_getter(row)
        if displacements is None:
            missing_rows.append({
                "figure": figure_name,
                "run_id": row["run_id"],
                "reason": reason,
            })
            continue

        step_value = row.get(step_col)
        if step_value is None or pd.isna(step_value):
            missing_rows.append({
                "figure": figure_name,
                "run_id": row["run_id"],
                "reason": f"Missing {step_col}",
            })
            continue

        rows.append({
            "run_id": row["run_id"],
            "material_slug": row["material_slug"],
            "calculator": row["calculator"],
            "attack_label": row["attack_label"],
            "epsilon": row["epsilon"],
            "n_steps": row["n_steps"],
            "median_displacement_a": float(np.median(displacements)),
            "relax_steps": float(step_value),
        })

    return pd.DataFrame(rows)


def draw_convergence_displacement_panel(ax, data, attack, show_ylabel):
    subset = data[data["attack_label"] == attack].copy()

    if subset.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(attack)
        ax.set_xlabel(r"Median displacement ($\AA$)")
        if show_ylabel:
            ax.set_ylabel("Relaxation steps")
        ax.grid(True, alpha=0.35)
        return

    for calculator, color in CALCULATOR_COLORS.items():
        calc_data = subset[subset["calculator"] == calculator].copy()
        if calc_data.empty:
            continue

        sizes = epsilon_bubble_sizes(calc_data["epsilon"])
        ax.scatter(
            calc_data["median_displacement_a"],
            calc_data["relax_steps"],
            s=sizes,
            color=color,
            alpha=0.58,
            edgecolor="white",
            linewidth=0.45,
            label=calculator.upper(),
        )

        x, y = finite_xy(calc_data, "median_displacement_a", "relax_steps")
        add_std_ellipse(ax, x, y, color, n_std=1, label=f"{calculator.upper()} 1 std")
        add_std_ellipse(ax, x, y, color, n_std=2, label=f"{calculator.upper()} 2 std")

    ax.set_title(attack)
    ax.set_xlabel(r"Median displacement ($\AA$)")
    if show_ylabel:
        ax.set_ylabel("Relaxation steps")
    ax.grid(True, alpha=0.35)
    ax.margins(x=0.08, y=0.10)


def make_convergence_displacement_bubble_ellipse_figure(
    records,
    output_dir,
    figure_name,
    title,
    step_col,
    displacement_getter,
):
    missing_rows = []
    data = convergence_displacement_rows(
        records=records,
        displacement_getter=displacement_getter,
        step_col=step_col,
        missing_rows=missing_rows,
        figure_name=figure_name,
    )

    if data.empty:
        return missing_rows

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharex=False, sharey=False)

    for col_index, attack in enumerate(ATTACK_ORDER):
        draw_convergence_displacement_panel(
            axes[col_index],
            data,
            attack,
            show_ylabel=(col_index == 0),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 1.06),
            frameon=False,
        )

    label_axes(axes)
    fig.suptitle(title, y=1.12, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, output_dir / figure_name)
    plt.close(fig)

    return missing_rows


def metric_median(row, getter):
    values, reason = getter(row)
    if values is None:
        return None, reason
    return float(np.median(values)), None


def parametric_rows(records, x_getter, y_getter, missing_rows, figure_name):
    rows = []

    for _, row in records.iterrows():
        x_value, x_reason = x_getter(row)
        y_value, y_reason = y_getter(row)

        if x_value is None or y_value is None:
            missing_rows.append({
                "figure": figure_name,
                "run_id": row["run_id"],
                "reason": x_reason or y_reason,
            })
            continue

        rows.append({
            "run_id": row["run_id"],
            "calculator": row["calculator"],
            "attack_label": row["attack_label"],
            "epsilon": row["epsilon"],
            "x": x_value,
            "y": y_value,
        })

    return pd.DataFrame(rows)


def draw_parametric_panel(ax, data, attack, x_label, y_label, show_ylabel):
    subset = data[data["attack_label"] == attack].copy()

    if subset.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(attack)
        ax.set_xlabel(x_label)
        if show_ylabel:
            ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.35)
        return

    for calculator, color in CALCULATOR_COLORS.items():
        calc_data = subset[subset["calculator"] == calculator].copy()
        if calc_data.empty:
            continue

        ax.scatter(
            calc_data["x"],
            calc_data["y"],
            s=epsilon_bubble_sizes(calc_data["epsilon"]),
            color=color,
            alpha=0.58,
            edgecolor="white",
            linewidth=0.45,
            label=calculator.upper(),
        )

        x, y = finite_xy(calc_data, "x", "y")
        add_std_ellipse(ax, x, y, color, n_std=1, label=f"{calculator.upper()} 1 std")
        add_std_ellipse(ax, x, y, color, n_std=2, label=f"{calculator.upper()} 2 std")

    ax.set_title(attack)
    ax.set_xlabel(x_label)
    if show_ylabel:
        ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.35)
    ax.margins(x=0.08, y=0.10)


def make_parametric_bubble_ellipse_figure(
    records,
    output_dir,
    figure_name,
    title,
    x_label,
    y_label,
    x_getter,
    y_getter,
):
    missing_rows = []
    data = parametric_rows(records, x_getter, y_getter, missing_rows, figure_name)

    if data.empty:
        return missing_rows

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharex=False, sharey=False)

    for col_index, attack in enumerate(ATTACK_ORDER):
        draw_parametric_panel(
            axes[col_index],
            data,
            attack,
            x_label=x_label,
            y_label=y_label,
            show_ylabel=(col_index == 0),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 1.06),
            frameon=False,
        )

    label_axes(axes)
    fig.suptitle(title, y=1.12, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, output_dir / figure_name)
    plt.close(fig)

    return missing_rows


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
        whiskerprops={"color": "#444444", "linewidth": 0.9},
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
    ax.set_ylabel("steps until convergence")
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

    fig.legend(
        handles=model_legend_handles(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.tight_layout(rect=[0.03, 0.00, 1.00, 0.96])
    save_figure(fig, output_dir / "figure_1_convergence_by_epsilon")
    plt.close(fig)


def whisker_span(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    lower_values = values[values >= lower_limit]
    upper_values = values[values <= upper_limit]

    if len(lower_values) == 0 or len(upper_values) == 0:
        return None

    return float(upper_values.max() - lower_values.min())


def draw_grouped_whisker_span(ax, records, attack, value_getter, ylabel, missing_rows):
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

    series = {"mace": {"x": [], "y": []}, "uma": {"x": [], "y": []}}

    for position, box_values in zip(positions, values):
        span = whisker_span(box_values)
        if span is None:
            continue

        center_position = round(position)
        offset = position - center_position
        calculator = min(
            MODEL_OFFSETS,
            key=lambda name: abs(offset - MODEL_OFFSETS[name]),
        )

        x_value = round(position - MODEL_OFFSETS[calculator])
        series[calculator]["x"].append(x_value)
        series[calculator]["y"].append(span)

    for calculator, data in series.items():
        if not data["x"]:
            continue

        ax.plot(
            data["x"],
            data["y"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=CALCULATOR_COLORS[calculator],
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


def make_whisker_span_figure(records, output_dir, figure_name, ylabel, rows):
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2), sharex=False, sharey=False)

    all_missing = []
    panel_index = 0

    for row_index, (row_title, getter_factory) in enumerate(rows):
        for col_index, attack in enumerate(ATTACK_ORDER):
            ax = axes[row_index, col_index]
            attack_missing = []

            draw_grouped_whisker_span(
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

    fig.legend(
        handles=model_legend_handles(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.tight_layout(rect=[0.03, 0.00, 1.00, 0.96])
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

    fig.legend(
        handles=model_legend_handles(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.tight_layout(rect=[0.03, 0.00, 1.00, 0.96])
    save_figure(fig, output_dir / figure_name)
    plt.close(fig)

    return all_missing


def make_per_attack_figures(records, output_dir):
    missing_rows = []

    for attack in ATTACK_ORDER:
        attack_dir = output_dir / ATTACK_FOLDER[attack]
        attack_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        plot_convergence_panel(
            ax,
            records,
            attack,
            "after_relax_steps",
            "after_relax_converged",
        )
        ax.set_title(f"{attack}: convergence after perturbation")
        ax.legend(handles=model_legend_handles())
        fig.tight_layout()
        save_figure(fig, attack_dir / "convergence_steps_vs_epsilon")
        plt.close(fig)

        plots = [
            (
                "delta_force_after_perturb_before_relax",
                r"$\Delta$ force (eV/$\AA$)",
                lambda row: force_delta_values(row["run_dir"], "before_forces.csv", "perturbed_forces.csv"),
                f"{attack}: force change before relaxation",
            ),
            (
                "delta_force_after_perturb_after_relax",
                r"$\Delta$ force (eV/$\AA$)",
                lambda row: force_delta_values(row["run_dir"], "before_forces.csv", "after_forces.csv"),
                f"{attack}: force change after relaxation",
            ),
            (
                "displacement_after_perturb_before_relax",
                r"atomic displacement ($\AA$)",
                lambda row: displacement_values(row["run_dir"], "before_forces.csv", "perturbed_forces.csv"),
                f"{attack}: displacement before relaxation",
            ),
            (
                "displacement_after_perturb_after_relax",
                r"atomic displacement ($\AA$)",
                lambda row: displacement_values(row["run_dir"], "before_forces.csv", "after_forces.csv"),
                f"{attack}: displacement after relaxation",
            ),
        ]

        for filename, ylabel, getter, title in plots:
            fig, ax = plt.subplots(figsize=(5.4, 3.5))
            attack_missing = []
            draw_grouped_boxplot(
                ax=ax,
                records=records,
                attack=attack,
                value_getter=getter,
                ylabel=ylabel,
                missing_rows=attack_missing,
            )
            ax.set_title(title)
            fig.tight_layout()
            save_figure(fig, attack_dir / filename)
            plt.close(fig)
            missing_rows.extend(attack_missing)

        pd.DataFrame(missing_rows).to_csv(
            attack_dir / "missing_data_report.csv",
            index=False,
        )

    return missing_rows


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
        whiskerprops={"color": "#444444", "linewidth": 0.9},
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
    ax.set_ylabel("steps until convergence")
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

    fig.suptitle(rf"Fixed $\epsilon$ = {epsilon:g} $\AA$", y=1.02, fontsize=9)
    fig.legend(
        handles=model_legend_handles(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.00),
    )

    fig.tight_layout(rect=[0.05, 0.00, 1.00, 0.94])
    save_figure(fig, output_dir / "figure_4_convergence_by_n_steps")
    plt.close(fig)


def draw_grouped_whisker_span_by_steps(ax, records, attack, epsilon, value_getter, ylabel, missing_rows):
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

    series = {"mace": {"x": [], "y": []}, "uma": {"x": [], "y": []}}

    for position, box_values in zip(positions, values):
        span = whisker_span(box_values)
        if span is None:
            continue

        center_position = round(position)
        offset = position - center_position
        calculator = min(
            MODEL_OFFSETS,
            key=lambda name: abs(offset - MODEL_OFFSETS[name]),
        )

        x_value = round(position - MODEL_OFFSETS[calculator])
        series[calculator]["x"].append(x_value)
        series[calculator]["y"].append(span)

    for calculator, data in series.items():
        if not data["x"]:
            continue

        ax.plot(
            data["x"],
            data["y"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=CALCULATOR_COLORS[calculator],
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


def make_whisker_span_by_steps_figure(records, output_dir, figure_name, ylabel, rows, epsilon=0.1):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=False, sharey=False)

    all_missing = []
    panel_index = 0

    for row_index, (row_title, getter_factory) in enumerate(rows):
        for col_index, attack in enumerate(STEP_ATTACK_ORDER):
            ax = axes[row_index, col_index]
            attack_missing = []

            draw_grouped_whisker_span_by_steps(
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

    fig.legend(
        handles=model_legend_handles(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.00),
    )

    fig.tight_layout(rect=[0.05, 0.00, 1.00, 0.94])
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

    fig.suptitle(rf"Fixed $\epsilon$ = {epsilon:g} $\AA$", y=1.02, fontsize=9)
    fig.legend(
        handles=model_legend_handles(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.00),
    )

    fig.tight_layout(rect=[0.05, 0.00, 1.00, 0.94])
    save_figure(fig, output_dir / figure_name)
    plt.close(fig)

    return all_missing


def main():
    apply_plot_style()

    parser = argparse.ArgumentParser(
        description="Create publication-quality comprehensive MACE vs UMA plots."
    )
    parser.add_argument("--mace-dir", default=BASE_DIR / "outputs_mace", type=Path)
    parser.add_argument("--uma-dir", default=BASE_DIR / "outputs_uma", type=Path)
    parser.add_argument("--output-dir", default=BASE_DIR / "comprehensive_outputs", type=Path)
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

    force_whisker_missing = make_whisker_span_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_2_delta_force_whisker_span_by_epsilon",
        ylabel=r"whisker span of $\Delta$ force (eV/$\AA$)",
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
        ylabel=r"atomic displacement ($\AA$)",
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

    displacement_whisker_missing = make_whisker_span_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_3_displacement_whisker_span_by_epsilon",
        ylabel=r"whisker span of atomic displacement ($\AA$)",
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

    convergence_displacement_before_missing = make_convergence_displacement_bubble_ellipse_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_7_convergence_vs_displacement_before_attack_bubble_ellipse",
        title="Relaxation before attack: convergence vs displacement",
        step_col="before_relax_steps",
        displacement_getter=lambda row: displacement_values(
            row["run_dir"],
            "before_forces.csv",
            "perturbed_forces.csv",
        ),
    )

    convergence_displacement_after_missing = make_convergence_displacement_bubble_ellipse_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_7_convergence_vs_displacement_after_attack_bubble_ellipse",
        title="Relaxation after attack: convergence vs displacement",
        step_col="after_relax_steps",
        displacement_getter=lambda row: displacement_values(
            row["run_dir"],
            "before_forces.csv",
            "after_forces.csv",
        ),
    )

    convergence_force_before_missing = make_parametric_bubble_ellipse_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_8_convergence_vs_delta_force_before_attack_bubble_ellipse",
        title="Relaxation before attack: convergence vs delta force",
        x_label=r"Median $\Delta$ force (eV/$\AA$)",
        y_label="Relaxation steps",
        x_getter=lambda row: metric_median(row, lambda item: force_delta_values(
            item["run_dir"],
            "before_forces.csv",
            "perturbed_forces.csv",
        )),
        y_getter=lambda row: (row["before_relax_steps"], None)
            if pd.notna(row["before_relax_steps"])
            else (None, "Missing before_relax_steps"),
    )

    convergence_force_after_missing = make_parametric_bubble_ellipse_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_8_convergence_vs_delta_force_after_attack_bubble_ellipse",
        title="Relaxation after attack: convergence vs delta force",
        x_label=r"Median $\Delta$ force (eV/$\AA$)",
        y_label="Relaxation steps",
        x_getter=lambda row: metric_median(row, lambda item: force_delta_values(
            item["run_dir"],
            "before_forces.csv",
            "after_forces.csv",
        )),
        y_getter=lambda row: (row["after_relax_steps"], None)
            if pd.notna(row["after_relax_steps"])
            else (None, "Missing after_relax_steps"),
    )

    force_displacement_before_missing = make_parametric_bubble_ellipse_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_9_delta_force_vs_displacement_before_attack_bubble_ellipse",
        title="Relaxation before attack: delta force vs displacement",
        x_label=r"Median displacement ($\AA$)",
        y_label=r"Median $\Delta$ force (eV/$\AA$)",
        x_getter=lambda row: metric_median(row, lambda item: displacement_values(
            item["run_dir"],
            "before_forces.csv",
            "perturbed_forces.csv",
        )),
        y_getter=lambda row: metric_median(row, lambda item: force_delta_values(
            item["run_dir"],
            "before_forces.csv",
            "perturbed_forces.csv",
        )),
    )

    force_displacement_after_missing = make_parametric_bubble_ellipse_figure(
        records=epsilon_records,
        output_dir=args.output_dir,
        figure_name="figure_9_delta_force_vs_displacement_after_attack_bubble_ellipse",
        title="Relaxation after attack: delta force vs displacement",
        x_label=r"Median displacement ($\AA$)",
        y_label=r"Median $\Delta$ force (eV/$\AA$)",
        x_getter=lambda row: metric_median(row, lambda item: displacement_values(
            item["run_dir"],
            "before_forces.csv",
            "after_forces.csv",
        )),
        y_getter=lambda row: metric_median(row, lambda item: force_delta_values(
            item["run_dir"],
            "before_forces.csv",
            "after_forces.csv",
        )),
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

    force_by_steps_whisker_missing = make_whisker_span_by_steps_figure(
        records=n_step_records,
        output_dir=args.output_dir,
        figure_name="figure_5_delta_force_whisker_span_by_n_steps",
        ylabel=r"whisker span of $\Delta$ force (eV/$\AA$)",
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
        ylabel=r"atomic displacement ($\AA$)",
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

    displacement_by_steps_whisker_missing = make_whisker_span_by_steps_figure(
        records=n_step_records,
        output_dir=args.output_dir,
        figure_name="figure_6_displacement_whisker_span_by_n_steps",
        ylabel=r"whisker span of atomic displacement ($\AA$)",
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

            make_distribution_figure(
                records=material_epsilon_records,
                output_dir=material_output_dir,
                figure_name="figure_3_displacement_by_epsilon",
                ylabel=r"atomic displacement ($\AA$)",
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

        if not material_n_step_records.empty:
            make_convergence_by_steps_figure(material_n_step_records, material_output_dir, epsilon=0.1)

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

            make_distribution_by_steps_figure(
                records=material_n_step_records,
                output_dir=material_output_dir,
                figure_name="figure_6_displacement_by_n_steps",
                ylabel=r"atomic displacement ($\AA$)",
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

    missing_rows.extend(force_missing)
    missing_rows.extend(force_whisker_missing)
    missing_rows.extend(displacement_missing)
    missing_rows.extend(displacement_whisker_missing)
    missing_rows.extend(force_by_steps_missing)
    missing_rows.extend(force_by_steps_whisker_missing)
    missing_rows.extend(displacement_by_steps_missing)
    missing_rows.extend(displacement_by_steps_whisker_missing)
    missing_rows.extend(convergence_displacement_before_missing)
    missing_rows.extend(convergence_displacement_after_missing)
    missing_rows.extend(convergence_force_before_missing)
    missing_rows.extend(convergence_force_after_missing)
    missing_rows.extend(force_displacement_before_missing)
    missing_rows.extend(force_displacement_after_missing)

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
    print(f"  {args.output_dir / 'figure_3_displacement_by_epsilon.png'}")
    print(f"  {args.output_dir / 'figure_3_displacement_whisker_span_by_epsilon.png'}")
    print(f"  {args.output_dir / 'figure_4_convergence_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_5_delta_force_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_5_delta_force_whisker_span_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_6_displacement_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_6_displacement_whisker_span_by_n_steps.png'}")
    print(f"  {args.output_dir / 'figure_7_convergence_vs_displacement_before_attack_bubble_ellipse.png'}")
    print(f"  {args.output_dir / 'figure_7_convergence_vs_displacement_after_attack_bubble_ellipse.png'}")
    print(f"  {args.output_dir / 'figure_8_convergence_vs_delta_force_before_attack_bubble_ellipse.png'}")
    print(f"  {args.output_dir / 'figure_8_convergence_vs_delta_force_after_attack_bubble_ellipse.png'}")
    print(f"  {args.output_dir / 'figure_9_delta_force_vs_displacement_before_attack_bubble_ellipse.png'}")
    print(f"  {args.output_dir / 'figure_9_delta_force_vs_displacement_after_attack_bubble_ellipse.png'}")

if __name__ == "__main__":
    main()
