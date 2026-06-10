#!/usr/bin/env python3
from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    for calculator, color in CALCULATOR_COLORS.items():
        data = attack_records[
            (attack_records["calculator"] == calculator)
            & attack_records[step_col].notna()
        ].sort_values("epsilon")

        if data.empty:
            continue

        grouped = data.groupby("epsilon", as_index=False)[step_col].mean()

        ax.plot(
            grouped["epsilon"],
            grouped[step_col],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=calculator.upper(),
        )

        not_converged = data[data[conv_col] == False]
        if not not_converged.empty:
            ax.scatter(
                not_converged["epsilon"],
                not_converged[step_col],
                s=45,
                facecolors="none",
                edgecolors=color,
                linewidths=1.4,
                zorder=3,
            )

    ax.set_xscale("log")
    epsilons = sorted(attack_records["epsilon"].dropna().unique())
    ax.set_xticks(epsilons)
    ax.set_xticklabels([format_epsilon_label(epsilon) for epsilon in epsilons])
    style_epsilon_tick_labels(ax, rotate=len(epsilons) >= 6)
    ax.set_xlabel(r"$\epsilon$ ($\AA$)")
    ax.set_ylabel("steps until convergence")
    ax.grid(True, axis="y")
    ax.grid(True, axis="x", alpha=0.25)
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

        if position % 1 < 0.5:
            calculator = "mace"
        else:
            calculator = "uma"

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
                r"$\Delta$ force magnitude (eV/$\AA$)",
                lambda row: force_delta_values(row["run_dir"], "before_forces.csv", "perturbed_forces.csv"),
                f"{attack}: force change before relaxation",
            ),
            (
                "delta_force_after_perturb_after_relax",
                r"$\Delta$ force magnitude (eV/$\AA$)",
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

        not_converged = data[data[conv_col] == False]
        if not not_converged.empty:
            ax.scatter(
                not_converged["n_steps"],
                not_converged[step_col],
                s=45,
                facecolors="none",
                edgecolors=color,
                linewidths=1.4,
                zorder=3,
            )

    steps = sorted(attack_records["n_steps"].dropna().unique())
    ax.set_xscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels([str(int(step)) for step in steps])
    ax.tick_params(axis="x", labelrotation=35, pad=2)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    ax.set_xlabel("n_steps")
    ax.set_ylabel("steps until convergence")
    ax.grid(True, axis="y")
    ax.grid(True, axis="x", alpha=0.25)

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

        if position % 1 < 0.5:
            calculator = "mace"
        else:
            calculator = "uma"

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

    ax.set_xlabel("attack steps")
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
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        ylabel=r"$\Delta$ force magnitude (eV/$\AA$)",
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
        ylabel=r"whisker span of $\Delta$ force magnitude (eV/$\AA$)",
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

    make_convergence_by_steps_figure(n_step_records, args.output_dir, epsilon=0.1)

    force_by_steps_missing = make_distribution_by_steps_figure(
        records=n_step_records,
        output_dir=args.output_dir,
        figure_name="figure_5_delta_force_by_n_steps",
        ylabel=r"$\Delta$ force magnitude (eV/$\AA$)",
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
        ylabel=r"whisker span of $\Delta$ force magnitude (eV/$\AA$)",
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

    per_attack_missing = make_per_attack_figures(epsilon_records, args.output_dir)

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
                ylabel=r"$\Delta$ force magnitude (eV/$\AA$)",
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
                ylabel=r"$\Delta$ force magnitude (eV/$\AA$)",
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
    missing_rows.extend(per_attack_missing)

    pd.DataFrame(missing_rows).to_csv(
        args.output_dir / "missing_data_report.csv",
        index=False,
    )

    print(f"Saved comprehensive plots to {args.output_dir}")
    print(f"Saved combined dataset to {args.output_dir / 'combined_dataset.csv'}")
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


if __name__ == "__main__":
    main()
