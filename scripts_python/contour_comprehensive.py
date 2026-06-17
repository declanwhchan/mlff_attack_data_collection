#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        "legend.fontsize": 7,
        "legend.frameon": False,
        "savefig.dpi": 600,
    })


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


def contour_stats(rows, metric):
    values = contour_metric_values(rows, metric)
    if values.size == 0:
        return None
    return {
        "p05": float(np.percentile(values, 5)),
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
        after_attack_before_relaxation_disp = attack_displacement(
            row,
            "before_forces.csv",
            "perturbed_forces.csv",
        )
        after_attack_before_relaxation_force = attack_force_delta(
            row,
            "before_forces.csv",
            "perturbed_forces.csv",
        )

        after_attack_after_relaxation_disp = attack_displacement(
            row,
            "before_forces.csv",
            "after_forces.csv",
        )
        after_attack_after_relaxation_force = attack_force_delta(
            row,
            "before_forces.csv",
            "after_forces.csv",
        )

        run_id = str(row.get("run_id", ""))
        is_step_sweep = "_steps" in run_id

        after_attack_before_relaxation_median_displacement = (
            float(np.median(after_attack_before_relaxation_disp))
            if after_attack_before_relaxation_disp.size
            else np.nan
        )
        after_attack_before_relaxation_median_force_delta = (
            float(np.median(after_attack_before_relaxation_force))
            if after_attack_before_relaxation_force.size
            else np.nan
        )

        rows.append({
            "material_slug": row.get("material_slug"),
            "calculator": row.get("calculator"),
            "attack_label": attack_label_from_row(row),
            "epsilon": float(row["epsilon"]) if pd.notna(row.get("epsilon")) else np.nan,
            "n_steps": int(float(row["n_steps"])) if pd.notna(row.get("n_steps")) else np.nan,
            "is_step_sweep": is_step_sweep,

            "after_attack_before_relaxation_median_displacement_a": after_attack_before_relaxation_median_displacement,
            "after_attack_before_relaxation_median_force_delta_ev_a": after_attack_before_relaxation_median_force_delta,
            "after_attack_after_relaxation_median_displacement_a": (
                float(np.median(after_attack_after_relaxation_disp))
                if after_attack_after_relaxation_disp.size
                else np.nan
            ),
            "after_attack_after_relaxation_median_force_delta_ev_a": (
                float(np.median(after_attack_after_relaxation_force))
                if after_attack_after_relaxation_force.size
                else np.nan
            ),

            # Keep existing per-material plots unchanged.
            "attack_median_displacement_a": after_attack_before_relaxation_median_displacement,
            "attack_median_force_delta_ev_a": after_attack_before_relaxation_median_force_delta,
        })

    return pd.DataFrame(rows)


def draw_attack_panels(fig, axes, data, x_col, x_label, calculator, contour_rows, title, attacks_to_plot):
    disp_stats = contour_stats(contour_rows, "mean_displacement_from_initial_a")
    force_stats = contour_stats(contour_rows, "mean_force_delta_from_initial_ev_a")
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

        if x_col == "epsilon":
            ax_disp.set_xscale("log")
            ax_force.set_xscale("log")

        for ax in [ax_disp, ax_force]:
            ax.grid(True, axis="y")
            ax.margins(x=0.04)
            if subset.empty:
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


def plot_contour_vs_attack(material_slug, calculator, contour_rows, attacks, output_dir):
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
            contour_rows=contour_rows,
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
            x_label="attack steps",
            calculator=calculator,
            contour_rows=contour_rows,
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

    disp_stats = contour_stats(contour_rows, "mean_displacement_from_initial_a")
    force_stats = contour_stats(contour_rows, "mean_force_delta_from_initial_ev_a")
    energy_values = contour_metric_values(contour_rows, "energy_deviation_mev_per_atom")

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


def axis_limit(values, pad=0.06):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values >= 0]

    if values.size == 0:
        return 1.0

    upper = float(values.max())
    if upper <= 0:
        return 1.0

    return upper * (1.0 + pad)


def plot_one_global_panel(ax, data, x_col, y_col, xlabel, ylabel, title):
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
            s=26,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            label=calculator.upper(),
        )

    x_max = axis_limit(data[x_col])
    y_max = axis_limit(data[y_col])

    line_max = min(x_max, y_max)
    ax.plot(
        [0, line_max],
        [0, line_max],
        color="#555555",
        lw=1.0,
        linestyle="--",
        label="1:1",
    )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)


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

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))

    plot_one_global_panel(
        ax=axes[0],
        data=displacement,
        x_col="contour_displacement_p95_a",
        y_col=displacement_col,
        xlabel=r"Contour p95 displacement ($\AA$)",
        ylabel=r"Median displacement ($\AA$)",
        title="Displacement",
    )

    plot_one_global_panel(
        ax=axes[1],
        data=force,
        x_col="contour_force_delta_p95_ev_a",
        y_col=force_col,
        xlabel=r"Contour p95 $\Delta$ force (eV/$\AA$)",
        ylabel=r"Median $\Delta$ force (eV/$\AA$)",
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
            s=24,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            label=calculator.upper(),
        )

    x_max = axis_limit(subset[x_col])
    y_max = axis_limit(subset[y_col])
    line_max = min(x_max, y_max)

    ax.plot(
        [0, line_max],
        [0, line_max],
        color="#555555",
        lw=1.0,
        linestyle="--",
        label="1:1",
    )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_title(attack_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if row_label else "")
    ax.grid(True, alpha=0.35)


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

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), sharex=False, sharey=False)

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
        xlabel=r"Contour p95 displacement ($\AA$)",
        ylabel=r"Median displacement ($\AA$)",
        title="Relaxation vs contour exploration by attack type: displacement",
        output_name="global_relaxation_vs_contour_exploration_by_attack_type_displacement.png",
    )

    plot_global_relaxation_attack_grid(
        records=records,
        output_dir=output_dir,
        x_col="contour_force_delta_p95_ev_a",
        before_col="after_attack_before_relaxation_median_force_delta_ev_a",
        after_col="after_attack_after_relaxation_median_force_delta_ev_a",
        xlabel=r"Contour p95 $\Delta$ force (eV/$\AA$)",
        ylabel=r"Median $\Delta$ force (eV/$\AA$)",
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
    all_rows.to_csv(args.output_dir / "contour_summary_combined.csv", index=False)

    attacks = load_attack_dataset(args.comprehensive_dir)
    comparison_tables = []

    for (material_slug, calculator), rows in all_rows.groupby(["material_slug", "calculator"]):
        plot_six_panel(material_slug, calculator, rows, args.output_dir)

        if not attacks.empty:
            table = plot_contour_vs_attack(material_slug, calculator, rows, attacks, args.output_dir)
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