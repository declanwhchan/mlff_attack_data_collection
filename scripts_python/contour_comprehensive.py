#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

BETA_COLORS = {
    0.10: "#E69F00",  # orange
    0.05: "#009E73",  # green
    0.00: "#7B3294",  # purple
}

CALC_COLORS = {
    "mace": "#0072B2",
    "uma": "#D55E00",
}

ATTACK_ORDER = ["FGSM", "I-FGSM", "PGD"]


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
    if data is None:
        return pd.DataFrame()
    if "status" not in data.columns:
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
            values.extend(metrics[metric].replace([np.inf, -np.inf], np.nan).dropna().tolist())
    return np.asarray(values, dtype=float)


def contour_stats(rows, metric):
    values = contour_metric_values(rows, metric)
    if values.size == 0:
        return None
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def add_contour_reference(ax, stats, ylabel_kind):
    if stats is None:
        return

    lower = max(0.0, stats["p05"])
    upper = stats["p95"]

    ax.axhspan(
        lower,
        upper,
        color="#9E9E9E",
        alpha=0.20,
        linewidth=0,
        label="contour p05-p95",
        zorder=0,
    )
    ax.axhline(
        stats["median"],
        color="#555555",
        linestyle="--",
        linewidth=1.1,
        label="contour median",
        zorder=1,
    )
    ax.text(
        0.02,
        0.94,
        f"contour median={stats['median']:.3g}\ncontour p95={stats['p95']:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#333333",
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
        disp = attack_displacement(row, "before_forces.csv", "perturbed_forces.csv")
        force = attack_force_delta(row, "before_forces.csv", "perturbed_forces.csv")

        run_id = str(row.get("run_id", ""))
        is_step_sweep = "_steps" in run_id

        rows.append({
            "material_slug": row.get("material_slug"),
            "calculator": row.get("calculator"),
            "attack_label": attack_label_from_row(row),
            "epsilon": float(row["epsilon"]) if pd.notna(row.get("epsilon")) else np.nan,
            "n_steps": int(float(row["n_steps"])) if pd.notna(row.get("n_steps")) else np.nan,
            "is_step_sweep": is_step_sweep,
            "attack_median_displacement_a": float(np.median(disp)) if disp.size else np.nan,
            "attack_p95_displacement_a": float(np.percentile(disp, 95)) if disp.size else np.nan,
            "attack_median_force_delta_ev_a": float(np.median(force)) if force.size else np.nan,
            "attack_p95_force_delta_ev_a": float(np.percentile(force, 95)) if force.size else np.nan,
        })

    return pd.DataFrame(rows)


def draw_attack_panels(fig, axes, data, x_col, x_label, calculator, contour_rows, title):
    disp_stats = contour_stats(contour_rows, "mean_displacement_from_initial_a")
    force_stats = contour_stats(contour_rows, "mean_force_delta_from_initial_ev_a")
    color = CALC_COLORS.get(calculator, "#333333")

    for col, attack in enumerate(ATTACK_ORDER):
        subset = data[data["attack_label"] == attack].copy()
        subset = subset[np.isfinite(subset[x_col])]
        subset = subset.sort_values(x_col)

        ax_disp = axes[0, col]
        ax_force = axes[1, col]

        add_contour_reference(ax_disp, disp_stats, "displacement")
        add_contour_reference(ax_force, force_stats, "force")

        if not subset.empty:
            grouped = subset.groupby(x_col, as_index=False).agg({
                "attack_median_displacement_a": "median",
                "attack_p95_displacement_a": "median",
                "attack_median_force_delta_ev_a": "median",
                "attack_p95_force_delta_ev_a": "median",
            })

            ax_disp.plot(
                grouped[x_col],
                grouped["attack_median_displacement_a"],
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=color,
                label="attack median",
                zorder=3,
            )
            ax_disp.plot(
                grouped[x_col],
                grouped["attack_p95_displacement_a"],
                marker="^",
                markersize=3.5,
                linewidth=1.0,
                linestyle=":",
                color=color,
                alpha=0.75,
                label="attack p95",
                zorder=2,
            )

            ax_force.plot(
                grouped[x_col],
                grouped["attack_median_force_delta_ev_a"],
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=color,
                label="attack median",
                zorder=3,
            )
            ax_force.plot(
                grouped[x_col],
                grouped["attack_p95_force_delta_ev_a"],
                marker="^",
                markersize=3.5,
                linewidth=1.0,
                linestyle=":",
                color=color,
                alpha=0.75,
                label="attack p95",
                zorder=2,
            )

        ax_disp.set_title(attack)
        ax_disp.set_ylabel(r"Displacement ($\AA$)" if col == 0 else "")
        ax_force.set_ylabel(r"Force delta (eV/$\AA$)" if col == 0 else "")
        ax_force.set_xlabel(x_label)

        if x_col == "epsilon":
            ax_disp.set_xscale("log")
            ax_force.set_xscale("log")

        for ax in [ax_disp, ax_force]:
            ax.grid(True, axis="y")
            ax.margins(x=0.04)
            if subset.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))

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
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(material_dir / f"{calculator}_contour_vs_attack_by_epsilon.png", bbox_inches="tight")
        plt.close(fig)

    step_data = table[table["is_step_sweep"]].copy()
    if not step_data.empty:
        fig, axes = plt.subplots(2, 3, figsize=(8.8, 4.8), sharex=False, sharey="row")
        draw_attack_panels(
            fig=fig,
            axes=axes,
            data=step_data,
            x_col="n_steps",
            x_label="attack steps",
            calculator=calculator,
            contour_rows=contour_rows,
            title=f"{material_slug} {calculator.upper()}: attacks vs contour baseline by n_steps",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(material_dir / f"{calculator}_contour_vs_attack_by_n_steps.png", bbox_inches="tight")
        plt.close(fig)

    disp_stats = contour_stats(contour_rows, "mean_displacement_from_initial_a")
    force_stats = contour_stats(contour_rows, "mean_force_delta_from_initial_ev_a")
    energy_stats = contour_stats(contour_rows, "energy_deviation_mev_per_atom")

    table["contour_displacement_median_a"] = np.nan if disp_stats is None else disp_stats["median"]
    table["contour_displacement_p95_a"] = np.nan if disp_stats is None else disp_stats["p95"]
    table["contour_force_delta_median_ev_a"] = np.nan if force_stats is None else force_stats["median"]
    table["contour_force_delta_p95_ev_a"] = np.nan if force_stats is None else force_stats["p95"]
    table["contour_abs_energy_deviation_p95_mev_atom"] = (
        np.nan if energy_stats is None else abs(energy_stats["p95"])
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

    axes[0, 0].axhline(0, color="#222222", lw=0.8, alpha=0.65)
    axes[0, 1].legend(loc="upper right")
    axes[0, 2].legend(loc="upper right")
    axes[1, 2].legend(loc="upper right")

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
    fig.suptitle(f"{material_slug}: MACE vs UMA contour exploration", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    material_dir = output_dir / material_slug
    material_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(material_dir / "mace_vs_uma_contour.png", bbox_inches="tight")
    plt.close(fig)


def plot_global(records, output_dir):
    if records.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    usable = records[
        records["contour_displacement_p95_a"].notna()
        & records["attack_median_displacement_a"].notna()
    ].copy()

    if usable.empty:
        return

    fig, ax = plt.subplots(figsize=(5.8, 4.4))

    for calculator, color in CALC_COLORS.items():
        subset = usable[usable["calculator"] == calculator]
        if subset.empty:
            continue
        ax.scatter(
            subset["contour_displacement_p95_a"],
            subset["attack_median_displacement_a"],
            s=26,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            label=calculator.upper(),
        )

    max_value = float(np.nanmax([
        usable["contour_displacement_p95_a"].max(),
        usable["attack_median_displacement_a"].max(),
    ]))
    ax.plot([0, max_value], [0, max_value], color="#555555", lw=1.0, linestyle="--", label="1:1")

    ax.set_xlabel(r"Contour p95 displacement ($\AA$)")
    ax.set_ylabel(r"Attack median displacement ($\AA$)")
    ax.set_title("Attack displacement relative to contour baseline")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_dir / "global_contour_vs_attack_displacement.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mace-contour-dir", default=BASE_DIR / "outputs_mace" / "contour", type=Path)
    parser.add_argument("--uma-contour-dir", default=BASE_DIR / "outputs_uma" / "contour", type=Path)
    parser.add_argument("--comprehensive-dir", default=BASE_DIR / "comprehensive_outputs", type=Path)
    parser.add_argument("--output-dir", default=BASE_DIR / "comprehensive_outputs" / "contour", type=Path)
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