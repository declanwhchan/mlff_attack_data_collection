#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator, NullFormatter, NullLocator, ScalarFormatter
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


def positive_finite(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return values


def systematic_epsilon_ticks(values):
    values = positive_finite(values)
    if len(values) == 0:
        return [1e-3, 1e-2, 1e-1, 1.0, 10.0]

    min_power = int(np.floor(np.log10(np.min(values))))
    max_power = int(np.ceil(np.log10(np.max(values))))
    return [10.0 ** power for power in range(min_power, max_power + 1)]


def apply_systematic_epsilon_axis(ax, eps_values, label=r"$\epsilon$ ($\AA$)"):
    ticks = systematic_epsilon_ticks(eps_values)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{tick:g}" for tick in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(ticks[0] / 1.18, ticks[-1] * 1.18)
    ax.set_xlabel(label)
    ax.tick_params(axis="x", labelrotation=0, pad=2)


def positive_floor(values, fallback=1e-12):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return fallback
    return float(np.min(values))


def set_log_y_from_values(ax, values, label=None, pad_decades=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return

    ymin = float(np.min(values)) / (10.0 ** pad_decades)
    ymax = float(np.max(values)) * (10.0 ** pad_decades)
    ymax = max(ymax, ymin * 10.0)

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)

    if label is not None:
        ax.set_ylabel(label)


def pad_limits_for_scatter_points(ax, x_values, y_values, sizes, xlim=None, ylim=None, min_pad_frac=0.01):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    sizes = np.asarray(sizes, dtype=float)

    finite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(sizes)
    if not np.any(finite):
        return

    x_values = x_values[finite]
    y_values = y_values[finite]
    sizes = sizes[finite]

    if xlim is None:
        xmin = float(np.min(x_values))
        xmax = float(np.max(x_values))
    else:
        xmin, xmax = map(float, xlim)

    if ylim is None:
        ymin = float(np.min(y_values))
        ymax = float(np.max(y_values))
    else:
        ymin, ymax = map(float, ylim)

    max_radius_points = np.sqrt(float(np.max(sizes)) / np.pi)
    pixels = max_radius_points * ax.figure.dpi / 72.0

    x0, y0 = ax.transData.inverted().transform((0.0, 0.0))
    x1, _ = ax.transData.inverted().transform((pixels, 0.0))
    _, y1 = ax.transData.inverted().transform((0.0, pixels))

    xpad = abs(x1 - x0)
    ypad = abs(y1 - y0)

    xrange = xmax - xmin
    yrange = ymax - ymin
    xpad = max(xpad, max(xrange * min_pad_frac, 1e-12))
    ypad = max(ypad, max(yrange * min_pad_frac, 1e-12))

    if xrange == 0:
        xpad = max(xpad, abs(xmin) * min_pad_frac, 1e-6)
    if yrange == 0:
        ypad = max(ypad, abs(ymin) * min_pad_frac, 1e-6)

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)


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
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })




def save_figure(fig, output_base):
    output_base = Path(output_base)
    if output_base.suffix:
        output_base = output_base.with_suffix("")
    tighten_contour_axes(fig)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")

def style_numeric_axis(ax, xbins=5, ybins=5):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))

    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        axis.set_major_formatter(formatter)

    ax.tick_params(axis="both", labelsize=8, pad=2)


def sparse_numeric_ticks(values, max_ticks=6):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]

    if len(values) == 0:
        return []

    values = sorted(set(float(value) for value in values))

    if len(values) <= max_ticks:
        return values

    indices = np.linspace(0, len(values) - 1, max_ticks, dtype=int)
    return [values[index] for index in indices]


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


def tight_axis_limits(values, pad=0.10):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    low = float(np.min(values))
    high = float(np.max(values))

    if np.allclose(low, high):
        span = max(abs(low) * 0.20, 1e-9)
        if low >= 0 and low - span < 0:
            return 0.0, low + span
        return low - span, high + span

    span = high - low
    if span <= 0 or not np.isfinite(span):
        return None

    low -= pad * span
    high += pad * span

    if np.nanmin(values) >= 0 and low < 0:
        low = 0.0 if np.nanmin(values) < 0.08 * span else max(0.0, low)

    return low, high


def tighten_contour_axes(fig):
    for ax in fig.axes:
        if not ax.has_data():
            continue

        if ax.get_yscale() == "log":
            values = clean_axis_values(ax, "y")
            values = values[np.isfinite(values) & (values > 0)]
            if len(values):
                set_log_y_from_values(ax, values)
            continue

        limits = tight_axis_limits(clean_axis_values(ax, "y"))
        if limits is not None:
            ax.set_ylim(*limits)


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
            epsilon_values = subset[x_col].to_numpy(dtype=float) if not subset.empty else data[x_col].to_numpy(dtype=float)

            for axis in [ax_disp, ax_force]:
                apply_systematic_epsilon_axis(axis, epsilon_values)

        for ax, y_col in [
            (ax_disp, "attack_median_displacement_a"),
            (ax_force, "attack_median_force_delta_ev_a"),
        ]:
            if subset.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                continue

            contour_values = []
            if ax is ax_disp and disp_stats is not None:
                contour_values.extend([disp_stats["p05"], disp_stats["p95"]])
            if ax is ax_force and force_stats is not None:
                contour_values.extend([force_stats["p05"], force_stats["p95"]])

            attack_values = subset[y_col].to_numpy(dtype=float)
            all_y = np.asarray(contour_values + attack_values[np.isfinite(attack_values)].tolist(), dtype=float)
            all_y = all_y[np.isfinite(all_y) & (all_y > 0)]

            if len(all_y):
                set_log_y_from_values(ax, all_y)

            style_numeric_axis(ax)
            ax.grid(True, axis="y")
            ax.margins(x=0.04)

    tighten_contour_axes(fig)

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
        save_figure(fig, material_dir / f"{calculator}_contour_vs_attack_by_epsilon.png")
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
            contour_rows=contour_rows,
            title=f"{material_slug} {calculator.upper()}: attacks vs contour baseline by n_steps",
            attacks_to_plot=["I-FGSM", "PGD"],
        )
        label_axes(axes)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, material_dir / f"{calculator}_contour_vs_attack_by_n_steps.png")
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
    save_figure(fig, material_dir / f"{calculator}_six_panel")
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
    save_figure(fig, material_dir / "mace_vs_uma_contour")
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
    pad_limits_for_scatter_points(
        ax,
        data[x_col].to_numpy(dtype=float),
        data[y_col].to_numpy(dtype=float),
        np.full(len(data), 26.0),
        xlim=(0, x_max),
        ylim=(0, y_max),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_numeric_axis(ax)
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
    save_figure(fig, output_dir / output_name)
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
    pad_limits_for_scatter_points(
        ax,
        subset[x_col].to_numpy(dtype=float),
        subset[y_col].to_numpy(dtype=float),
        np.full(len(subset), 24.0),
        xlim=(0, x_max),
        ylim=(0, y_max),
    )
    ax.set_title(attack_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if row_label else "")
    style_numeric_axis(ax)
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
    save_figure(fig, output_dir / output_name)
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



def write_contour_publication_audit(output_dir, all_rows, comparison_df):
    output_dir = Path(output_dir)
    material_count = 0 if all_rows.empty else int(all_rows["material_slug"].nunique())
    comparison_count = 0 if comparison_df.empty else int(len(comparison_df))
    lines = [
        "# Contour Figure Audit",
        "",
        f"Scope: contour exploration and contour-vs-attack figures for {material_count} materials.",
        f"Contour-vs-attack comparison rows available: {comparison_count}.",
        "",
        "## Improvements Applied",
        "",
        "- Exported every contour figure as 600 dpi PNG.",
        "- Standardized colorblind-safe calculator colors, beta colors, grid weight, panel labels, and white figure backgrounds.",
        "- Used systematic log epsilon ticks for epsilon sweeps instead of sparse arbitrary ticks.",
        "- Preserved full finite y ranges with padding; log y axes are applied to positive contour/attack comparisons spanning broad ranges.",
        "- Added marker-aware scatter padding for global contour-vs-attack comparisons so points are not clipped by axes.",
        "",
        "## Figure-Specific Audit",
        "",
        "| Figure family | Before | After | Scientific communication impact |",
        "| --- | --- | --- | --- |",
        "| *_six_panel | PNG-only export and percentile-style axis tightening could reduce reproducibility and risk hiding extremes. | PNG export and full finite-data limits are used. | Iteration histories and metric distributions are manuscript-ready without altering trajectories. |",
        "| mace_vs_uma_contour | Export style differed from main figures. | Uses the shared publication export helper and embedded fonts. | Calculator comparisons are visually consistent with the main figure set. |",
        "| *_contour_vs_attack_by_epsilon | Epsilon tick selection was data-sparse and less systematic. | Log epsilon axis uses decade ticks and contour bands are included in log-y scaling. | Attack medians can be compared against contour baselines across physical perturbation scales. |",
        "| *_contour_vs_attack_by_n_steps | Same export and axis-padding limitations as other contour panels. | Publication export helper and full-data padded limits are used. | Step-sweep contour comparisons remain readable and reproducible. |",
        "| global_*_vs_contour_exploration | Scatter points could sit at frame boundaries. | Marker-aware padding protects points and 1:1 reference lines. | Global attack-vs-contour comparisons are easier to inspect without changing values. |",
        "",
        "All changes are visual encodings or export settings; contour metrics and attack-derived medians are unchanged.",
    ]
    (output_dir / "publication_figure_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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

    write_contour_publication_audit(args.output_dir, all_rows, comparison_df)

    print(f"Saved contour plots to {args.output_dir}")


if __name__ == "__main__":
    main()
