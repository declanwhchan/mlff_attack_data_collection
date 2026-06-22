#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd


KEY_COLUMNS = [
    "material_slug",
    "calculator",
    "run_folder",
    "attack_type",
    "epsilon",
    "n_steps",
    "alpha",
]

METRIC_COLUMNS = [
    "max_displacement",
    "max_delta_force",
    "final_energy",
    "before_relax_steps",
    "after_relax_steps",
]


def read_dataset(path):
    path = Path(path) / "combined_dataset.csv"
    if not path.exists():
        raise SystemExit(f"ERROR: missing {path}")
    data = pd.read_csv(path)
    if data.empty:
        raise SystemExit(f"ERROR: empty {path}")
    return add_max_delta_force(data)


def available_keys(float32, float64):
    return [column for column in KEY_COLUMNS if column in float32.columns and column in float64.columns]


def available_metrics(float32, float64):
    return [column for column in METRIC_COLUMNS if column in float32.columns and column in float64.columns]


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def force_delta_from_run(run_dir):
    run_dir = Path(run_dir)
    before_path = run_dir / "before_forces.csv"
    after_path = run_dir / "perturbed_forces.csv"

    if not before_path.exists() or not after_path.exists():
        return np.nan

    before = pd.read_csv(before_path)
    after = pd.read_csv(after_path)

    required = {"atom_index", "fx", "fy", "fz"}
    if not required.issubset(before.columns) or not required.issubset(after.columns):
        return np.nan

    merged = before.merge(after, on="atom_index", suffixes=("_before", "_after"))
    if merged.empty:
        return np.nan

    before_forces = merged[["fx_before", "fy_before", "fz_before"]].to_numpy(dtype=float)
    after_forces = merged[["fx_after", "fy_after", "fz_after"]].to_numpy(dtype=float)
    delta = np.linalg.norm(after_forces - before_forces, axis=1)

    if len(delta) == 0:
        return np.nan
    return float(np.nanmax(delta))


def add_max_delta_force(data):
    if "max_delta_force" in data.columns:
        return data
    if "run_dir" not in data.columns:
        return data

    data = data.copy()
    data["max_delta_force"] = [force_delta_from_run(path) for path in data["run_dir"]]
    return data


def style_numeric_axis(ax, xbins=5, ybins=5):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))

    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        axis.set_major_formatter(formatter)

    ax.tick_params(axis="both", labelsize=8, pad=2)


def r2_value(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.nan

    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return np.nan
    return float(corr ** 2)


def save_metric_plot(data, metric, output_dir):
    x = clean_numeric(data[f"{metric}_float64"])
    y = clean_numeric(data[f"{metric}_float32"])
    mask = x.notna() & y.notna()

    if not mask.any():
        return

    x = x[mask]
    y = y[mask]
    plot_data = data.loc[mask].copy()
    r2 = r2_value(x, y)

    fig = plt.figure(figsize=(7.2, 6.4))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(4.0, 1.15),
        height_ratios=(1.15, 4.0),
        hspace=0.05,
        wspace=0.05,
    )

    ax_hist_x = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_hist_x)
    ax_hist_y = fig.add_subplot(grid[1, 1], sharey=ax)

    colors = {"mace": "#0072B2", "uma": "#D55E00"}

    for calculator, color in colors.items():
        subset = plot_data[plot_data["calculator"] == calculator]
        if subset.empty:
            continue

        sx = clean_numeric(subset[f"{metric}_float64"])
        sy = clean_numeric(subset[f"{metric}_float32"])
        valid = sx.notna() & sy.notna()

        ax.scatter(
            sx[valid],
            sy[valid],
            s=28,
            alpha=0.72,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=calculator.upper(),
        )

    lower = float(min(x.min(), y.min()))
    upper = float(max(x.max(), y.max()))
    if lower == upper:
        pad = abs(lower) * 0.05 if lower else 1.0
        lower -= pad
        upper += pad

    bins = min(30, max(8, int(np.sqrt(len(x)))))
    ax_hist_x.hist(x, bins=bins, color="#777777", alpha=0.75)
    ax_hist_y.hist(y, bins=bins, orientation="horizontal", color="#777777", alpha=0.75)

    ax.plot([lower, upper], [lower, upper], color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(f"{metric} float64")
    ax.set_ylabel(f"{metric} float32")
    ax.set_title(f"float32 vs float64: {metric}")
    ax.text(
        0.04,
        0.96,
        f"$R^2$ = {r2:.4f}" if np.isfinite(r2) else "$R^2$ = n/a",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.85, "pad": 3},
    )

    style_numeric_axis(ax)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)

    ax_hist_x.grid(True, axis="y", alpha=0.25)
    ax_hist_y.grid(True, axis="x", alpha=0.25)
    ax_hist_x.tick_params(axis="x", labelbottom=False)
    ax_hist_y.tick_params(axis="y", labelleft=False)
    ax_hist_x.set_ylabel("count")
    ax_hist_y.set_xlabel("count")

    fig.savefig(output_dir / f"{metric}_float32_vs_float64.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--float32-dir", required=True, type=Path)
    parser.add_argument("--float64-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    float32 = read_dataset(args.float32_dir)
    float64 = read_dataset(args.float64_dir)

    keys = available_keys(float32, float64)
    metrics = available_metrics(float32, float64)

    if not keys:
        raise SystemExit("ERROR: no shared key columns for float comparison")
    if not metrics:
        raise SystemExit("ERROR: no shared metric columns for float comparison")

    merged = float32.merge(
        float64,
        on=keys,
        how="inner",
        suffixes=("_float32", "_float64"),
    )

    if merged.empty:
        raise SystemExit("ERROR: float32 and float64 datasets had no matching rows")

    for metric in metrics:
        a = clean_numeric(merged[f"{metric}_float32"])
        b = clean_numeric(merged[f"{metric}_float64"])
        merged[f"{metric}_delta_float32_minus_float64"] = a - b
        merged[f"{metric}_abs_delta"] = (a - b).abs()

    merged.to_csv(args.output_dir / "float32_float64_comparison.csv", index=False)

    summary_rows = []
    for metric in metrics:
        delta = clean_numeric(merged[f"{metric}_delta_float32_minus_float64"]).dropna()
        abs_delta = clean_numeric(merged[f"{metric}_abs_delta"]).dropna()

        if delta.empty:
            continue

        summary_rows.append({
            "metric": metric,
            "n": int(delta.shape[0]),
            "mean_delta_float32_minus_float64": float(delta.mean()),
            "median_delta_float32_minus_float64": float(delta.median()),
            "max_abs_delta": float(abs_delta.max()),
            "mean_abs_delta": float(abs_delta.mean()),
        })

        save_metric_plot(merged, metric, args.output_dir)

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "float_comprehensive_summary.csv", index=False)

    print(f"Saved float comparison outputs to {args.output_dir}")


if __name__ == "__main__":
    main()