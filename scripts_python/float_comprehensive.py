#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
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
    "mean_displacement",
    "max_displacement",
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
    return data


def available_keys(float32, float64):
    return [column for column in KEY_COLUMNS if column in float32.columns and column in float64.columns]


def available_metrics(float32, float64):
    return [column for column in METRIC_COLUMNS if column in float32.columns and column in float64.columns]


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def save_metric_plot(data, metric, output_dir):
    x = clean_numeric(data[f"{metric}_float64"])
    y = clean_numeric(data[f"{metric}_float32"])
    mask = x.notna() & y.notna()

    if not mask.any():
        return

    x = x[mask]
    y = y[mask]
    plot_data = data.loc[mask].copy()

    fig, ax = plt.subplots(figsize=(6.0, 5.4))

    for calculator, color in [("mace", "#0072B2"), ("uma", "#D55E00")]:
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

    ax.plot([lower, upper], [lower, upper], color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(f"{metric} float64")
    ax.set_ylabel(f"{metric} float32")
    ax.set_title(f"float32 vs float64: {metric}")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}_float32_vs_float64.png", dpi=300)
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