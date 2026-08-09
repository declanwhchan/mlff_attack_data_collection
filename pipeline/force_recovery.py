"""Force-residual recovery metrics and compact aggregate plots.

These metrics deliberately compare each potential's own force residual during
relaxation.  They are not MLFF--DFT force-vector errors.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _rms_from_force_csv(path):
    try:
        data = pd.read_csv(path)
    except Exception:
        return np.nan
    columns = ["fx", "fy", "fz"]
    if not set(columns).issubset(data.columns):
        return np.nan
    vectors = data[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    norms_sq = np.sum(vectors ** 2, axis=1)
    norms_sq = norms_sq[np.isfinite(norms_sq)]
    return float(np.sqrt(np.mean(norms_sq))) if norms_sq.size else np.nan


def _force_path(row, column, default_name):
    value = row.get(column)
    if value is not None and str(value).strip() not in {"", "nan"}:
        path = Path(str(value))
        if path.is_file():
            return path
    run_dir = Path(str(row.get("run_dir", "")))
    path = run_dir / default_name
    return path if path.is_file() else None


def _dft_ionic_summary(path):
    try:
        data = pd.read_csv(path)
    except Exception:
        return np.nan, np.nan, np.nan
    if data.empty or "rms_force_eV_A" not in data.columns:
        return np.nan, np.nan, np.nan
    rms = pd.to_numeric(data["rms_force_eV_A"], errors="coerce").dropna()
    if rms.empty:
        return np.nan, np.nan, np.nan
    return float(rms.iloc[0]), float(rms.iloc[-1]), float(len(rms) - 1)


def add_rms_force_metrics(records):
    """Return records augmented with endpoint RMS-force recovery fields."""
    data = records.copy()
    baseline, initial, final, steps = [], [], [], []
    for _, row in data.iterrows():
        calculator = str(row.get("calculator", ""))
        before = np.nan
        if calculator.startswith("dft_"):
            first, last, count = _dft_ionic_summary(row.get("dft_ionic_steps_csv"))
        else:
            before_path = _force_path(row, "before_force_csv", "before_forces.csv")
            perturbed = _force_path(row, "perturbed_force_csv", "perturbed_forces.csv")
            after = _force_path(row, "after_force_csv", "after_forces.csv")
            before = _rms_from_force_csv(before_path) if before_path else np.nan
            first = _rms_from_force_csv(perturbed) if perturbed else np.nan
            last = _rms_from_force_csv(after) if after else np.nan
            count = _number(row.get("after_relax_steps"))
        baseline.append(before)
        initial.append(first)
        final.append(last)
        steps.append(count)

    data["baseline_relaxed_rms_force_ev_a"] = baseline
    data["initial_rms_force_ev_a"] = initial
    data["final_rms_force_ev_a"] = final
    data["force_recovery_steps"] = steps
    initial_array = np.asarray(initial, dtype=float)
    final_array = np.asarray(final, dtype=float)
    valid = np.isfinite(initial_array) & np.isfinite(final_array) & (initial_array > 0)
    data["rms_force_recovery_ratio"] = np.where(valid, final_array / initial_array, np.nan)
    data["log10_rms_force_reduction"] = np.where(
        valid & (final_array > 0), np.log10(initial_array / final_array), np.nan
    )
    return data


def make_rms_force_figures(records, output_dir, model_labels, colors, prefix=""):
    """Save initial-RMS, final-RMS, and recovery-ratio plots by epsilon."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("initial_rms_force_ev_a", "Post-attack RMS force (eV/$\\AA$)", "rms_force_initial", True),
        ("final_rms_force_ev_a", "Post-relaxation RMS force (eV/$\\AA$)", "rms_force_final", True),
        ("rms_force_recovery_ratio", "RMS-force recovery ratio", "rms_force_recovery_ratio", False),
    ]
    data = records.copy()
    data["epsilon"] = pd.to_numeric(data.get("epsilon"), errors="coerce")
    attacks = [x for x in ["FGSM", "I-FGSM", "PGD"] if x in set(data.get("attack_label", []))]
    if not attacks:
        attacks = ["all"]

    written = []
    for column, ylabel, stem, show_baseline in metrics:
        subset = data[np.isfinite(pd.to_numeric(data.get(column), errors="coerce"))].copy()
        subset = subset[subset["epsilon"] > 0]
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, len(attacks), figsize=(4.2 * len(attacks), 3.4), squeeze=False)
        for ax, attack in zip(axes[0], attacks):
            panel = subset if attack == "all" else subset[subset["attack_label"] == attack]
            for calculator, group in panel.groupby("calculator", sort=False):
                summary = group.groupby("epsilon", as_index=False)[column].median().sort_values("epsilon")
                if summary.empty:
                    continue
                ax.plot(
                    summary["epsilon"], summary[column], marker="o", linewidth=1.6,
                    markersize=3.5, label=model_labels.get(str(calculator), str(calculator)),
                    color=colors.get(str(calculator), None),
                )
                if show_baseline:
                    baseline = group.groupby("epsilon", as_index=False)["baseline_relaxed_rms_force_ev_a"].median().sort_values("epsilon")
                    baseline = baseline[np.isfinite(baseline["baseline_relaxed_rms_force_ev_a"])]
                    if not baseline.empty:
                        ax.plot(baseline["epsilon"], baseline["baseline_relaxed_rms_force_ev_a"], "--x", linewidth=1.2, markersize=3.5,
                                label=f"{model_labels.get(str(calculator), str(calculator))} pre-attack relaxed", color=colors.get(str(calculator), None))
            ax.set_title(attack if attack != "all" else "All attacks")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Epsilon (Å)")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.3)
        handles, labels = axes[0][-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.86 if handles else 1))
        path = output_dir / f"{prefix}{stem}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written
