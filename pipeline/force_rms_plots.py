"""Four-model RMS-force summaries for post-attack relaxation analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read as read_structure


RMS_MODELS = ("mace_mh", "uma", "dft_mace_mh", "dft_uma")
POST_ATTACK_RELAXED_RMS_FMAX_005_COLUMN = (
    "post_attack_relaxed_rms_force_fmax_005_ev_a"
)


def _rms_force_csv(path):
    try:
        data = pd.read_csv(path)
        vectors = data[["fx", "fy", "fz"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
    except Exception:
        return np.nan
    squared = np.sum(vectors ** 2, axis=1)
    squared = squared[np.isfinite(squared)]
    return float(np.sqrt(np.mean(squared))) if squared.size else np.nan


def _artifact(row, column, filename):
    value = row.get(column)
    if value is not None and str(value).strip().lower() not in {"", "nan"}:
        path = Path(str(value))
        if path.is_file():
            return path
    path = Path(str(row.get("run_dir", ""))) / filename
    return path if path.is_file() else None


def _dft_rms_endpoints(path):
    try:
        values = pd.to_numeric(
            pd.read_csv(path)["rms_force_eV_A"], errors="coerce"
        ).dropna()
    except Exception:
        return np.nan, np.nan
    if values.empty:
        return np.nan, np.nan
    return float(values.iloc[0]), float(values.iloc[-1])


def _rms_force_from_vectors(vectors):
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        return np.nan

    squared = np.sum(vectors ** 2, axis=1)
    squared = squared[np.isfinite(squared)]
    return float(np.sqrt(np.mean(squared))) if squared.size else np.nan


def _mlff_rms_at_fmax(path, fmax):
    """Return RMS force at the first saved frame meeting ``fmax``."""
    if path is None:
        return np.nan

    try:
        trajectory = read_structure(path, index=":")
    except Exception:
        return np.nan

    for atoms in trajectory:
        try:
            forces = np.asarray(atoms.get_forces(), dtype=float)
        except Exception:
            continue

        norms = np.linalg.norm(forces, axis=1)
        finite_norms = norms[np.isfinite(norms)]
        if finite_norms.size != len(norms):
            continue
        if finite_norms.size and float(np.max(finite_norms)) <= fmax:
            return _rms_force_from_vectors(forces)

    return np.nan


def _dft_rms_at_fmax(path, fmax):
    """Return RMS force at the first DFT ionic step meeting ``fmax``."""
    try:
        data = pd.read_csv(path)
        max_forces = pd.to_numeric(
            data["max_force_eV_A"], errors="coerce"
        )
        rms_forces = pd.to_numeric(
            data["rms_force_eV_A"], errors="coerce"
        )
    except Exception:
        return np.nan

    qualifying = data.loc[
        (max_forces <= fmax) & np.isfinite(rms_forces),
    ]
    if qualifying.empty:
        return np.nan

    return float(rms_forces.loc[qualifying.index[0]])

def add_post_attack_rms_columns(records):
    """Add RMS force at attack and after subsequent relaxation."""
    data = records.copy()
    attacked, relaxed = [], []
    for _, row in data.iterrows():
        if str(row.get("calculator", "")).startswith("dft_"):
            first, last = _dft_rms_endpoints(row.get("dft_ionic_steps_csv"))
        else:
            attack_path = _artifact(
                row, "perturbed_force_csv", "perturbed_forces.csv"
            )
            final_path = _artifact(row, "after_force_csv", "after_forces.csv")
            first = _rms_force_csv(attack_path) if attack_path else np.nan
            last = _rms_force_csv(final_path) if final_path else np.nan
        attacked.append(first)
        relaxed.append(last)
    data["post_attack_rms_force_ev_a"] = attacked
    data["post_attack_relaxed_rms_force_ev_a"] = relaxed
    return data


def add_post_attack_relaxed_rms_at_fmax_column(records, fmax):
    """Add RMS forces at the first post-attack state converged to ``fmax``."""
    data = records.copy()
    values = []

    for _, row in data.iterrows():
        if str(row.get("calculator", "")).startswith("dft_"):
            value = _dft_rms_at_fmax(
                row.get("dft_ionic_steps_csv"),
                fmax,
            )
        else:
            trajectory_path = _artifact(
                row,
                "after_attack_relax_traj",
                "after_attack_relaxation.traj",
            )
            value = _mlff_rms_at_fmax(trajectory_path, fmax)
        values.append(value)

    data[POST_ATTACK_RELAXED_RMS_FMAX_005_COLUMN] = values
    return data

def rms_value_getter(column):
    """Adapt one per-record RMS value to the ranking violin interface."""
    def getter(row):
        try:
            value = float(row.get(column))
        except (TypeError, ValueError):
            return None, f"Missing {column}"
        if not np.isfinite(value) or value <= 0:
            return None, f"Invalid {column}"
        return np.asarray([value]), None
    return getter


def save_random_seed_rms_plots(records, output_dir, labels, colors):
    """Save clean post-attack and post-relaxation four-model RMS curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = records[records["calculator"].isin(RMS_MODELS)].copy()
    data["epsilon"] = pd.to_numeric(data.get("epsilon"), errors="coerce")
    attacks = [name for name in ("FGSM", "I-FGSM", "PGD")
               if name in set(data.get("attack_label", []))]
    if not attacks:
        attacks = ["all"]

    figures = (
        ("post_attack_rms_force_ev_a", "Post-attack RMS force (eV/$\\AA$)",
         "rms_force_post_attack.png"),
        ("post_attack_relaxed_rms_force_ev_a",
         "Post-attack + relaxation RMS force (eV/$\\AA$)",
         "rms_force_post_attack_relaxed.png"),
    )
    for column, ylabel, filename in figures:
        subset = data[np.isfinite(pd.to_numeric(data[column], errors="coerce"))]
        subset = subset[subset["epsilon"] > 0]
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, len(attacks), figsize=(4.3 * len(attacks), 3.5), squeeze=False)
        for axis, attack in zip(axes[0], attacks):
            panel = subset if attack == "all" else subset[subset["attack_label"] == attack]
            for calculator in RMS_MODELS:
                group = panel[panel["calculator"] == calculator].copy()
                per_seed = group.groupby(["seed", "epsilon"], as_index=False)[column].median()
                summary = per_seed.groupby("epsilon", as_index=False).agg(
                    median=(column, "median"),
                    q25=(column, lambda values: values.quantile(0.25)),
                    q75=(column, lambda values: values.quantile(0.75)),
                ).sort_values("epsilon")
                if summary.empty:
                    continue
                x = summary["epsilon"].to_numpy(float)
                median = summary["median"].to_numpy(float)
                q25 = summary["q25"].to_numpy(float)
                q75 = summary["q75"].to_numpy(float)
                axis.scatter(per_seed["epsilon"], per_seed[column], s=17,
                             color=colors[calculator], alpha=0.46,
                             edgecolors="white", linewidths=0.35, zorder=3)
                axis.fill_between(x, q25, q75, color=colors[calculator],
                                  alpha=0.18, linewidth=0, zorder=1)
                axis.plot(x, median, linewidth=2.25, color=colors[calculator],
                          label=labels[calculator], zorder=4)
            axis.set_title(attack if attack != "all" else "All attacks")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel(r"$\epsilon$ ($\AA$)")
            axis.set_ylabel(ylabel)
            axis.grid(True, which="both", alpha=0.28)
        handles, legend_labels = axes[0][-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.84 if handles else 1))
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
