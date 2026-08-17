#!/usr/bin/env python3
"""Lean presentation diagnostics for post-attack + relaxation anomalies.

The script deliberately reads existing comprehensive/random-seed records.  It
does not rerun attacks or DFT.  Optional ASE phonons and MLFF PES evaluations
are restricted to the automatically selected presentation case studies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, NullLocator
from matplotlib.patches import Patch, Polygon, Rectangle
import numpy as np
import pandas as pd
from ase.io import read as ase_read
from ase.phonons import Phonons
from scipy.interpolate import make_interp_spline

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from force_rms_plots import add_post_attack_rms_columns
from run_tests import setup_calculator


MODELS = ("mace_mh", "uma", "dft_mace_mh", "dft_uma")
MLFF_MODELS = ("mace_mh", "uma")
COLORS = {
    "mace_mh": "#0072B2", "uma": "#D55E00",
    "dft_mace_mh": "#7A7A7A", "dft_uma": "#B0B0B0",
}
ATTACK_COLORS = {
    "Contour": "#009E73",
    "FGSM": "#0072B2",
    "I-FGSM": "#E69F00",
    "PGD": "#CC79A7",
}
LABELS = {
    "mace_mh": "MACE-MH-1", "uma": "UMA-S-1p1",
    "dft_mace_mh": "DFT (MACE-MH)", "dft_uma": "DFT (UMA)",
}
FINAL_JACCARD = "after_attack_after_relaxation__neighbor_jaccard_distance"
FINAL_MAX_CN = "after_attack_after_relaxation__coordination_change_max"
DEFAULT_MACE_MH_HEAD = "omat_pbe"
ENSEMBLE_EPSILONS = np.array((0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0))
EXPECTED_ENSEMBLE_MATERIALS = 20


def numeric(frame, column):
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def existing_path(row, *columns, fallback=None):
    for column in columns:
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = Path(str(row.get("run_dir", ""))) / candidate
            if candidate.is_file():
                return candidate
    if fallback:
        candidate = Path(str(row.get("run_dir", ""))) / fallback
        if candidate.is_file():
            return candidate
    return None


def load_records(project_root):
    random_file = project_root / "random_seed" / "random_seed_combined.csv"
    if random_file.is_file():
        data = pd.read_csv(random_file, low_memory=False)
    else:
        frames = []
        for trial in sorted(project_root.glob("trial*_seed*")):
            path = trial / "outputs_comprehensive" / "float64" / "combined_dataset.csv"
            if path.is_file():
                frame = pd.read_csv(path)
                frame["trial"] = trial.name
                frames.append(frame)
        if not frames:
            raise FileNotFoundError("No random_seed_combined.csv or trial combined_dataset.csv found")
        data = pd.concat(frames, ignore_index=True, sort=False)
        data = add_post_attack_rms_columns(data)

    data = data[data.get("calculator", "").astype(str).isin(MODELS)].copy()
    data = data[~data.get("run_id", "").astype(str).str.contains("_steps", regex=False)].copy()
    data["epsilon_percent"] = numeric(data, "epsilon_percent_displacement")
    data["epsilon"] = numeric(data, "epsilon")
    # Nominal epsilon is shared by every material in one attack step. The
    # measured percentage is material-dependent, so it cannot identify a
    # cross-material transition.
    data["transition_epsilon"] = data["epsilon"].fillna(data["epsilon_percent"])
    data["final_jaccard"] = numeric(data, FINAL_JACCARD)
    # Prefer final-stage columns but fall back per row for legacy records.
    data["final_jaccard"] = data["final_jaccard"].fillna(numeric(data, "neighbor_jaccard_distance"))
    data["final_max_cn"] = numeric(data, FINAL_MAX_CN).fillna(numeric(data, "coordination_change_max"))
    # Backfill partial legacy records so MLFF--DFT force pairs are retained.
    calculated_rms = add_post_attack_rms_columns(data)
    stored_rms = numeric(data, "post_attack_relaxed_rms_force_ev_a")
    data["post_attack_relaxed_rms_force_ev_a"] = stored_rms.fillna(
        numeric(calculated_rms, "post_attack_relaxed_rms_force_ev_a")
    )
    data["final_rms_force"] = numeric(data, "post_attack_relaxed_rms_force_ev_a")
    data["final_energy"] = numeric(data, "final_energy")
    data["relax_steps"] = numeric(data, "after_relax_steps")
    data["converged"] = data.get("after_relax_converged", False).fillna(False).astype(bool)
    return data


def paired_final_rms_force_records(data):
    """Pair final MLFF RMS forces to their DFT reruns by source run ID."""
    mlff = data[data["calculator"].isin(MLFF_MODELS)]
    dft = data[data["calculator"].astype(str).str.startswith("dft_")]
    by_trial = {(str(row.get("trial", "")), str(row.get("run_id", ""))): row for _, row in mlff.iterrows()}
    pairs = []
    for _, dft_row in dft.iterrows():
        source_id = dft_row.get("dft_source_run_id")
        if source_id is None or pd.isna(source_id):
            continue
        source = by_trial.get((str(dft_row.get("trial", "")), str(source_id)))
        if source is None:
            matches = mlff[mlff["run_id"].astype(str) == str(source_id)]
            source = matches.iloc[0] if len(matches) == 1 else None
        if source is None:
            continue
        mlff_rms = pd.to_numeric(source.get("final_rms_force"), errors="coerce")
        dft_rms = pd.to_numeric(dft_row.get("final_rms_force"), errors="coerce")
        if np.isfinite(mlff_rms) and np.isfinite(dft_rms):
            pairs.append({
                "run_id": source_id, "source_model": source["calculator"],
                "epsilon": source.get("epsilon"),
                "mlff_final_rms_force_ev_a": float(mlff_rms),
                "dft_final_rms_force_ev_a": float(dft_rms),
                "rms_force_difference_ev_a": abs(float(mlff_rms) - float(dft_rms)),
            })
    return pd.DataFrame(pairs)


def save_final_rms_force_heatmap(data, output):
    """Plot final relaxed RMS-force differences for MACE-MH and UMA vs DFT."""
    pairs = paired_final_rms_force_records(data)
    pairs.to_csv(output / "paired_mlff_dft_final_rms_forces.csv", index=False)
    values = {model: np.full(len(ENSEMBLE_EPSILONS), np.nan) for model in MLFF_MODELS}
    if not pairs.empty:
        pairs["epsilon"] = pd.to_numeric(pairs["epsilon"], errors="coerce")
        for model in MLFF_MODELS:
            rows = pairs[(pairs["source_model"] == model) & np.isfinite(pairs["epsilon"]) & (pairs["epsilon"] > 0)].copy()
            if rows.empty:
                continue
            bins = np.abs(np.log10(rows["epsilon"].to_numpy(float))[:, None] - np.log10(ENSEMBLE_EPSILONS)[None, :]).argmin(axis=1)
            for index, value in pd.to_numeric(rows["rms_force_difference_ev_a"], errors="coerce").groupby(bins).median().items():
                values[model][int(index)] = value
    finite = np.concatenate([row[np.isfinite(row)] for row in values.values()])
    normalized = {model: np.full(len(ENSEMBLE_EPSILONS), np.nan) for model in MLFF_MODELS}
    if finite.size:
        lower, upper = float(finite.min()), float(finite.max())
        for model, row in values.items():
            normalized[model] = np.where(np.isfinite(row), 1.0 if np.isclose(lower, upper) and upper > 0 else (0.0 if np.isclose(lower, upper) else (row - lower) / (upper - lower)), np.nan)
    fig, ax = plt.subplots(figsize=(11.2, 2.6), facecolor="white")
    cmap = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.colormaps["Blues"]); cmap.set_array([])
    for column in range(len(ENSEMBLE_EPSILONS)):
        for model, corners, position in (("mace_mh", ((column-.5, -.5), (column+.5, -.5), (column+.5, .5)), (column+.18, -.18)), ("uma", ((column-.5, -.5), (column-.5, .5), (column+.5, .5)), (column-.18, .18))):
            value = normalized[model][column]
            ax.add_patch(Polygon(corners, closed=True, facecolor=cmap.to_rgba(value) if np.isfinite(value) else "#F1F5F9", edgecolor="white", linewidth=.85))
            if np.isfinite(value): ax.text(*position, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value >= .58 else "#0F172A")
    ax.set(xlim=(-.5, len(ENSEMBLE_EPSILONS)-.5), ylim=(.5, -.5), yticks=[0], yticklabels=[r"RMS $\Delta$ force"])
    ax.set_xticks((0, 2, 4, 6, 8), [r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"]); ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle("MLFF--DFT final RMS-force difference after perturbation + relaxation", fontsize=14, fontweight="bold")
    fig.legend(handles=[Patch(facecolor="#4B8CC0", label="MACE-MH, upper triangle"), Patch(facecolor="#4B8CC0", label="UMA, lower triangle")], loc="upper center", bbox_to_anchor=(.5, .88), ncol=2, frameon=False, fontsize=9)
    colorbar = fig.colorbar(cmap, ax=ax, fraction=.04, pad=.035); colorbar.set_label("Normalized absolute RMS-force difference from DFT", fontsize=9); colorbar.outline.set_visible(False)
    ax.set_xlabel(r"$\epsilon$ strength (% min lattice)"); fig.subplots_adjust(left=.16, right=.89, bottom=.25, top=.62)
    fig.savefig(output / "direct_comparison_normalized_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def grouped_curve(data, metric, models):
    usable = data[data["calculator"].isin(models)].dropna(subset=["epsilon_percent", metric]).copy()
    return usable.groupby(["calculator", "attack_label", "epsilon_percent"], as_index=False)[metric].median()


def jaccard_seed_summary(group):
    """Summarise material medians within each seed, then across seeds."""
    usable = group.dropna(subset=["epsilon_percent", "final_jaccard"]).copy()
    usable = usable[usable["epsilon_percent"] > 0]
    if usable.empty:
        return pd.DataFrame(columns=["epsilon_percent", "median", "q25", "q75", "seed_count"])

    # Trial is a stable fallback for legacy files without an explicit seed.
    seed_key = usable.get("seed", pd.Series(np.nan, index=usable.index)).astype("string")
    trial_key = usable.get("trial", pd.Series(np.nan, index=usable.index)).astype("string")
    usable["_seed_key"] = seed_key.fillna(trial_key).fillna("unlabelled")
    per_seed = usable.groupby(["_seed_key", "epsilon_percent"], as_index=False)["final_jaccard"].median()
    return per_seed.groupby("epsilon_percent", as_index=False).agg(
        median=("final_jaccard", "median"),
        q25=("final_jaccard", lambda values: values.quantile(.25)),
        q75=("final_jaccard", lambda values: values.quantile(.75)),
        seed_count=("_seed_key", "nunique"),
    )


def jaccard_trajectory_key(rows):
    """Return an identity that cannot connect records from separate seeds."""
    seed = rows.get("seed", pd.Series(np.nan, index=rows.index)).astype("string")
    trial = rows.get("trial", pd.Series(np.nan, index=rows.index)).astype("string")
    run = rows.get("run_id", pd.Series(np.nan, index=rows.index)).astype("string")
    return seed.fillna(trial).fillna(run).fillna("unlabelled")


def material_seed_medians(data, metric, models=None):
    """One material response per seed and epsilon, with legacy fallbacks."""
    usable = data.dropna(subset=["material_slug", "transition_epsilon", metric]).copy()
    if models is not None:
        usable = usable[usable["calculator"].isin(models)]
    seed_key = usable.get("seed", pd.Series(np.nan, index=usable.index)).astype("string")
    trial_key = usable.get("trial", pd.Series(np.nan, index=usable.index)).astype("string")
    usable["_seed_key"] = seed_key.fillna(trial_key).fillna("unlabelled")
    return usable.groupby(
        ["calculator", "attack_label", "_seed_key", "material_slug", "transition_epsilon"],
        as_index=False,
    ).agg(**{metric: (metric, "median"), "epsilon_percent": ("epsilon_percent", "median")})


def seed_median_curve(data, metric, models):
    """Typical material response: material median within seed, then seed median."""
    per_material = material_seed_medians(data, metric, models)
    return per_material.groupby(
        ["calculator", "attack_label", "_seed_key", "transition_epsilon"], as_index=False,
    ).agg(**{metric: (metric, "median"), "epsilon_percent": ("epsilon_percent", "median")}).groupby(
        ["calculator", "attack_label", "transition_epsilon"], as_index=False,
    ).agg(value=(metric, "median"), epsilon_percent=("epsilon_percent", "median"), seed_count=("_seed_key", "nunique"))


def choose_anomalies(data):
    """Select the largest MLFF Jaccard dip and post-10% Jaccard spike."""
    def adjacent_change(curve, direction, minimum_percent=None):
        candidates = []
        for _, group in curve.groupby(["calculator", "attack_label"]):
            group = group.sort_values("transition_epsilon").copy()
            group["epsilon"] = group["transition_epsilon"]
            group["previous_epsilon"] = group["transition_epsilon"].shift()
            group["change"] = group["value"].diff()
            valid = group.dropna(subset=["previous_epsilon", "change"])
            if minimum_percent is not None:
                valid = valid[valid["epsilon_percent"] > minimum_percent]
            if not valid.empty:
                candidates.append(valid.loc[valid["change"].idxmax() if direction == "max" else valid["change"].idxmin()])
        return (max(candidates, key=lambda row: row["change"]) if direction == "max" else min(candidates, key=lambda row: row["change"])) if candidates else pd.Series(dtype=object)

    dip = adjacent_change(seed_median_curve(data, "final_jaccard", MLFF_MODELS), "min")
    spike = adjacent_change(seed_median_curve(data, "final_jaccard", MLFF_MODELS), "max", minimum_percent=10)
    return dip, spike


def contribution_rows(data, dip, spike):
    """Rank material-level changes at the selected MLFF topology transition.

    Prefer within-seed pairs.  Legacy combined files can label the same seeds
    differently at adjacent epsilons; in that case compare the two material
    seed-median distributions instead of suppressing the attribution entirely.
    """
    parts = []
    for anomaly, metric, kind in ((dip, "final_jaccard", "Jaccard dip"), (spike, "final_jaccard", "Jaccard spike (>10%)")):
        if anomaly.empty or pd.isna(anomaly.get("previous_epsilon")):
            continue
        subset = data[(data["calculator"] == anomaly["calculator"]) & (data["attack_label"] == anomaly["attack_label"])]
        values = material_seed_medians(subset, metric)
        current = values[np.isclose(values["transition_epsilon"], anomaly["epsilon"])]
        previous = values[np.isclose(values["transition_epsilon"], anomaly["previous_epsilon"])]
        paired = current.merge(previous, on=["material_slug", "_seed_key"], suffixes=("_current", "_previous"))
        if not paired.empty:
            paired["delta"] = paired[f"{metric}_current"] - paired[f"{metric}_previous"]
            score = paired.groupby("material_slug", as_index=False).agg(
                score=("delta", "median"), seed_count=("_seed_key", "nunique"),
            )
            score["comparison"] = "paired seed medians"
        else:
            # Retain an auditable material attribution when adjacent records
            # cannot be paired by seed/trial labels.
            current_summary = current.groupby("material_slug", as_index=False).agg(
                current=(metric, "median"), current_seed_count=("_seed_key", "nunique"),
            )
            previous_summary = previous.groupby("material_slug", as_index=False).agg(
                previous=(metric, "median"), previous_seed_count=("_seed_key", "nunique"),
            )
            score = current_summary.merge(previous_summary, on="material_slug", how="inner")
            if score.empty:
                continue
            score["score"] = score["current"] - score["previous"]
            score["seed_count"] = score[["current_seed_count", "previous_seed_count"]].min(axis=1).astype(int)
            score["comparison"] = "unpaired seed medians"
        score["kind"] = kind
        parts.append(score[["material_slug", "kind", "score", "seed_count", "comparison"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["material_slug", "kind", "score", "seed_count", "comparison"]
    )
def save_attribution(data, contributions, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, kind, ascending, color in zip(axes, ("Jaccard dip", "Jaccard spike (>10%)"), (True, False), ("#F58518", "#0072B2")):
        selected = contributions[contributions["kind"] == kind].sort_values("score", ascending=ascending)
        if selected.empty:
            ax.text(.5, .5, "No material-level records\nfor this transition", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(kind, fontweight="bold")
            continue
        selected = selected.head(20).iloc[::-1]
        ax.barh(selected["material_slug"], selected["score"], color=color)
        ax.axvline(0, color="#222", linewidth=.8)
        comparison = selected["comparison"].iloc[0]
        ax.set_title(f"{kind}\n({comparison})", fontweight="bold")
        ax.set_xlabel("? final neighbor-Jaccard distance")
        ax.grid(axis="x", alpha=.25)
    fig.savefig(output / "01_material_attribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_cases(data, contributions, dip, spike):
    """Choose and label materials for optional local diagnostics.

    A material can be selected for both anomalies. Preserve those roles even
    when the material names are later de-duplicated.
    """
    selected = {}
    preferences = {"Jaccard dip": dip, "Jaccard spike (>10%)": spike}
    for kind, ascending in (("Jaccard dip", True), ("Jaccard spike (>10%)", False)):
        ranked = contributions[contributions["kind"] == kind].sort_values("score", ascending=ascending)
        if not ranked.empty:
            preference = preferences[kind]
            # Preserve the actual top contributor. Diagnostic availability is
            # recorded separately and never changes the attribution ranking.
            selected[kind] = {
                "material_slug": ranked.iloc[0]["material_slug"],
                "median_change": float(ranked.iloc[0]["score"]),
                "diagnostic_ready": bool(
                    source_mlff_row(data, ranked.iloc[0]["material_slug"], preference) is not None
                ),
            }
    # A stable control is the material closest to median Jaccard at the dip.
    if not dip.empty:
        pool = data[(data["calculator"] == dip["calculator"]) & np.isclose(data["transition_epsilon"], dip["epsilon"])]
        median = pool["final_jaccard"].median()
        dip_material = selected.get("Jaccard dip", {}).get("material_slug")
        control = (pool.groupby("material_slug")["final_jaccard"].median() - median).abs().sort_values().index
        control = [material for material in control if material != dip_material]
        if control:
            selected["Jaccard control"] = {"material_slug": control[0]}
    return selected


def choose_zero_jaccard_control(data, target_percent=1.0):
    """Choose a zero-topology-change point near ``target_percent``.

    The control is selected on measured displacement: unlike a shared attack
    transition, its purpose is to represent a small perturbation whose final
    neighbour graph is unchanged.
    """
    usable = data[data["calculator"].isin(MLFF_MODELS)].dropna(
        subset=["material_slug", "transition_epsilon", "epsilon_percent", "final_jaccard"]
    ).copy()
    usable = usable[(usable["epsilon_percent"] > 0) & np.isclose(usable["final_jaccard"], 0.0)]
    # Prefer UMA for this optional control because its runtime is commonly
    # available when MACE is not; retain MACE only when no UMA control exists.
    uma_controls = usable[usable["calculator"] == "uma"]
    if not uma_controls.empty:
        usable = uma_controls
    if usable.empty:
        return pd.Series(dtype=object)
    usable["_log_distance_to_target"] = np.abs(
        np.log10(usable["epsilon_percent"] / target_percent)
    )
    return usable.sort_values(
        ["converged", "_log_distance_to_target"], ascending=[False, True]
    ).iloc[0]


def structural_artifacts(row):
    """Return the three saved states needed for a local diagnostic, if present.

    Comprehensive records may store paths relative to either ``run_dir`` or
    ``actual_output_dir``.  The latter was not considered by the original
    why-plots resolver, which made valid post-processed records look absent.
    """
    run_dir = Path(str(row.get("run_dir", "")))
    output_dir = Path(str(row.get("actual_output_dir", "")))

    def resolve(columns, names):
        candidates = []
        for column in columns:
            value = row.get(column)
            if value is None or pd.isna(value) or not str(value).strip():
                continue
            candidate = Path(str(value).strip())
            candidates.append(candidate)
            if not candidate.is_absolute():
                candidates.extend((run_dir / candidate, output_dir / candidate))
        for name in names:
            candidates.extend((run_dir / name, output_dir / name))
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    reference = resolve(("before_relax_traj",), ("before_attack_relaxation.traj",))
    perturbed = resolve(("output_cif", "perturbed_cif"), ("perturbed.cif",))
    final = resolve(
        ("final_relaxed_cif", "after_attack_relax_traj"),
        ("final_relaxed.cif", "after_attack_relaxation.traj"),
    )
    return reference, perturbed, final


def source_mlff_row(data, material, preference, diagnostic_model=None, require_zero_jaccard=False):
    """Select a diagnostic-ready MLFF record at the selected transition."""
    subset = data[(data["material_slug"] == material) & data["calculator"].isin(MLFF_MODELS)].copy()
    if subset.empty or preference is None or preference.empty:
        return None
    subset = subset[
        (subset["attack_label"] == preference.get("attack_label"))
        & np.isclose(subset["transition_epsilon"], preference.get("epsilon", np.nan))
    ]
    if require_zero_jaccard:
        subset = subset[np.isclose(subset["final_jaccard"], 0.0)]
    if subset.empty:
        return None
    if diagnostic_model:
        preferred = subset[subset["calculator"] == diagnostic_model]
        if not preferred.empty:
            subset = preferred
    elif require_zero_jaccard:
        # The MACE and UMA runtimes are often installed separately. Prefer an
        # available UMA control when no diagnostic model was explicitly set,
        # avoiding an accidental MACE-only selection for this optional panel.
        preferred = subset[subset["calculator"] == "uma"]
        if not preferred.empty:
            subset = preferred
    subset = subset.copy()
    subset["_states_available"] = subset.apply(
        lambda row: all(path is not None for path in structural_artifacts(row)), axis=1
    )
    ready = subset[subset["_states_available"]]
    if ready.empty:
        return None
    if require_zero_jaccard and "epsilon_percent" in preference:
        target = preference["epsilon_percent"]
        ready = ready.assign(_epsilon_distance=(ready["epsilon_percent"] - target).abs())
        return ready.sort_values(["converged", "_epsilon_distance"], ascending=[False, True]).iloc[0]
    return ready.sort_values(["converged", "final_jaccard"], ascending=[False, False]).iloc[0]


def load_states(row):
    traj, perturbed, final = structural_artifacts(row)
    if not traj or not perturbed or not final:
        raise FileNotFoundError("Missing reference, perturbed, or final structural artifact")
    return ase_read(traj, index=-1), ase_read(perturbed), ase_read(final, index=-1)


def load_relaxation_frames(row):
    """Load saved post-attack relaxation frames, if a trajectory is available."""
    run_dir = Path(str(row.get("run_dir", "")))
    output_dir = Path(str(row.get("actual_output_dir", "")))
    candidates = []
    value = row.get("after_attack_relax_traj")
    if value is not None and pd.notna(value) and str(value).strip():
        path = Path(str(value).strip())
        candidates.append(path)
        if not path.is_absolute():
            candidates.extend((run_dir / path, output_dir / path))
    candidates.extend((run_dir / "after_attack_relaxation.traj", output_dir / "after_attack_relaxation.traj"))
    trajectory = next((path for path in candidates if path.is_file() and path.suffix == ".traj"), None)
    return ase_read(trajectory, index=":") if trajectory is not None else []


def project_structures(structures, reference, u1, u2):
    """Project structures into the attack/orthogonal-relaxation PES slice."""
    return np.array([
        (np.dot((atoms.positions - reference.positions).reshape(-1), u1),
         np.dot((atoms.positions - reference.positions).reshape(-1), u2))
        for atoms in structures
    ])

def save_placeholder(path, title, reason):
    """Write a presentation-safe diagnostic when a calculation is unavailable."""
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.set_axis_off()
    ax.set_title(title, fontweight="bold", pad=18)
    ax.text(.5, .56, "Diagnostic unavailable", ha="center", va="center", fontsize=16, fontweight="bold", transform=ax.transAxes)
    ax.text(.5, .42, reason, ha="center", va="center", fontsize=10, wrap=True, transform=ax.transAxes)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def transition_seed_values(data, anomaly, metric):
    """Material-median response per seed at the two selected epsilons."""
    if anomaly.empty:
        return pd.DataFrame(columns=["_seed_key", "previous", "current"])
    subset = data[(data["calculator"] == anomaly["calculator"]) & (data["attack_label"] == anomaly["attack_label"])]
    values = material_seed_medians(subset, metric)
    previous = values[np.isclose(values["transition_epsilon"], anomaly["previous_epsilon"])]
    current = values[np.isclose(values["transition_epsilon"], anomaly["epsilon"])]
    previous = previous.groupby("_seed_key", as_index=False)[metric].median().rename(columns={metric: "previous"})
    current = current.groupby("_seed_key", as_index=False)[metric].median().rename(columns={metric: "current"})
    return previous.merge(current, on="_seed_key", how="outer")


def draw_threshold_evidence(ax, values, anomaly, metric_label, title, color):
    if anomaly.empty or values.empty:
        ax.set_axis_off()
        ax.set_title(title, fontweight="bold")
        ax.text(.5, .5, "No seed-level values at both threshold samples", ha="center", va="center", transform=ax.transAxes)
        return
    previous = values["previous"].dropna().to_numpy()
    current = values["current"].dropna().to_numpy()
    paired = values.dropna(subset=["previous", "current"])
    rng = np.random.default_rng(7)
    for x, sample in ((0, previous), (1, current)):
        ax.scatter(x + rng.normal(0, .035, len(sample)), sample, s=42, color=color, alpha=.78, zorder=3)
        if len(sample):
            ax.scatter(x, np.median(sample), marker="D", s=64, color="#222", edgecolor="white", linewidth=1, zorder=5)
    for _, row in paired.iterrows():
        ax.plot((0, 1), (row["previous"], row["current"]), color=color, alpha=.32, linewidth=1.2, zorder=2)
    prior_label = f"nominal epsilon={anomaly['previous_epsilon']:.4g}"
    current_label = f"nominal epsilon={anomaly['epsilon']:.4g}"
    ax.set_xticks((0, 1), (prior_label, current_label))
    ax.set_xlim(-.35, 1.35)
    ax.set_ylabel(metric_label)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=.25)
    if len(previous) and len(current):
        delta = np.median(current) - np.median(previous)
        ax.text(.5, .98, f"median: {np.median(previous):.3g} -> {np.median(current):.3g}   delta = {delta:+.3g}\nn = {len(paired)} paired seeds", ha="center", va="top", transform=ax.transAxes, fontsize=8.5, bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none"})


def save_anomaly_transitions(data, dip, spike, output):
    """Show the MLFF topology thresholds and coordination responses."""
    specifications = (
        (dip, "final_jaccard", "Final neighbor-Jaccard distance", "MLFF topology threshold", "#F58518"),
        (dip, "final_max_cn", "Max coordination-number change", "CN change at MLFF topology threshold", "#54A24B"),
        (spike, "final_jaccard", "Final neighbor-Jaccard distance", "MLFF topology spike (>10%)", "#0072B2"),
        (spike, "final_max_cn", "Max coordination-number change", "CN change at MLFF topology spike (>10%)", "#54A24B"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.6), constrained_layout=True)
    for ax, (anomaly, metric, ylabel, title, color) in zip(axes.flat, specifications):
        values = transition_seed_values(data, anomaly, metric)
        draw_threshold_evidence(ax, values, anomaly, ylabel, title, color)
    fig.suptitle("Threshold evidence: selected adjacent epsilon samples", fontweight="bold", fontsize=15)
    fig.savefig(output / "02_selected_anomaly_transitions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def calculator_for(row, atoms, mace_mh_head=DEFAULT_MACE_MH_HEAD):
    model_path = str(row.get("model_path", "")).strip()
    model_id = str(row.get("calculator", "")).replace("dft_", "")
    # Combined random-seed records intentionally omit model paths. Optional
    # case-study calculations therefore use the repository's canonical files.
    if model_path.lower() in {"", "nan", "none", "null"}:
        model_path = {
            "mace_mh": "mace-mh-1.model",
            "mace_model": "MACE_model.model",
            "uma": "uma-s-1p1.pt",
            "mtp": "pot.almtp",
        }.get(model_id, "")
    backend_by_model = {
        "mace_mh": "mace",
        "mace_model": "mace",
        "uma": "uma",
        "chgnet": "chgnet",
        "mtp": "mtp",
    }
    backend = backend_by_model.get(
        model_id,
        str(row.get("calculator_backend") or model_id),
    )

    if backend not in {"mace", "uma", "chgnet", "mtp"}:
        raise ValueError(
            f"Unsupported calculator/model combination: {model_id!r}"
        )

    if backend in {"mace", "mtp"}:
        path = Path(model_path)
        model_path = str(
            path if path.is_absolute() else SCRIPT_DIR.parent / path
        )
    elif backend == "uma":
        model_path = Path(model_path).stem

    recorded_head = row.get("mace_head")
    if pd.isna(recorded_head) or not str(recorded_head).strip():
        # random_seed_combined.csv does not retain model settings. MACE-MH-1
        # has several heads and therefore cannot infer one at load time.
        recorded_head = mace_mh_head if model_id == "mace_mh" else None

    def value_or_default(column, default):
        value = row.get(column, default)
        return default if value is None or pd.isna(value) or not str(value).strip() else value

    configured = setup_calculator(
        atoms.copy(),
        model_path,
        device=str(value_or_default("device", "cpu")),
        dtype_str=str(value_or_default("dtype_str", "float64")),
        calculator=backend,
        mace_head=recorded_head,
        uma_task=str(value_or_default("uma_task", "omat")),
        uma_charge=None if pd.isna(row.get("uma_charge")) else int(row.get("uma_charge")),
        uma_spin=None if pd.isna(row.get("uma_spin")) else int(row.get("uma_spin")),
    )
    if configured is None or configured.calc is None:
        raise RuntimeError(f"Could not attach the {backend} calculator for {model_id}.")
    return configured


def same_reference_structure(candidate, reference, atol=1e-6):
    """Return whether two saved references describe the exact same PES origin."""
    return (
        len(candidate) == len(reference)
        and candidate.get_chemical_symbols() == reference.get_chemical_symbols()
        and np.array_equal(candidate.pbc, reference.pbc)
        and np.allclose(candidate.cell.array, reference.cell.array, atol=atol, rtol=0)
        and np.allclose(candidate.positions, reference.positions, atol=atol, rtol=0)
    )


def comprehensive_basin_overlay(data, row, reference, u1, u2):
    """Return matched low-epsilon MLFF finals and the corresponding DFT final.

    A local PES coordinate system belongs to one material and attack family.
    The overlay deliberately excludes other materials and seeds rather than
    projecting incomparable structures into this slice.
    """
    family = data[
        (data["material_slug"] == row["material_slug"])
        & (data["calculator"] == row["calculator"])
        & (data["attack_label"] == row["attack_label"])
        & (data["transition_epsilon"] > 0)
        & (data["transition_epsilon"] < 10)
    ].copy()
    family["_trajectory_key"] = jaccard_trajectory_key(family)
    selected_key = jaccard_trajectory_key(pd.DataFrame([row])).iloc[0]
    family = family[family["_trajectory_key"] == selected_key]
    points = []
    for _, candidate in family.iterrows():
        try:
            candidate_reference, _, candidate_final = load_states(candidate)
            if same_reference_structure(candidate_reference, reference):
                points.append(project_structures((candidate_final,), reference, u1, u2)[0])
        except Exception:
            continue
    low_epsilon_points = np.asarray(points, dtype=float).reshape((-1, 2)) if points else np.empty((0, 2))

    dft_point = None
    dft_model = f"dft_{row['calculator']}"
    source_id = str(row.get("run_id", ""))
    dft_rows = data[
        (data["calculator"] == dft_model)
        & (data.get("dft_source_run_id", pd.Series("", index=data.index)).astype(str) == source_id)
    ]
    for _, dft_row in dft_rows.iterrows():
        try:
            dft_reference, _, dft_final = load_states(dft_row)
            if same_reference_structure(dft_reference, reference):
                dft_point = project_structures((dft_final,), reference, u1, u2)[0]
                break
        except Exception:
            continue
    return low_epsilon_points, dft_point


def save_pes(data, material, preference, output, grid, mace_mh_head, anomaly_key, anomaly_label, diagnostic_model, require_zero_jaccard=False):
    """Render a 2-D MLFF energy landscape for one selected anomaly."""
    destination = output / f"03_basin_map_2d_{anomaly_key}.png"
    if not material:
        save_placeholder(destination, "Local MLFF basin", f"No material contributor was available for {anomaly_label}.")
        return
    row = source_mlff_row(data, material, preference, diagnostic_model, require_zero_jaccard)
    if row is None:
        save_placeholder(destination, "Local MLFF basin", f"The selected material has no matching MLFF record at the {anomaly_label} threshold.")
        return
    try:
        reference, perturbed, final = load_states(row)
        d1 = (perturbed.positions - reference.positions).reshape(-1)
        if np.linalg.norm(d1) < 1e-8:
            save_placeholder(destination, "Local MLFF basin", "The recorded perturbation has zero displacement from the reference structure.")
            return
        d2 = (final.positions - reference.positions).reshape(-1)
        u1 = d1 / np.linalg.norm(d1)
        d2 -= np.dot(d2, u1) * u1
        if np.linalg.norm(d2) < 1e-8:
            save_placeholder(destination, "Local MLFF basin", "Relaxation remains collinear with the attack, so no independent 2-D basin slice exists.")
            return
        u2 = d2 / np.linalg.norm(d2)
        endpoints = project_structures((reference, perturbed, final), reference, u1, u2)
        relaxation_frames = load_relaxation_frames(row)
        relaxation_path = project_structures(relaxation_frames, reference, u1, u2) if relaxation_frames else np.empty((0, 2))
        # Relaxation starts from the perturbed structure, never from the reference.
        path_points = np.vstack((endpoints[1], relaxation_path)) if len(relaxation_path) else endpoints[1:]
        # The trajectory captures actual optimizer steps; retain the final saved state if needed.
        if len(path_points) and not np.allclose(path_points[-1], endpoints[-1]):
            path_points = np.vstack((path_points, endpoints[-1]))
        if anomaly_key == "zero_jaccard_control_near_0p1_percent":
            # Match the presentation basin-map window for direct comparison.
            x = np.linspace(0.0, 8.0, grid)
            y = np.linspace(-8.0, 8.0, grid)
        else:
            extent = np.maximum(np.max(np.abs(np.vstack((endpoints, path_points))), axis=0) * 1.25, .08)
            x = np.linspace(-.2 * extent[0], extent[0], grid)
            y = np.linspace(-extent[1], extent[1], grid)
        energies = np.full((grid, grid), np.nan)
        base = calculator_for(row, reference, mace_mh_head)
        reference_energy = base.get_potential_energy()
        for iy, yy in enumerate(y):
            for ix, xx in enumerate(x):
                atoms = reference.copy(); atoms.positions = reference.positions + (xx * u1 + yy * u2).reshape((-1, 3)); atoms.calc = base.calc
                try: energies[iy, ix] = atoms.get_potential_energy() - reference_energy
                except Exception: pass
        finite = energies[np.isfinite(energies)]
        if not finite.size:
            save_placeholder(destination, "Local MLFF basin", "The MLFF returned no finite energies on the selected structural slice.")
            return
        cap = np.percentile(finite, 92); display = np.minimum(energies, cap)
        xx, yy = np.meshgrid(x, y)
        fig, ax = plt.subplots(figsize=(7.4, 6.4), constrained_layout=True)
        contour = ax.contourf(xx, yy, display, levels=128, cmap="magma")
        ax.contour(xx, yy, display, levels=8, colors="white", alpha=.16, linewidths=.35)
        ax.plot(endpoints[:2, 0], endpoints[:2, 1], color="#A9A9A9", alpha=.72, linewidth=.9, linestyle="--", zorder=3, label="attack displacement")
        ax.plot(path_points[:, 0], path_points[:, 1], color="#C7C7C7", alpha=.9, linewidth=1.15, zorder=4, label="relaxation trajectory")
        for point, label, marker, color in zip(endpoints, ("reference", "perturbed", "final relaxed"), ("o", "^", "s"), ("#35B7EB", "#FFB000", "#00A878")):
            ax.scatter(*point, marker=marker, s=68, color=color, edgecolor="white", linewidth=.85, zorder=5, label=label)
        ax.set(title=f"Local MLFF basin at {anomaly_label}: {material}", xlabel=r"Attack direction ($\AA\sqrt{\mathrm{atoms}}$)", ylabel=r"Orthogonal relaxation direction ($\AA\sqrt{\mathrm{atoms}}$)")
        ax.tick_params(labelsize=13, width=.85, length=4)
        ax.xaxis.label.set_size(15); ax.yaxis.label.set_size(15); ax.title.set_size(17)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_linewidth(.7)
        ax.legend(fontsize=12, loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3, borderaxespad=0, frameon=False)
        tick_step = 25 if cap <= 250 else 50
        colorbar_ticks = np.arange(0, cap + tick_step, tick_step)
        colorbar = fig.colorbar(contour, ax=ax, label="Relative energy (eV)", fraction=.035, pad=.035, aspect=34, ticks=colorbar_ticks)
        colorbar.ax.tick_params(labelsize=13, width=.7, length=3)
        colorbar.set_label("Relative energy (eV)", fontsize=15)
        colorbar.outline.set_linewidth(.6)
        fig.savefig(destination, dpi=300, bbox_inches="tight")
        np.savez_compressed(output / f"basin_map_data_{anomaly_key}.npz", x=x, y=y, energy=energies, endpoints=endpoints, relaxation_path=path_points)

        if anomaly_key == "mlff_jaccard_dip":
            low_points, dft_point = comprehensive_basin_overlay(data, row, reference, u1, u2)
            if len(low_points):
                ax.scatter(low_points[:, 0], low_points[:, 1], s=19, color="#38BDF8", alpha=.45,
                           linewidth=0, zorder=5, label=r"low-$\epsilon$ final states")
            if len(low_points) >= 4:
                q25, q75 = np.quantile(low_points, (.25, .75), axis=0)
                iqr_box = Rectangle(q25, *(q75 - q25), facecolor="#38BDF8", edgecolor="#0284C7",
                                    linewidth=1.1, linestyle="--", alpha=.18, zorder=4,
                                    label=rf"central 50% ($\epsilon<10\%$, n={len(low_points)})")
                ax.add_patch(iqr_box)
            if dft_point is not None:
                ax.scatter(*dft_point, marker="s", s=74, color="#7C3AED", edgecolor="white",
                           linewidth=.9, zorder=7, label="DFT final")
            else:
                ax.text(.98, .02, "Matched DFT final unavailable", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=8, color="white",
                        bbox={"facecolor": "#111827", "alpha": .7, "edgecolor": "none"})
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            ax.legend(fontsize=9, loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3,
                      borderaxespad=0, frameon=False)
            ax.set_title(f"Comprehensive local MLFF basin at {anomaly_label}: {material}")
            fig.savefig(output / "03_basin_map_comprehensive.png", dpi=300, bbox_inches="tight")
            np.savez_compressed(output / "basin_map_data_comprehensive.npz", x=x, y=y, energy=energies,
                                endpoints=endpoints, relaxation_path=path_points,
                                low_epsilon_finals=low_points,
                                dft_final=np.asarray(dft_point) if dft_point is not None else np.empty((0, 2)))
        plt.close(fig)
    except Exception as error:
        (output / f"basin_map_error_{anomaly_key}.txt").write_text(f"PES skipped: {error}\n", encoding="utf-8")
        save_placeholder(destination, "Local MLFF basin", f"MLFF basin evaluation failed: {error}")

def save_phonons(data, material, preference, output, mace_mh_head, anomaly_key, anomaly_label, diagnostic_model, require_zero_jaccard=False):
    """Render phonons for one selected anomaly, or a clear placeholder."""
    destination = output / f"04_phonon_stability_{anomaly_key}.png"
    rows = []
    if not material:
        pd.DataFrame(rows).to_csv(output / f"phonon_modes_{anomaly_key}.csv", index=False)
        save_placeholder(destination, "Local phonon stability", f"No material contributor was available for {anomaly_label}.")
        return
    row = source_mlff_row(data, material, preference, diagnostic_model, require_zero_jaccard)
    if row is None:
        pd.DataFrame(rows).to_csv(output / f"phonon_modes_{anomaly_key}.csv", index=False)
        save_placeholder(destination, "Local phonon stability", "The selected material has no matching MLFF record at the Jaccard threshold.")
        return
    if not bool(row.get("converged")):
        pd.DataFrame(rows).to_csv(output / f"phonon_modes_{anomaly_key}.csv", index=False)
        save_placeholder(destination, "Local phonon stability", "The selected final structure did not converge, so a local phonon interpretation would be unreliable.")
        return
    try:
        reference, _, final = load_states(row)
        for state_name, atoms in (("reference", reference), ("final", final)):
            configured = calculator_for(row, atoms, mace_mh_head)
            cache = output / "phonon_cache" / f"{anomaly_key}_{material}_{row['calculator']}_{state_name}"
            cache.parent.mkdir(exist_ok=True)
            phonons = Phonons(configured, configured.calc, supercell=(1, 1, 1), delta=.01, name=str(cache))
            phonons.run(); phonons.read(acoustic=True)
            frequencies, modes = phonons.band_structure(np.array([[0., 0., 0.]]), modes=True, verbose=False)
            for index, value in enumerate(frequencies[0] * 1000.0):
                rows.append({"material_slug": material, "calculator": row["calculator"], "state": state_name, "mode": index + 1, "frequency_meV": value})
    except Exception as error:
        rows.append({"material_slug": material, "error": str(error)})
    table = pd.DataFrame(rows); table.to_csv(output / f"phonon_modes_{anomaly_key}.csv", index=False)
    usable = table.dropna(subset=["frequency_meV"]) if "frequency_meV" in table else pd.DataFrame()
    if usable.empty:
        reason = table.get("error", pd.Series(["No usable phonon frequencies were produced."])).dropna().iloc[0]
        save_placeholder(destination, "Local phonon stability", f"Phonon diagnostic unavailable: {reason}")
        return
    fig, ax = plt.subplots(figsize=(5.4, 5.4), constrained_layout=True, facecolor="white")
    ax.set_facecolor("white")
    for state, group in usable.groupby("state"):
        group = group.sort_values("mode")
        modes = group["mode"].to_numpy(); frequencies = group["frequency_meV"].to_numpy()
        if len(modes) > 2:
            smooth_modes = np.linspace(modes.min(), modes.max(), 300)
            spline_order = min(3, len(modes) - 1)
            smooth_frequencies = make_interp_spline(modes, frequencies, k=spline_order)(smooth_modes)
            ax.plot(smooth_modes, smooth_frequencies, linewidth=1.8, label=state)
        else:
            ax.plot(modes, frequencies, linewidth=1.8, label=state)
    ax.set(xlabel=r"$\Gamma$-point mode index", ylabel="Signed phonon energy (meV)", title=f"Local phonon stability at {anomaly_label}: {material}")
    ax.legend(fontsize=8)
    fig.savefig(destination, dpi=300, bbox_inches="tight"); plt.close(fig)

def phonon_ensemble_rows(data, preference, diagnostic_model):
    """Choose the best available seed for every material in the epsilon sweep."""
    if preference is None or preference.empty:
        return pd.DataFrame()
    calculator = diagnostic_model or preference.get("calculator")
    rows = data[
        (data["calculator"] == calculator)
        & (data["attack_label"] == preference.get("attack_label"))
        & data["material_slug"].notna()
        & data["transition_epsilon"].notna()
        & data["converged"]
    ].copy()
    rows = rows[rows["transition_epsilon"].apply(
        lambda value: np.isclose(value, ENSEMBLE_EPSILONS).any()
    )]
    rows["_seed_key"] = jaccard_trajectory_key(rows)
    rows = rows[rows.apply(
        lambda candidate: all(path is not None for path in structural_artifacts(candidate)),
        axis=1,
    )]
    if rows.empty:
        return rows
    selected = []
    for _, material_rows in rows.groupby("material_slug", sort=True):
        coverage = material_rows.groupby("_seed_key")["transition_epsilon"].nunique()
        seed = coverage.sort_values(ascending=False, kind="stable").index[0]
        selected.append(
            material_rows[material_rows["_seed_key"] == seed].drop_duplicates(
                "transition_epsilon", keep="first"
            )
        )
    return pd.concat(selected, ignore_index=True).sort_values(
        ["transition_epsilon", "material_slug"]
    )


def gamma_phonon_modes(row, atoms, output, cache_key, mace_mh_head):
    """Calculate signed Gamma-point phonon energies in meV."""
    configured = calculator_for(row, atoms, mace_mh_head)
    cache = output / "phonon_cache" / cache_key
    cache.parent.mkdir(exist_ok=True)
    phonons = Phonons(
        configured, configured.calc, supercell=(1, 1, 1), delta=.01, name=str(cache)
    )
    phonons.run()
    phonons.read(acoustic=True)
    frequencies, _ = phonons.band_structure(
        np.array([[0., 0., 0.]]), modes=True, verbose=False
    )
    return frequencies[0] * 1000.0


def _smooth_curve(modes, values, points=300):
    modes = np.asarray(modes, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(modes) < 2:
        return modes, values
    smooth_modes = np.linspace(modes.min(), modes.max(), points)
    return smooth_modes, make_interp_spline(
        modes, values, k=min(3, len(modes) - 1)
    )(smooth_modes)


def save_phonon_ensemble(data, preference, output, mace_mh_head, diagnostic_model):
    """Plot central phonon distributions with visible epsilon-wise medians."""
    destination = output / "05_phonon_ensemble_epsilon_bundle.png"
    table_path = output / "phonon_modes_ensemble_epsilon_bundle.csv"
    records = []
    selected = phonon_ensemble_rows(data, preference, diagnostic_model)
    if selected.empty:
        pd.DataFrame(records).to_csv(table_path, index=False)
        save_placeholder(destination, "Phonon ensemble across epsilon", "No usable converged epsilon-sweep structures were available.")
        return
    for material, material_rows in selected.groupby("material_slug", sort=True):
        reference_row = material_rows.iloc[0]
        try:
            reference_atoms, _, _ = load_states(reference_row)
            frequencies = gamma_phonon_modes(
                reference_row, reference_atoms, output,
                f"05_ensemble_{material}_{reference_row['calculator']}_reference",
                mace_mh_head,
            )
            records.extend({"material_slug": material, "state": "reference", "mode": mode, "frequency_meV": frequency} for mode, frequency in enumerate(frequencies, start=1))
        except Exception as error:
            records.append({"material_slug": material, "state": "reference", "error": str(error)})
    for _, row in selected.iterrows():
        try:
            _, _, final = load_states(row)
            frequencies = gamma_phonon_modes(
                row, final, output,
                f"05_ensemble_{row['material_slug']}_{row['calculator']}_epsilon_{row['transition_epsilon']:.8g}",
                mace_mh_head,
            )
            records.extend({"material_slug": row["material_slug"], "state": "final", "mode": mode, "frequency_meV": frequency, "epsilon": row["transition_epsilon"], "seed": row["_seed_key"]} for mode, frequency in enumerate(frequencies, start=1))
        except Exception as error:
            records.append({"material_slug": row["material_slug"], "state": "final", "epsilon": row["transition_epsilon"], "error": str(error)})
    table = pd.DataFrame(records)
    table.to_csv(table_path, index=False)
    usable = table.dropna(subset=["frequency_meV"]) if "frequency_meV" in table else pd.DataFrame()
    finals = usable[usable["state"] == "final"] if not usable.empty else pd.DataFrame()
    reference = usable[usable["state"] == "reference"] if not usable.empty else pd.DataFrame()
    if finals.empty or reference.empty:
        save_placeholder(destination, "Phonon ensemble across epsilon", "Available phonon calculations did not produce both reference and final modes.")
        return
    # The display scale intentionally stops at 10^2: this makes the requested
    # 10^-2 -> 10^2 progression readable and renders any larger strength with
    # the same darkest colour instead of expanding the useful part of the map.
    norm = LogNorm(vmin=1e-2, vmax=1e2, clip=True)
    cmap = plt.colormaps["viridis_r"]  # light at 10^-2, dark at 10^2
    fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True, facecolor="white")
    ax.set_facecolor("white")
    plotted_strengths = []
    for epsilon in ENSEMBLE_EPSILONS:
        group = finals[np.isclose(finals["epsilon"], epsilon)]
        # The former IQR (25th--75th percentile) hid small epsilon-dependent
        # shifts under wide, overlapping bands. Use a central 20% ribbon and
        # draw its exact median, without spline interpolation, so differences
        # between strengths remain inspectable mode by mode.
        summary = group.groupby("mode")["frequency_meV"].quantile([.40, .50, .60]).unstack().dropna().sort_index()
        if summary.empty:
            continue
        color = cmap(norm(epsilon))
        ax.fill_between(summary.index, summary[.40], summary[.60], color=color, alpha=.16, linewidth=0, zorder=2)
        ax.plot(summary.index, summary[.50], color=color, linewidth=.85, alpha=.95, zorder=3)
        plotted_strengths.append(epsilon)
    summary = reference.groupby("mode")["frequency_meV"].median().sort_index()
    ax.plot(summary.index, summary.to_numpy(), color="black", linewidth=2.4, label="reference", zorder=5)
    ax.axhline(0, color="#555", linewidth=.75, alpha=.55, zorder=1)
    ax.set(xlabel=r"$\Gamma$-point mode index", ylabel="Signed phonon energy (meV)", title=f"Phonon ensemble across attack strengths ({selected['material_slug'].nunique()} available materials)")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=.02)
    colorbar.set_label(r"$\epsilon$ strength (% min. lattice parameter)")
    colorbar.set_ticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
    colorbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    colorbar.ax.yaxis.set_minor_locator(NullLocator())
    ax.legend(frameon=False, loc="best")
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pes-grid", type=int, default=31)
    parser.add_argument(
        "--mace-mh-head",
        default=DEFAULT_MACE_MH_HEAD,
        help="MACE-MH-1 head to use when combined records omit mace_head.",
    )
    parser.add_argument("--with-pes", action="store_true", help="Generate the optional MLFF basin map (loads a calculator).")
    parser.add_argument("--with-phonons", action="store_true", help="Generate optional phonon diagnostics (loads a calculator).")
    parser.add_argument("--diagnostic-model", choices=MLFF_MODELS, default=None, help="Prefer this MLFF's rows for calculator-backed diagnostics.")
    parser.add_argument("--skip-phonons", action="store_true", help="Deprecated compatibility option; phonons are already off by default.")
    args = parser.parse_args()
    root = args.project_root.resolve(); output = (args.output_dir or root / "why_plots").resolve(); output.mkdir(parents=True, exist_ok=True)
    data = load_records(root); data.to_csv(output / "why_plot_records.csv", index=False)
    save_final_rms_force_heatmap(data, output)
    dip, spike = choose_anomalies(data); contributions = contribution_rows(data, dip, spike)
    contributions.to_csv(output / "anomaly_contributions.csv", index=False)
    save_attribution(data, contributions, output)
    case_selection = select_cases(data, contributions, dip, spike)
    zero_jaccard_control = choose_zero_jaccard_control(data)
    if not zero_jaccard_control.empty:
        case_selection["Zero-Jaccard control (~1%)"] = {
            "material_slug": zero_jaccard_control["material_slug"],
            "epsilon_percent": float(zero_jaccard_control["epsilon_percent"]),
            "final_jaccard": float(zero_jaccard_control["final_jaccard"]),
        }
    case_materials = list(dict.fromkeys(
        selection["material_slug"] for selection in case_selection.values()
    ))
    (output / "analysis_selection.json").write_text(json.dumps({"jaccard_dip": dip.to_dict(), "jaccard_spike_after_10_percent": spike.to_dict(), "zero_jaccard_control_near_0p1_percent": zero_jaccard_control.to_dict(), "case_selection": case_selection, "case_materials": case_materials}, indent=2, default=str), encoding="utf-8")
    save_anomaly_transitions(data, dip, spike, output)
    diagnostic_cases = (("mlff_jaccard_dip", "Jaccard dip", dip, False), ("mlff_jaccard_spike_after_10_percent", "Jaccard spike (>10%)", spike, False), ("zero_jaccard_control_near_0p1_percent", "zero-Jaccard control (~1% displacement)", zero_jaccard_control, True))
    for anomaly_key, anomaly_label, preference, require_zero_jaccard in diagnostic_cases:
        material = case_selection.get(anomaly_label, {}).get("material_slug")
        if require_zero_jaccard:
            material = case_selection.get("Zero-Jaccard control (~1%)", {}).get("material_slug")
        if args.with_pes:
            save_pes(data, material, preference, output, args.pes_grid, args.mace_mh_head, anomaly_key, anomaly_label, args.diagnostic_model, require_zero_jaccard)
        if args.with_phonons and not args.skip_phonons:
            save_phonons(data, material, preference, output, args.mace_mh_head, anomaly_key, anomaly_label, args.diagnostic_model, require_zero_jaccard)

    if args.with_phonons and not args.skip_phonons:
        save_phonon_ensemble(data, spike, output, args.mace_mh_head, args.diagnostic_model)
    print(f"why_plots written to {output}")


if __name__ == "__main__":
    main()
