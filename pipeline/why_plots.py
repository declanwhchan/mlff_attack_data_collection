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
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from ase.io import read as ase_read
from ase.phonons import Phonons

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
    if "post_attack_relaxed_rms_force_ev_a" not in data:
        data = add_post_attack_rms_columns(data)
    data["final_rms_force"] = numeric(data, "post_attack_relaxed_rms_force_ev_a")
    data["final_energy"] = numeric(data, "final_energy")
    data["relax_steps"] = numeric(data, "after_relax_steps")
    data["converged"] = data.get("after_relax_converged", False).fillna(False).astype(bool)
    return data


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
    """Select DFT force/Jaccard rises and the largest MLFF Jaccard drop."""
    def adjacent_change(curve, direction):
        candidates = []
        for _, group in curve.groupby(["calculator", "attack_label"]):
            group = group.sort_values("transition_epsilon").copy()
            group["epsilon"] = group["transition_epsilon"]
            group["previous_epsilon"] = group["transition_epsilon"].shift()
            group["change"] = group["value"].diff()
            valid = group.dropna(subset=["previous_epsilon", "change"])
            if not valid.empty:
                candidates.append(valid.loc[valid["change"].idxmax() if direction == "max" else valid["change"].idxmin()])
        return (max(candidates, key=lambda row: row["change"]) if direction == "max" else min(candidates, key=lambda row: row["change"])) if candidates else pd.Series(dtype=object)

    force = seed_median_curve(data, "final_rms_force", ("dft_mace_mh", "dft_uma"))
    if force.empty:
        force = seed_median_curve(data, "final_rms_force", MLFF_MODELS)
    spike = adjacent_change(force, "max")
    dip = adjacent_change(seed_median_curve(data, "final_jaccard", MLFF_MODELS), "min")
    dft_jaccard_spike = adjacent_change(seed_median_curve(data, "final_jaccard", ("dft_mace_mh", "dft_uma")), "max")
    return spike, dip, dft_jaccard_spike


def contribution_rows(data, spike, dip, dft_jaccard_spike):
    """Rank material-level changes at the two selected aggregate transitions.

    Prefer within-seed pairs.  Legacy combined files can label the same seeds
    differently at adjacent epsilons; in that case compare the two material
    seed-median distributions instead of suppressing the attribution entirely.
    """
    parts = []
    for anomaly, metric, kind in ((spike, "final_rms_force", "Force spike"), (dip, "final_jaccard", "Jaccard dip"), (dft_jaccard_spike, "final_jaccard", "DFT Jaccard spike")):
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
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, kind, ascending, color in zip(
        axes, ("Force spike", "Jaccard dip", "DFT Jaccard spike"), (False, True, False), ("#4C78A8", "#F58518", "#54A24B")
    ):
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
        ax.set_xlabel("Δ final RMS force (eV/Å)" if kind == "Force spike" else "Δ final neighbor-Jaccard distance")
        ax.grid(axis="x", alpha=.25)
    fig.savefig(output / "01_material_attribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_cases(data, contributions, spike, dip, dft_jaccard_spike):
    """Choose and label materials for optional local diagnostics.

    A material can be selected for both anomalies. Preserve those roles even
    when the material names are later de-duplicated.
    """
    selected = {}
    preferences = {"Force spike": spike, "Jaccard dip": dip, "DFT Jaccard spike": dft_jaccard_spike}
    for kind, ascending in (("Force spike", False), ("Jaccard dip", True), ("DFT Jaccard spike", False)):
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


def source_mlff_row(data, material, preference):
    """Select a diagnostic-ready MLFF record at the selected transition."""
    subset = data[(data["material_slug"] == material) & data["calculator"].isin(MLFF_MODELS)].copy()
    if subset.empty or preference is None or preference.empty:
        return None
    subset = subset[
        (subset["attack_label"] == preference.get("attack_label"))
        & np.isclose(subset["transition_epsilon"], preference.get("epsilon", np.nan))
    ]
    if subset.empty:
        return None
    subset = subset.copy()
    subset["_states_available"] = subset.apply(
        lambda row: all(path is not None for path in structural_artifacts(row)), axis=1
    )
    ready = subset[subset["_states_available"]]
    if ready.empty:
        return None
    return ready.sort_values(["converged", "final_jaccard"], ascending=[False, False]).iloc[0]


def load_states(row):
    traj, perturbed, final = structural_artifacts(row)
    if not traj or not perturbed or not final:
        raise FileNotFoundError("Missing reference, perturbed, or final structural artifact")
    return ase_read(traj, index=-1), ase_read(perturbed), ase_read(final, index=-1)


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


def save_anomaly_transitions(data, spike, dip, dft_jaccard_spike, output):
    """Show only the two threshold samples that define the aggregate anomalies."""
    specifications = (
        (spike, "final_rms_force", "Final RMS force (eV/A)", "DFT force threshold", "#4C78A8"),
        (dip, "final_jaccard", "Final neighbor-Jaccard distance", "MLFF topology threshold", "#F58518"),
        (dip, "final_max_cn", "Max coordination-number change", "CN change at MLFF topology threshold", "#54A24B"),
        (dft_jaccard_spike, "final_jaccard", "Final neighbor-Jaccard distance", "DFT topology threshold", "#7A7A7A"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    for ax, (anomaly, metric, ylabel, title, color) in zip(axes, specifications):
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

    return setup_calculator(
        atoms.copy(),
        model_path,
        device=str(row.get("device", "cpu")),
        dtype_str=str(row.get("dtype_str", "float64")),
        seed=int(float(row.get("seed", 42))),
        calculator=backend,
        mace_head=recorded_head,
        uma_task=None if pd.isna(row.get("uma_task")) else row.get("uma_task"),
        uma_charge=None if pd.isna(row.get("uma_charge")) else int(row.get("uma_charge")),
        uma_spin=None if pd.isna(row.get("uma_spin")) else int(row.get("uma_spin")),
    )


def save_pes(data, material, preference, output, grid, mace_mh_head, anomaly_key, anomaly_label):
    """Render a 2-D MLFF energy landscape for one selected anomaly."""
    destination = output / f"03_basin_map_2d_{anomaly_key}.png"
    if not material:
        save_placeholder(destination, "Local MLFF basin", f"No material contributor was available for {anomaly_label}.")
        return
    row = source_mlff_row(data, material, preference)
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
        endpoints = np.array([[0, 0], [np.dot(d1, u1), np.dot((perturbed.positions-reference.positions).reshape(-1), u2)], [np.dot((final.positions-reference.positions).reshape(-1), u1), np.dot((final.positions-reference.positions).reshape(-1), u2)]])
        extent = np.maximum(np.max(np.abs(endpoints), axis=0) * 1.25, .08)
        x = np.linspace(-.2 * extent[0], extent[0], grid); y = np.linspace(-extent[1], extent[1], grid)
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
        fig, ax = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
        contour = ax.contourf(xx, yy, display, levels=24, cmap="magma")
        ax.contour(xx, yy, display, levels=10, colors="white", alpha=.32, linewidths=.5)
        ax.plot(endpoints[:, 0], endpoints[:, 1], color="white", alpha=.8, linewidth=1.8, zorder=4, label="structural path")
        for point, label, marker, color in zip(endpoints, ("reference", "perturbed", "final relaxed"), ("o", "^", "s"), ("#35B7EB", "#FFB000", "#00A878")):
            ax.scatter(*point, marker=marker, s=92, color=color, edgecolor="white", linewidth=1.25, zorder=5, label=label)
        ax.set(title=f"Local MLFF basin at {anomaly_label}: {material}", xlabel="Attack direction (A sqrt(atoms))", ylabel="Orthogonal relaxation direction (A sqrt(atoms))")
        ax.legend(fontsize=8, loc="upper right"); fig.colorbar(contour, ax=ax, label="Relative MLFF energy (eV; capped)")
        fig.savefig(destination, dpi=300, bbox_inches="tight"); plt.close(fig)
        np.savez_compressed(output / f"basin_map_data_{anomaly_key}.npz", x=x, y=y, energy=energies, endpoints=endpoints)
    except Exception as error:
        (output / f"basin_map_error_{anomaly_key}.txt").write_text(f"PES skipped: {error}\n", encoding="utf-8")
        save_placeholder(destination, "Local MLFF basin", f"MLFF basin evaluation failed: {error}")

def save_phonons(data, material, preference, output, mace_mh_head, anomaly_key, anomaly_label):
    """Render phonons for one selected anomaly, or a clear placeholder."""
    destination = output / f"04_phonon_stability_{anomaly_key}.png"
    rows = []
    if not material:
        pd.DataFrame(rows).to_csv(output / f"phonon_modes_{anomaly_key}.csv", index=False)
        save_placeholder(destination, "Local phonon stability", f"No material contributor was available for {anomaly_label}.")
        return
    row = source_mlff_row(data, material, preference)
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
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for state, group in usable.groupby("state"):
        ax.plot(group["mode"], group["frequency_meV"], marker="o", markersize=3.5, linewidth=1.5, label=state)
    ax.axhline(0, color="#222", linewidth=.8); ax.set(xlabel="Gamma-point mode index", ylabel="Signed phonon energy (meV)", title=f"Local phonon stability at {anomaly_label}: {material}")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(destination, dpi=300, bbox_inches="tight"); plt.close(fig)

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
    parser.add_argument("--skip-phonons", action="store_true", help="Deprecated compatibility option; phonons are already off by default.")
    args = parser.parse_args()
    root = args.project_root.resolve(); output = (args.output_dir or root / "why_plots").resolve(); output.mkdir(parents=True, exist_ok=True)
    data = load_records(root); data.to_csv(output / "why_plot_records.csv", index=False)
    spike, dip, dft_jaccard_spike = choose_anomalies(data); contributions = contribution_rows(data, spike, dip, dft_jaccard_spike)
    contributions.to_csv(output / "anomaly_contributions.csv", index=False)
    save_attribution(data, contributions, output)
    case_selection = select_cases(data, contributions, spike, dip, dft_jaccard_spike)
    case_materials = list(dict.fromkeys(
        selection["material_slug"] for selection in case_selection.values()
    ))
    (output / "analysis_selection.json").write_text(json.dumps({"force_spike": spike.to_dict(), "jaccard_dip": dip.to_dict(), "dft_jaccard_spike": dft_jaccard_spike.to_dict(), "case_selection": case_selection, "case_materials": case_materials}, indent=2, default=str), encoding="utf-8")
    save_anomaly_transitions(data, spike, dip, dft_jaccard_spike, output)
    diagnostic_cases = (
        ("force_spike", "Force spike", spike),
        ("mlff_jaccard_dip", "Jaccard dip", dip),
        ("dft_jaccard_spike", "DFT Jaccard spike", dft_jaccard_spike),
    )
    for anomaly_key, anomaly_label, preference in diagnostic_cases:
        material = case_selection.get(anomaly_label, {}).get("material_slug")
        if args.with_pes:
            save_pes(data, material, preference, output, args.pes_grid, args.mace_mh_head, anomaly_key, anomaly_label)
        if args.with_phonons and not args.skip_phonons:
            save_phonons(data, material, preference, output, args.mace_mh_head, anomaly_key, anomaly_label)
    print(f"why_plots written to {output}")


if __name__ == "__main__":
    main()
