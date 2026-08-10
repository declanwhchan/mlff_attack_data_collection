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
from run_tests import neighbor_edge_set, setup_calculator


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
    data["final_jaccard"] = numeric(data, FINAL_JACCARD)
    if data["final_jaccard"].isna().all():
        data["final_jaccard"] = numeric(data, "neighbor_jaccard_distance")
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
    usable = data.dropna(subset=["material_slug", "epsilon_percent", metric]).copy()
    if models is not None:
        usable = usable[usable["calculator"].isin(models)]
    seed_key = usable.get("seed", pd.Series(np.nan, index=usable.index)).astype("string")
    trial_key = usable.get("trial", pd.Series(np.nan, index=usable.index)).astype("string")
    usable["_seed_key"] = seed_key.fillna(trial_key).fillna("unlabelled")
    return usable.groupby(
        ["calculator", "attack_label", "_seed_key", "material_slug", "epsilon_percent"],
        as_index=False,
    )[metric].median()


def seed_median_curve(data, metric, models):
    """Typical material response: material median within seed, then seed median."""
    per_material = material_seed_medians(data, metric, models)
    return per_material.groupby(
        ["calculator", "attack_label", "_seed_key", "epsilon_percent"], as_index=False,
    )[metric].median().groupby(
        ["calculator", "attack_label", "epsilon_percent"], as_index=False,
    ).agg(value=(metric, "median"), seed_count=("_seed_key", "nunique"))


def choose_anomalies(data):
    """Select the largest DFT force rise and the largest MLFF Jaccard drop."""
    def adjacent_change(curve, direction):
        candidates = []
        for _, group in curve.groupby(["calculator", "attack_label"]):
            group = group.sort_values("epsilon_percent").copy()
            group["previous_epsilon"] = group["epsilon_percent"].shift()
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
    return spike, dip


def contribution_rows(data, spike, dip):
    parts = []
    for anomaly, metric, kind in ((spike, "final_rms_force", "Force spike"), (dip, "final_jaccard", "Jaccard dip")):
        if anomaly.empty or pd.isna(anomaly.get("previous_epsilon")):
            continue
        subset = data[(data["calculator"] == anomaly["calculator"]) & (data["attack_label"] == anomaly["attack_label"])]
        values = material_seed_medians(subset, metric)
        current = values[np.isclose(values["epsilon_percent"], anomaly["epsilon_percent"])]
        previous = values[np.isclose(values["epsilon_percent"], anomaly["previous_epsilon"])]
        paired = current.merge(previous, on=["material_slug", "_seed_key"], suffixes=("_current", "_previous"))
        if paired.empty:
            continue
        paired["delta"] = paired[f"{metric}_current"] - paired[f"{metric}_previous"]
        score = paired.groupby("material_slug", as_index=False)["delta"].median()
        score["kind"] = kind
        score["score"] = score["delta"]
        parts.append(score[["material_slug", "kind", "score"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["material_slug", "kind", "score"])


def save_attribution(data, contributions, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, kind, ascending in zip(axes, ("Force spike", "Jaccard dip"), (False, True)):
        selected = contributions[contributions["kind"] == kind].sort_values("score", ascending=ascending)
        if selected.empty:
            ax.text(.5, .5, "No matched records", ha="center", va="center", transform=ax.transAxes)
            continue
        selected = selected.head(20).iloc[::-1]
        ax.barh(selected["material_slug"], selected["score"], color="#4C78A8")
        ax.set_title(kind, fontweight="bold")
        ax.set_xlabel("Median final RMS force (eV/Å)" if kind == "Force spike" else "Δ Jaccard to previous ε")
        ax.grid(axis="x", alpha=.25)
    fig.savefig(output / "01_material_attribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



def select_cases(data, contributions, spike, dip):
    chosen = []
    for kind, ascending in (("Force spike", False), ("Jaccard dip", True)):
        ranked = contributions[contributions["kind"] == kind].sort_values("score", ascending=ascending)
        for material in ranked["material_slug"]:
            if material not in chosen:
                chosen.append(material); break
    # A stable control is the material closest to median Jaccard at the dip.
    if not dip.empty:
        pool = data[(data["calculator"] == dip["calculator"]) & np.isclose(data["epsilon_percent"], dip["epsilon_percent"])]
        median = pool["final_jaccard"].median()
        control = (pool.groupby("material_slug")["final_jaccard"].median() - median).abs().sort_values().index
        for material in control:
            if material not in chosen:
                chosen.append(material); break
    return chosen[:3]


def source_mlff_row(data, material, preference):
    subset = data[(data["material_slug"] == material) & data["calculator"].isin(MLFF_MODELS)].copy()
    if subset.empty:
        return None
    if preference is not None and not preference.empty:
        match = subset[(subset["attack_label"] == preference.get("attack_label")) & np.isclose(subset["epsilon_percent"], preference.get("epsilon_percent", np.nan))]
        if not match.empty:
            return match.sort_values("converged", ascending=False).iloc[0]
    return subset.sort_values(["final_jaccard", "converged"], ascending=[False, False]).iloc[0]


def load_states(row):
    traj = existing_path(row, "before_relax_traj", fallback="before_attack_relaxation.traj")
    perturbed = existing_path(row, "output_cif", fallback="perturbed.cif")
    final = existing_path(row, "final_relaxed_cif", fallback="final_relaxed.cif")
    if not final:
        final = existing_path(row, "after_attack_relax_traj", fallback="after_attack_relaxation.traj")
    if not traj or not perturbed or not final:
        raise FileNotFoundError("Missing reference, perturbed, or final structural artifact")
    return ase_read(traj, index=-1), ase_read(perturbed), ase_read(final, index=-1)


def projected_coordinates(reference, states):
    coordinates = np.vstack([atoms.positions for atoms in states])
    center = reference.positions.mean(axis=0)
    _, _, vh = np.linalg.svd((coordinates - center).reshape(len(states), -1), full_matrices=False)
    basis = vh[:2]
    return np.array([np.dot((atoms.positions - reference.positions).reshape(-1), basis.T) for atoms in states]), basis


def save_structure_trajectory(data, cases, output):
    rows = []
    for material in cases:
        row = source_mlff_row(data, material, None)
        if row is None: continue
        try:
            reference, perturbed, final = load_states(row)
            points, _ = projected_coordinates(reference, (reference, perturbed, final))
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
            for ax, atoms, name in zip(axes, (reference, perturbed, final), ("Relaxed reference", "Perturbed", "Final relaxed")):
                displacement = np.linalg.norm(atoms.positions - reference.positions, axis=1)
                scatter = ax.scatter(atoms.positions[:, 0], atoms.positions[:, 1], c=displacement, cmap="viridis", s=28)
                ax.set_title(name); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(scatter, ax=axes, shrink=.75, label="Displacement from reference (Å)")
            fig.suptitle(f"{material}: {LABELS.get(row['calculator'], row['calculator'])}", fontweight="bold")
            fig.savefig(output / f"03_structure_transition_{material}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            rows.append({"material_slug": material, "run_id": row["run_id"], "calculator": row["calculator"], "attack_label": row.get("attack_label"), "epsilon_percent": row.get("epsilon_percent"), "final_jaccard": row.get("final_jaccard"), "converged": row.get("converged"), "projected_perturbed_x": points[1, 0], "projected_final_x": points[2, 0]})
        except Exception as error:
            rows.append({"material_slug": material, "error": str(error)})
    pd.DataFrame(rows).to_csv(output / "selected_case_studies.csv", index=False)


def calculator_for(row, atoms):
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

    return setup_calculator(
        atoms.copy(),
        model_path,
        device=str(row.get("device", "cpu")),
        dtype_str=str(row.get("dtype_str", "float64")),
        seed=int(float(row.get("seed", 42))),
        calculator=backend,
        mace_head=None if pd.isna(row.get("mace_head")) else row.get("mace_head"),
        uma_task=None if pd.isna(row.get("uma_task")) else row.get("uma_task"),
        uma_charge=None if pd.isna(row.get("uma_charge")) else int(row.get("uma_charge")),
        uma_spin=None if pd.isna(row.get("uma_spin")) else int(row.get("uma_spin")),
    )


def save_pes(data, cases, output, grid):
    # One strongest case only: a 2D contour is enough for the presentation.
    if not cases: return
    row = source_mlff_row(data, cases[0], None)
    if row is None: return
    try:
        reference, perturbed, final = load_states(row)
        d1 = (perturbed.positions - reference.positions).reshape(-1)
        d2 = (final.positions - reference.positions).reshape(-1)
        u1 = d1 / np.linalg.norm(d1)
        d2 -= np.dot(d2, u1) * u1
        if np.linalg.norm(d2) < 1e-8: return
        u2 = d2 / np.linalg.norm(d2)
        endpoints = np.array([[0, 0], [np.dot(d1, u1), np.dot((perturbed.positions-reference.positions).reshape(-1), u2)], [np.dot((final.positions-reference.positions).reshape(-1), u1), np.dot((final.positions-reference.positions).reshape(-1), u2)]])
        extent = np.maximum(np.max(np.abs(endpoints), axis=0) * 1.25, .08)
        x = np.linspace(-.2 * extent[0], extent[0], grid); y = np.linspace(-extent[1], extent[1], grid)
        energies = np.full((grid, grid), np.nan)
        base = calculator_for(row, reference)
        reference_energy = base.get_potential_energy()
        for iy, yy in enumerate(y):
            for ix, xx in enumerate(x):
                atoms = reference.copy(); atoms.positions = reference.positions + (xx * u1 + yy * u2).reshape((-1, 3)); atoms.calc = base.calc
                try: energies[iy, ix] = atoms.get_potential_energy() - reference_energy
                except Exception: pass
        finite = energies[np.isfinite(energies)]
        if not finite.size: return
        cap = np.percentile(finite, 92); display = np.minimum(energies, cap)
        xx, yy = np.meshgrid(x, y)
        fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
        contour = ax.contourf(xx, yy, display, levels=24, cmap="magma")
        ax.contour(xx, yy, display, levels=10, colors="white", alpha=.35, linewidths=.5)
        ax.plot(endpoints[:, 0], endpoints[:, 1], "wo-", markeredgecolor="#222", label="reference → perturbed → final")
        ax.set(title=f"Local MLFF basin: {cases[0]}", xlabel="Attack direction (Å·√atoms)", ylabel="Orthogonal relaxation direction (Å·√atoms)")
        ax.legend(fontsize=8); fig.colorbar(contour, ax=ax, label="Relative MLFF energy (eV; capped)")
        fig.savefig(output / "04_basin_map_2d.png", dpi=300, bbox_inches="tight"); plt.close(fig)
        np.savez_compressed(output / "basin_map_data.npz", x=x, y=y, energy=energies, endpoints=endpoints)
    except Exception as error:
        (output / "basin_map_error.txt").write_text(f"PES skipped: {error}\n", encoding="utf-8")


def save_phonons(data, cases, output):
    if not cases: return
    rows = []
    for material in cases[:2]:
        row = source_mlff_row(data, material, None)
        if row is None or not bool(row.get("converged")): continue
        try:
            reference, _, final = load_states(row)
            for state_name, atoms in (("reference", reference), ("final", final)):
                configured = calculator_for(row, atoms)
                cache = output / "phonon_cache" / f"{material}_{row['calculator']}_{state_name}"
                cache.parent.mkdir(exist_ok=True)
                phonons = Phonons(configured, configured.calc, supercell=(1, 1, 1), delta=.01, name=str(cache))
                phonons.run(); phonons.read(acoustic=True)
                frequencies, modes = phonons.band_structure(np.array([[0., 0., 0.]]), modes=True, verbose=False)
                frequencies = frequencies[0] * 1000.0  # meV, signed for imaginary modes
                for index, value in enumerate(frequencies): rows.append({"material_slug": material, "calculator": row["calculator"], "state": state_name, "mode": index + 1, "frequency_meV": value})
        except Exception as error:
            rows.append({"material_slug": material, "error": str(error)})
    table = pd.DataFrame(rows); table.to_csv(output / "phonon_modes.csv", index=False)
    usable = table.dropna(subset=["frequency_meV"]) if "frequency_meV" in table else pd.DataFrame()
    if usable.empty: return
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for (material, state), group in usable.groupby(["material_slug", "state"]):
        ax.plot(group["mode"], group["frequency_meV"], marker="o", markersize=2.5, linewidth=1, label=f"{material} — {state}")
    ax.axhline(0, color="#222", linewidth=.8); ax.set(xlabel="Γ-point mode index", ylabel="Signed phonon energy (meV)", title="Local phonon stability near the threshold")
    ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2)
    fig.savefig(output / "05_phonon_stability.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pes-grid", type=int, default=31)
    parser.add_argument("--with-pes", action="store_true", help="Generate the optional MLFF basin map (loads a calculator).")
    parser.add_argument("--with-phonons", action="store_true", help="Generate optional phonon diagnostics (loads a calculator).")
    parser.add_argument("--skip-phonons", action="store_true", help="Deprecated compatibility option; phonons are already off by default.")
    args = parser.parse_args()
    root = args.project_root.resolve(); output = (args.output_dir or root / "why_plots").resolve(); output.mkdir(parents=True, exist_ok=True)
    data = load_records(root); data.to_csv(output / "anomaly_contributions.csv", index=False)
    spike, dip = choose_anomalies(data); contributions = contribution_rows(data, spike, dip)
    save_attribution(data, contributions, output)
    cases = select_cases(data, contributions, spike, dip)
    (output / "analysis_selection.json").write_text(json.dumps({"force_spike": spike.to_dict(), "jaccard_dip": dip.to_dict(), "case_materials": cases}, indent=2, default=str), encoding="utf-8")
    save_structure_trajectory(data, cases, output)
    if args.with_pes:
        save_pes(data, cases, output, args.pes_grid)
    if args.with_phonons and not args.skip_phonons:
        save_phonons(data, cases, output)
    print(f"why_plots written to {output}")


if __name__ == "__main__":
    main()
