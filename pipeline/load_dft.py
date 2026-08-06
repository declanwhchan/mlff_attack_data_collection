#!/usr/bin/env python3
"""Load fixed-cell VASP DFT relaxations as comprehensive plot records.

Geometry metrics use the originating MLFF-relaxed structure as reference.
Per-atom DFT force changes use the first and final complete TOTAL-FORCE
blocks in OUTCAR; KPOINTS, OSZICAR, and harvest CSVs do not contain the
required per-atom force vectors.
"""

from io import StringIO
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from ase.io import read as read_structure

from run_tests import (
    coordination_by_atom,
    edge_jaccard_distance,
    neighbor_edge_set,
    rdf_l1_distance,
)


DFT_SOURCE_MODELS = {
    "mace": "mace_mh",
    "mace_mh": "mace_mh",
    "uma": "uma",
    "chgnet": "chgnet",
}


def dft_coverage_table(records):
    """Summarize the DFT rows that are actually available to plotting."""
    data = pd.DataFrame(records)
    required = {"calculator", "attack_label", "epsilon", "material_slug"}
    if data.empty or not required.issubset(data.columns):
        return pd.DataFrame(columns=[
            "calculator",
            "attack_label",
            "epsilon",
            "records",
            "materials",
            "records_with_force_metrics",
        ])

    data = data[
        data["calculator"].fillna("").astype(str).str.startswith("dft_")
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=[
            "calculator",
            "attack_label",
            "epsilon",
            "records",
            "materials",
            "records_with_force_metrics",
        ])

    data["epsilon"] = pd.to_numeric(data["epsilon"], errors="coerce")
    force_blocks = pd.to_numeric(
        data.get(
            "dft_outcar_force_blocks",
            pd.Series(0, index=data.index),
        ),
        errors="coerce",
    ).fillna(0)
    data["_has_force_metrics"] = force_blocks.gt(0)

    return (
        data.groupby(
            ["calculator", "attack_label", "epsilon"],
            dropna=False,
            sort=True,
        )
        .agg(
            records=("calculator", "size"),
            materials=("material_slug", "nunique"),
            records_with_force_metrics=("_has_force_metrics", "sum"),
        )
        .reset_index()
    )


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return None if not text or text.lower() == "nan" else value


def _number(value, kind=float):
    value = _clean(value)
    if value is None:
        return None
    try:
        return kind(float(value))
    except (TypeError, ValueError):
        return None


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _manifest_path(root, value):
    """Resolve manifest paths written on either Windows or Linux."""
    value = _clean(value)
    if value is None:
        return None
    normalized = str(value).replace("\\", "/")
    path = Path(normalized)
    return path if path.is_absolute() else Path(root) / path


def read_dft_structure(path, index=-1):
    """Read normal structures and POSCAR text stored with a .cif suffix."""
    path = Path(path)
    try:
        return read_structure(path, index=index)
    except Exception as original_error:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            counts = lines[6].split()
            symbols = [token.rstrip("/") for token in lines[5].split()]
            valid = bool(counts) and all(int(token) >= 0 for token in counts)
        except Exception:
            raise original_error
        if not valid or len(symbols) != len(counts) or any(not x for x in symbols):
            raise original_error
        lines[5] = "  " + "  ".join(symbols)
        return read_structure(
            StringIO("\n".join(lines) + "\n"),
            format="vasp",
            index=index,
        )


def _normal_material(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _epsilon(token):
    token = str(token)
    return float(f"0.{token[1:]}") if token.startswith("0") else float(token)


def _parse_name(name):
    match = re.match(
        r"^(mace_mh|mace|uma|chgnet)_(.+)_"
        r"(fgsm|ifgsm|pgd)_eps([0-9]+)"
        r"(?:_steps([0-9]+))?_perturbed$",
        str(name).lower(),
    )
    if match is None:
        return None
    model, material, attack, epsilon_token, steps = match.groups()
    return {
        "source_model": DFT_SOURCE_MODELS[model],
        "material_slug": material,
        "attack_label": {
            "fgsm": "FGSM",
            "ifgsm": "I-FGSM",
            "pgd": "PGD",
        }[attack],
        "epsilon": _epsilon(epsilon_token),
        "n_steps": int(steps) if steps else None,
    }


def _change_metrics(reference, target):
    empty = {
        "displacements": np.asarray([], dtype=float),
        "neighbor_jaccard_distance": np.nan,
        "coordination_change_mean": np.nan,
        "coordination_change_max": np.nan,
        "rdf_l1_distance": np.nan,
    }
    if reference is None or target is None or len(reference) != len(target):
        return empty
    result = dict(empty)
    result["displacements"] = np.linalg.norm(
        target.positions - reference.positions,
        axis=1,
    )
    try:
        before_edges = neighbor_edge_set(reference)
        after_edges = neighbor_edge_set(target)
        before_cn = coordination_by_atom(before_edges, reference)
        after_cn = coordination_by_atom(after_edges, target)
        keys = set(before_cn) | set(after_cn)
        changes = np.asarray(
            [abs(after_cn.get(key, 0) - before_cn.get(key, 0)) for key in keys],
            dtype=float,
        )
        result.update({
            "neighbor_jaccard_distance": edge_jaccard_distance(
                before_edges,
                after_edges,
            ),
            "coordination_change_mean": (
                float(np.mean(changes)) if changes.size else 0.0
            ),
            "coordination_change_max": (
                float(np.max(changes)) if changes.size else 0.0
            ),
            "rdf_l1_distance": float(rdf_l1_distance(reference, target)),
        })
    except Exception:
        pass
    return result


def _write_atom_csv(structure, path, forces=None):
    if forces is None:
        forces = np.full((len(structure), 3), np.nan, dtype=float)
    forces = np.asarray(forces, dtype=float)
    if forces.shape != (len(structure), 3):
        raise ValueError(
            f"Force shape {forces.shape}; expected {(len(structure), 3)}"
        )
    pd.DataFrame({
        "atom_index": np.arange(len(structure), dtype=int),
        "x": structure.positions[:, 0],
        "y": structure.positions[:, 1],
        "z": structure.positions[:, 2],
        "fx": forces[:, 0],
        "fy": forces[:, 1],
        "fz": forces[:, 2],
    }).to_csv(path, index=False)


def read_outcar_force_blocks(path, expected_atoms):
    """Return the first/final complete VASP TOTAL-FORCE blocks."""
    blocks = []
    rows = None
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "TOTAL-FORCE (eV/Angst)" in line:
                rows = []
                continue
            if rows is None:
                continue
            fields = line.split()
            try:
                values = [float(value) for value in fields[:6]]
            except (ValueError, TypeError):
                # VASP writes a dashed separator between the header and
                # the first atom row. Keep waiting while the block is empty.
                if not rows:
                    continue
                if len(rows) == expected_atoms:
                    blocks.append(np.asarray(rows, dtype=float))
                rows = None
                continue
            if len(values) != 6:
                continue
            rows.append(values[3:6])
            if len(rows) == expected_atoms:
                blocks.append(np.asarray(rows, dtype=float))
                rows = None
    if not blocks:
        raise ValueError(f"No complete TOTAL-FORCE blocks in {path}")
    return blocks[0], blocks[-1], len(blocks)


def _outcar_path(root, row, relaxed_path, structure_name):
    candidates = []
    jobdir = _clean(row.get("jobdir"))
    if jobdir is not None:
        candidates.append(_manifest_path(root, jobdir) / "OUTCAR")
    candidates.extend([
        relaxed_path.parent / "OUTCAR",
        root / "vasp_relax_fixedcell" / structure_name / "OUTCAR",
    ])
    return next((path for path in candidates if path.is_file()), None)


def _source_parent(source, parsed):
    """Find the MLFF run that supplies the common relaxed reference.

    Exact attack/epsilon matches are preferred. The initial relaxation is
    independent of the later attack, so a same-material, same-model sweep
    row is a valid baseline fallback when an extension point has no exact
    row in the comprehensive MLFF summary.
    """
    candidates = source[
        source["calculator"].astype(str).eq(parsed["source_model"])
    ].copy()
    candidates = candidates[
        candidates["material_slug"].map(_normal_material)
        == _normal_material(parsed["material_slug"])
    ].copy()
    if candidates.empty:
        return None, None

    run_ids = candidates.get(
        "run_id",
        pd.Series("", index=candidates.index),
    ).fillna("").astype(str)
    non_step = ~run_ids.str.contains("_steps", regex=False)
    if non_step.any():
        candidates = candidates.loc[non_step].copy()

    attacks = candidates.get(
        "attack_label",
        pd.Series("", index=candidates.index),
    ).fillna("").astype(str)
    epsilons = pd.to_numeric(candidates.get("epsilon"), errors="coerce")
    attack_match = attacks.eq(parsed["attack_label"])
    epsilon_match = np.isclose(
        epsilons,
        parsed["epsilon"],
        rtol=1e-8,
        atol=1e-12,
        equal_nan=False,
    )

    if parsed["n_steps"] is None:
        step_match = np.ones(len(candidates), dtype=bool)
    else:
        candidate_steps = pd.to_numeric(
            candidates.get("n_steps"),
            errors="coerce",
        )
        step_match = candidate_steps.eq(parsed["n_steps"]).to_numpy()

    exact = candidates.loc[attack_match & epsilon_match & step_match]
    if not exact.empty:
        return exact.iloc[0].to_dict(), "exact"

    candidates["_attack_priority"] = (~attack_match).astype(int)
    positive = epsilons.where(epsilons > 0)
    if parsed["epsilon"] > 0:
        distance = np.abs(np.log10(positive) - math.log10(parsed["epsilon"]))
    else:
        distance = np.abs(epsilons - parsed["epsilon"])
    candidates["_epsilon_distance"] = distance.fillna(np.inf)
    candidates = candidates.sort_values(
        ["_attack_priority", "_epsilon_distance"],
        kind="stable",
    )
    return candidates.iloc[0].to_dict(), "baseline_fallback"


def _epsilon_percent(parent, epsilon):
    """Convert a manifest epsilon to percent of its lattice reference."""
    reference = _number(parent.get("epsilon_reference_length_a"))
    if reference is not None and reference > 0:
        return 100.0 * float(epsilon) / reference

    parent_epsilon = _number(parent.get("epsilon"))
    parent_percent = _number(parent.get("epsilon_percent_displacement"))
    if (
        parent_epsilon is not None
        and parent_epsilon != 0
        and parent_percent is not None
    ):
        return float(epsilon) * parent_percent / parent_epsilon
    return np.nan


def load_dft_records(
    dft_root,
    source_records,
    cache_root,
    baseline_path_resolver,
):
    """Convert clean fixed-cell DFT relaxations into normal plot records."""
    dft_root = Path(dft_root)
    combined_manifest = dft_root / "manifests" / "combined_manifest.csv"
    preliminary_manifest = dft_root / "manifests" / "preliminary_manifest.csv"
    manifest_path = (
        combined_manifest
        if combined_manifest.is_file()
        else preliminary_manifest
    )
    manifest = _read_csv(manifest_path)
    if manifest is None or manifest.empty:
        return [], [f"Missing or empty DFT manifest: {manifest_path}"]
    source = pd.DataFrame(source_records)
    if source.empty:
        return [], ["No MLFF records were available for DFT matching"]

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    records, missing = [], []

    for _, manifest_row in manifest.iterrows():
        if str(manifest_row.get("delivery_group", "")).upper() != "FORCE_CONVERGED_CLEAN":
            continue
        structure_name = str(manifest_row.get("structure_name", ""))
        parsed = _parse_name(structure_name)
        if parsed is None:
            missing.append(f"Unrecognized DFT structure name: {structure_name}")
            continue

        parent, source_match = _source_parent(source, parsed)
        if parent is None:
            missing.append(
                "No same-model/material MLFF baseline matched: "
                f"{structure_name}"
            )
            continue

        relaxed_value = _clean(manifest_row.get("relaxed_structure_file"))
        if relaxed_value is None:
            missing.append(f"No relaxed DFT structure path: {structure_name}")
            continue
        relaxed_path = _manifest_path(dft_root, relaxed_value)
        perturbed_path = relaxed_path.parent / "initial.cif"
        try:
            baseline = read_structure(baseline_path_resolver(parent), index=-1)
            perturbed = read_dft_structure(perturbed_path)
            relaxed = read_dft_structure(relaxed_path)
        except Exception as error:
            missing.append(f"Could not read DFT structures for {structure_name}: {error}")
            continue
        if not (len(baseline) == len(perturbed) == len(relaxed)):
            missing.append(f"DFT atom-count mismatch: {structure_name}")
            continue

        first_forces = final_forces = None
        block_count = 0
        outcar = _outcar_path(dft_root, manifest_row, relaxed_path, structure_name)
        if outcar is None:
            missing.append(f"No OUTCAR for DFT force metrics: {structure_name}")
        else:
            try:
                first_forces, final_forces, block_count = read_outcar_force_blocks(
                    outcar,
                    len(relaxed),
                )
            except Exception as error:
                missing.append(f"Could not read DFT forces for {structure_name}: {error}")

        dft_median_delta_force_after_relaxation = np.nan
        if first_forces is not None and final_forces is not None:
            force_delta = np.linalg.norm(
                np.asarray(final_forces, dtype=float)
                - np.asarray(first_forces, dtype=float),
                axis=1,
            )
            finite_force_delta = force_delta[np.isfinite(force_delta)]
            if finite_force_delta.size:
                dft_median_delta_force_after_relaxation = float(
                    np.median(finite_force_delta)
                )

        run_dir = cache_root / structure_name
        run_dir.mkdir(parents=True, exist_ok=True)
        # The package has no unperturbed DFT force evaluation. Therefore
        # final DFT delta force is final OUTCAR minus first OUTCAR block;
        # immediate DFT delta force remains unavailable rather than zero.
        _write_atom_csv(baseline, run_dir / "before_forces.csv", first_forces)
        _write_atom_csv(perturbed, run_dir / "perturbed_forces.csv", None)
        _write_atom_csv(relaxed, run_dir / "after_forces.csv", final_forces)

        immediate = _change_metrics(baseline, perturbed)
        final = _change_metrics(baseline, relaxed)
        model = f"dft_{parsed['source_model']}"
        record = dict(parent)
        record.update({
            "run_id": f"dft_{structure_name}",
            "logical_run_id": f"dft_{structure_name}",
            "calculator": model,
            "model_id": model,
            "attack_label": parsed["attack_label"],
            "epsilon": parsed["epsilon"],
            "epsilon_percent_displacement": _epsilon_percent(
                parent,
                parsed["epsilon"],
            ),
            "n_steps": parsed["n_steps"],
            "run_dir": str(run_dir),
            "input_path": str(perturbed_path),
            "after_relax_steps": _number(manifest_row.get("n_ionic_steps"), int),
            "after_relax_converged": True,
            "mean_displacement": (
                float(np.mean(final["displacements"]))
                if final["displacements"].size else np.nan
            ),
            "max_displacement": (
                float(np.max(final["displacements"]))
                if final["displacements"].size else np.nan
            ),
            "final_energy": _number(manifest_row.get("final_energy_eV")),
            "perturbed_neighbor_jaccard_distance": immediate["neighbor_jaccard_distance"],
            "perturbed_coordination_change_mean": immediate["coordination_change_mean"],
            "perturbed_coordination_change_max": immediate["coordination_change_max"],
            "perturbed_rdf_l1_distance": immediate["rdf_l1_distance"],
            "neighbor_jaccard_distance": final["neighbor_jaccard_distance"],
            "coordination_change_mean": final["coordination_change_mean"],
            "coordination_change_max": final["coordination_change_max"],
            "rdf_l1_distance": final["rdf_l1_distance"],
            "dft_source_model": parsed["source_model"],
            "dft_source_run_id": parent.get("run_id"),
            "dft_source_match": source_match,
            "dft_structure_name": structure_name,
            "dft_perturbed_structure": str(perturbed_path),
            "dft_relaxed_structure": str(relaxed_path),
            "dft_final_fmax_eV_A": _number(manifest_row.get("final_fmax_eV_A")),
            "dft_outcar": str(outcar) if outcar else None,
            "dft_outcar_force_blocks": block_count,
            "dft_median_delta_force_after_relaxation": (
                dft_median_delta_force_after_relaxation
            ),
            "dft_force_reference": "first OUTCAR TOTAL-FORCE block (perturbed input)",
            "dft_force_target": "final OUTCAR TOTAL-FORCE block",
        })
        records.append(record)

    return records, missing
