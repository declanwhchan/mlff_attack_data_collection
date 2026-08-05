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
        candidates.append(root / str(jobdir) / "OUTCAR")
    candidates.extend([
        relaxed_path.parent / "OUTCAR",
        root / "vasp_relax_fixedcell" / structure_name / "OUTCAR",
    ])
    return next((path for path in candidates if path.is_file()), None)


def load_dft_records(
    dft_root,
    source_records,
    cache_root,
    baseline_path_resolver,
):
    """Convert clean fixed-cell DFT relaxations into normal plot records."""
    dft_root = Path(dft_root)
    manifest_path = dft_root / "manifests" / "preliminary_manifest.csv"
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

        candidates = source[
            (source["calculator"] == parsed["source_model"])
            & (source["attack_label"] == parsed["attack_label"])
        ].copy()
        candidates = candidates[
            candidates["material_slug"].map(_normal_material)
            == _normal_material(parsed["material_slug"])
        ]
        candidate_epsilon = pd.to_numeric(candidates.get("epsilon"), errors="coerce")
        candidates = candidates[
            np.isclose(candidate_epsilon, parsed["epsilon"], rtol=1e-8, atol=1e-12)
        ]
        if parsed["n_steps"] is None:
            candidates = candidates[
                ~candidates["run_id"].astype(str).str.contains("_steps", regex=False)
            ]
        else:
            candidate_steps = pd.to_numeric(candidates.get("n_steps"), errors="coerce")
            candidates = candidates[candidate_steps == parsed["n_steps"]]
        if candidates.empty:
            missing.append(f"No MLFF source record matched: {structure_name}")
            continue

        parent = candidates.iloc[0].to_dict()
        relaxed_value = _clean(manifest_row.get("relaxed_structure_file"))
        if relaxed_value is None:
            missing.append(f"No relaxed DFT structure path: {structure_name}")
            continue
        relaxed_path = dft_root / str(relaxed_value)
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
            "dft_structure_name": structure_name,
            "dft_perturbed_structure": str(perturbed_path),
            "dft_relaxed_structure": str(relaxed_path),
            "dft_final_fmax_eV_A": _number(manifest_row.get("final_fmax_eV_A")),
            "dft_outcar": str(outcar) if outcar else None,
            "dft_outcar_force_blocks": block_count,
            "dft_force_reference": "first OUTCAR TOTAL-FORCE block (perturbed input)",
            "dft_force_target": "final OUTCAR TOTAL-FORCE block",
        })
        records.append(record)

    return records, missing
