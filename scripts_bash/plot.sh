#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-5%5
#SBATCH --output=plot-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export PYTHONUNBUFFERED=1

module load gcc/12.3 python/3.11 arrow

TRIALS=(
  "trial1_seed42"
  "trial2_seed43"
  "trial3_seed44"
  "trial4_seed45"
  "trial5_seed46"
)

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

if [ "$TASK_INDEX" -lt 0 ] || [ "$TASK_INDEX" -ge "${#TRIALS[@]}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be 1..${#TRIALS[@]}, got $SLURM_ARRAY_TASK_ID"
  exit 1
fi

PROJECT_OUTPUT_ROOT="${PROJECT_OUTPUT_ROOT:-$PWD}"
SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection}"

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  RANDOM_SEED_JOB=$(sbatch \
    --parsable \
    --account=rrg-j3goals \
    --dependency="afterok:${SLURM_ARRAY_JOB_ID}" \
    --time=02:00:00 \
    --mem=8G \
    --cpus-per-task=4 \
    --output=random-seed-%j.out \
    --export=ALL,PROJECT_OUTPUT_ROOT="$PROJECT_OUTPUT_ROOT" \
    --wrap="cd '$PROJECT_OUTPUT_ROOT' && '$HOME/project/.venv-mace/bin/python' -u scripts_python/random_seed_comprehensive.py --project-root '$PROJECT_OUTPUT_ROOT' --output-dir '$PROJECT_OUTPUT_ROOT/random_seed'")

  echo "Submitted random-seed comparison job: $RANDOM_SEED_JOB"
  echo "It will run after plot array ${SLURM_ARRAY_JOB_ID} completes successfully."
fi

TRIAL_NAME="${TRIALS[$TASK_INDEX]}"

SCRATCH_TRIAL_DIR="$SCRATCH_OUTPUT_ROOT/$TRIAL_NAME"
PROJECT_TRIAL_DIR="$PROJECT_OUTPUT_ROOT/$TRIAL_NAME"

mkdir -p "$PROJECT_TRIAL_DIR/outputs_comprehensive"

echo "Plotting $TRIAL_NAME"
echo "Scratch trial dir: $SCRATCH_TRIAL_DIR"
echo "Project trial dir: $PROJECT_TRIAL_DIR"

if [ ! -d "$SCRATCH_TRIAL_DIR" ]; then
  echo "ERROR: missing trial directory: $SCRATCH_TRIAL_DIR"
  exit 1
fi

source ~/project/.venv-mace/bin/activate

# Backfill immediate-attack topology metrics without rerunning calculations.
python -u - <<PY
from pathlib import Path
import os
import sys
import time

import numpy as np
import pandas as pd
from ase.io import read

sys.path.insert(0, str(Path("$PROJECT_OUTPUT_ROOT") / "scripts_python"))

from run_tests import (
    RDF_METHOD,
    coordination_by_atom,
    neighbor_edge_set,
    rdf_l1_distance,
)

summary_dir = Path("$SCRATCH_TRIAL_DIR") / "array_summaries"


def path_exists(path, attempts=3, delay=2.0):
    path = Path(path)

    for attempt in range(1, attempts + 1):
        try:
            return path.exists()
        except OSError as error:
            print(
                f"Filesystem check failed "
                f"({attempt}/{attempts}) for {path}: {error}"
            )

            if attempt < attempts:
                time.sleep(delay)

    return False


def usable_path(value, fallback):
    if value is not None and not pd.isna(value):
        candidate = Path(str(value))

        if path_exists(candidate):
            return candidate

    return Path(fallback)


def topology_values(before_atoms, after_atoms):
    before_edges = neighbor_edge_set(before_atoms)
    after_edges = neighbor_edge_set(after_atoms)

    added_edges = after_edges - before_edges
    removed_edges = before_edges - after_edges
    union_edges = before_edges | after_edges

    if union_edges:
        jaccard_distance = 1.0 - (
            len(before_edges & after_edges) / len(union_edges)
        )
    else:
        jaccard_distance = 0.0

    before_coordination = coordination_by_atom(
        before_edges,
        before_atoms,
    )
    after_coordination = coordination_by_atom(
        after_edges,
        after_atoms,
    )

    atom_keys = sorted(
        set(before_coordination) | set(after_coordination)
    )

    coordination_changes = [
        abs(
            after_coordination.get(atom, 0)
            - before_coordination.get(atom, 0)
        )
        for atom in atom_keys
    ]

    return {
        "neighbor_edges_before": len(before_edges),
        "neighbor_edges_after": len(after_edges),
        "neighbor_edges_added": len(added_edges),
        "neighbor_edges_removed": len(removed_edges),
        "neighbor_edge_change_count": (
            len(added_edges) + len(removed_edges)
        ),
        "neighbor_jaccard_distance": float(jaccard_distance),
        "coordination_change_mean": (
            float(np.mean(coordination_changes))
            if coordination_changes
            else 0.0
        ),
        "coordination_change_max": (
            float(np.max(coordination_changes))
            if coordination_changes
            else 0.0
        ),
    }


for summary_path in sorted(summary_dir.glob("*_summary.csv")):
    try:
        summary = pd.read_csv(summary_path)
    except Exception as error:
        print(f"Skipping unreadable summary {summary_path}: {error}")
        continue

    changed = False

    for index, row in summary.iterrows():
        if str(row.get("status", "")).strip().lower() != "success":
            continue

        run_dir_value = row.get("actual_output_dir")

        if run_dir_value is None or pd.isna(run_dir_value):
            print(f"Missing output directory for {row.get('run_id')}")
            continue

        run_dir = Path(str(run_dir_value))

        before_path = usable_path(
            row.get("before_relax_traj"),
            run_dir / "before_attack_relaxation.traj",
        )
        perturbed_path = usable_path(
            row.get("output_cif"),
            run_dir / "perturbed.cif",
        )
        final_path = usable_path(
            row.get("final_relaxed_cif"),
            run_dir / "final_relaxed.cif",
        )

        try:
            if not path_exists(before_path):
                raise FileNotFoundError(
                    f"Baseline structure is inaccessible: {before_path}"
                )

            if not path_exists(perturbed_path):
                raise FileNotFoundError(
                    f"Perturbed structure is inaccessible: {perturbed_path}"
                )

            before_atoms = read(before_path, index=-1)
            perturbed_atoms = read(perturbed_path)

            immediate = topology_values(
                before_atoms,
                perturbed_atoms,
            )

            for column, value in immediate.items():
                summary.loc[
                    index,
                    f"perturbed_{column}",
                ] = value

            summary.loc[
                index,
                "perturbed_rdf_l1_distance",
            ] = rdf_l1_distance(
                before_atoms,
                perturbed_atoms,
            )
            summary.loc[
                index,
                "perturbed_rdf_method",
            ] = RDF_METHOD

            final_method = str(
                row.get("rdf_method", "")
            ).strip()

            if final_method != RDF_METHOD:
                if not path_exists(final_path):
                    raise FileNotFoundError(
                        f"Final structure is inaccessible: {final_path}"
                    )

                final_atoms = read(final_path)

                summary.loc[
                    index,
                    "rdf_l1_distance",
                ] = rdf_l1_distance(
                    before_atoms,
                    final_atoms,
                )
                summary.loc[index, "rdf_method"] = RDF_METHOD

            changed = True

        except Exception as error:
            print(
                f"Could not backfill topology for "
                f"{row.get('run_id')}: {error}"
            )

    if changed:
        temporary_summary = summary_path.with_suffix(".csv.tmp")
        summary.to_csv(temporary_summary, index=False)
        os.replace(temporary_summary, summary_path)
        print(f"Updated topology metrics: {summary_path}")
PY

python -u - <<PY
from pathlib import Path
import shutil
import pandas as pd

scratch_trial = Path("$SCRATCH_TRIAL_DIR")
project_trial = Path("$PROJECT_TRIAL_DIR")
summary_dir = scratch_trial / "array_summaries"

for dtype_str in ["float32", "float64"]:
    for calculator in ["mace", "uma"]:
        files = sorted(summary_dir.glob(f"{dtype_str}_{calculator}_*_summary.csv"))

        if not files:
            raise SystemExit(f"ERROR: no {dtype_str} {calculator} summary files found in {summary_dir}")

        combined = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

        scratch_output_dir = scratch_trial / f"outputs_{dtype_str}" / calculator
        scratch_output_dir.mkdir(parents=True, exist_ok=True)
        scratch_summary = scratch_output_dir / "summary.csv"
        combined.to_csv(scratch_summary, index=False)
        print(f"Wrote {len(combined)} rows to {scratch_summary}", flush=True)

        project_output_dir = project_trial / "outputs_comprehensive" / dtype_str / calculator
        project_output_dir.mkdir(parents=True, exist_ok=True)
        project_summary = project_output_dir / "summary.csv"
        shutil.copy2(scratch_summary, project_summary)
        print(f"Copied summary to {project_summary}", flush=True)
PY

run_dtype_branch() {
  local scratch_trial_dir="$1"
  local project_trial_dir="$2"
  local dtype_str="$3"
  local threads="$4"

  export OMP_NUM_THREADS="$threads"
  export MKL_NUM_THREADS="$threads"
  export OPENBLAS_NUM_THREADS="$threads"
  export NUMEXPR_NUM_THREADS="$threads"

  python -u scripts_python/run_comprehensive.py \
    --mace-dir "${scratch_trial_dir}/outputs_${dtype_str}/mace" \
    --uma-dir "${scratch_trial_dir}/outputs_${dtype_str}/uma" \
    --output-dir "${project_trial_dir}/outputs_comprehensive/${dtype_str}"

  if [ -f "${scratch_trial_dir}/outputs_${dtype_str}/mace/contour/summary.csv" ] || [ -f "${scratch_trial_dir}/outputs_${dtype_str}/uma/contour/summary.csv" ]; then
    python -u scripts_python/contour_comprehensive.py \
      --mace-contour-dir "${scratch_trial_dir}/outputs_${dtype_str}/mace/contour" \
      --uma-contour-dir "${scratch_trial_dir}/outputs_${dtype_str}/uma/contour" \
      --comprehensive-dir "${project_trial_dir}/outputs_comprehensive/${dtype_str}" \
      --output-dir "${project_trial_dir}/outputs_comprehensive/${dtype_str}/contour"
  else
    echo "No ${dtype_str} contour summaries found for ${scratch_trial_dir}; skipping contour comparison plots."
  fi
}

run_dtype_branch "$SCRATCH_TRIAL_DIR" "$PROJECT_TRIAL_DIR" float32 4 &
pid_float32=$!

run_dtype_branch "$SCRATCH_TRIAL_DIR" "$PROJECT_TRIAL_DIR" float64 4 &
pid_float64=$!

wait "$pid_float32"
wait "$pid_float64"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

python -u scripts_python/float_comprehensive.py \
  --float32-dir "${PROJECT_TRIAL_DIR}/outputs_comprehensive/float32" \
  --float64-dir "${PROJECT_TRIAL_DIR}/outputs_comprehensive/float64" \
  --output-dir "${PROJECT_TRIAL_DIR}/outputs_comprehensive/comparison"

deactivate

echo "Plotting complete for $SCRATCH_TRIAL_DIR"