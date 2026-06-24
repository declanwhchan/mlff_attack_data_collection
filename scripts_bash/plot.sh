#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=04:00:00
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