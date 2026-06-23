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

TRIAL_NAME="${TRIALS[$TASK_INDEX]}"

echo "Plotting $TRIAL_NAME"

if [ ! -d "$TRIAL_NAME" ]; then
  echo "ERROR: missing trial directory: $TRIAL_NAME"
  exit 1
fi

source ~/project/.venv-mace/bin/activate

python -u - <<PY
from pathlib import Path
import pandas as pd

trial = Path("$TRIAL_NAME")
summary_dir = trial / "array_summaries"

for dtype_str in ["float32", "float64"]:
    for calculator in ["mace", "uma"]:
        files = sorted(summary_dir.glob(f"{dtype_str}_{calculator}_*_summary.csv"))

        if not files:
            raise SystemExit(f"ERROR: no {dtype_str} {calculator} summary files found in {summary_dir}")

        combined = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

        output_dir = trial / f"outputs_{dtype_str}" / calculator
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "summary.csv"
        combined.to_csv(output_path, index=False)
        print(f"Wrote {len(combined)} rows to {output_path}", flush=True)
PY

run_dtype_branch() {
  local trial_name="$1"
  local dtype_str="$2"
  local threads="$3"

  export OMP_NUM_THREADS="$threads"
  export MKL_NUM_THREADS="$threads"
  export OPENBLAS_NUM_THREADS="$threads"
  export NUMEXPR_NUM_THREADS="$threads"

  python -u scripts_python/run_comprehensive.py \
    --mace-dir "${trial_name}/outputs_${dtype_str}/mace" \
    --uma-dir "${trial_name}/outputs_${dtype_str}/uma" \
    --output-dir "${trial_name}/outputs_comprehensive/${dtype_str}"

  if [ -f "${trial_name}/outputs_${dtype_str}/mace/contour/summary.csv" ] || [ -f "${trial_name}/outputs_${dtype_str}/uma/contour/summary.csv" ]; then
    python -u scripts_python/contour_comprehensive.py \
      --mace-contour-dir "${trial_name}/outputs_${dtype_str}/mace/contour" \
      --uma-contour-dir "${trial_name}/outputs_${dtype_str}/uma/contour" \
      --comprehensive-dir "${trial_name}/outputs_comprehensive/${dtype_str}" \
      --output-dir "${trial_name}/outputs_comprehensive/${dtype_str}/contour"
  else
    echo "No ${dtype_str} contour summaries found for ${trial_name}; skipping contour comparison plots."
  fi
}

run_dtype_branch "$TRIAL_NAME" float32 8 &
pid_float32=$!

run_dtype_branch "$TRIAL_NAME" float64 8 &
pid_float64=$!

wait "$pid_float32"
wait "$pid_float64"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

python -u scripts_python/float_comprehensive.py \
  --float32-dir "${TRIAL_NAME}/outputs_comprehensive/float32" \
  --float64-dir "${TRIAL_NAME}/outputs_comprehensive/float64" \
  --output-dir "${TRIAL_NAME}/outputs_comprehensive/comparison"

deactivate

echo "Plotting complete for $TRIAL_NAME"