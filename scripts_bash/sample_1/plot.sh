#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=sample-1-plot-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load gcc/12.3 python/3.11 arrow

TRIAL_NAME="trial1_seed42"

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
    for calculator in ["mace", "uma", "chgnet"]:
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

for dtype_str in float32 float64; do
  python -u scripts_python/run_comprehensive.py \
    --mace-dir "${TRIAL_NAME}/outputs_${dtype_str}/mace" \
    --uma-dir "${TRIAL_NAME}/outputs_${dtype_str}/uma" \
    --chgnet-dir "${TRIAL_NAME}/outputs_${dtype_str}/chgnet" \
    --output-dir "${TRIAL_NAME}/outputs_comprehensive/${dtype_str}"

  if [ -f "${TRIAL_NAME}/outputs_${dtype_str}/mace/contour/summary.csv" ] || \
     [ -f "${TRIAL_NAME}/outputs_${dtype_str}/uma/contour/summary.csv" ] || \
     [ -f "${TRIAL_NAME}/outputs_${dtype_str}/chgnet/contour/summary.csv" ]; then
    python -u scripts_python/contour_comprehensive.py \
      --mace-contour-dir "${TRIAL_NAME}/outputs_${dtype_str}/mace/contour" \
      --uma-contour-dir "${TRIAL_NAME}/outputs_${dtype_str}/uma/contour" \
      --chgnet-contour-dir "${TRIAL_NAME}/outputs_${dtype_str}/chgnet/contour" \
      --comprehensive-dir "${TRIAL_NAME}/outputs_comprehensive/${dtype_str}" \
      --output-dir "${TRIAL_NAME}/outputs_comprehensive/${dtype_str}/contour"
  else
    echo "No ${dtype_str} contour summaries found; skipping ${dtype_str} contour plots."
  fi
done

python -u scripts_python/float_comprehensive.py \
  --float32-dir "${TRIAL_NAME}/outputs_comprehensive/float32" \
  --float64-dir "${TRIAL_NAME}/outputs_comprehensive/float64" \
  --output-dir "${TRIAL_NAME}/outputs_comprehensive/comparison"

deactivate

echo "Smoke-test plotting complete."