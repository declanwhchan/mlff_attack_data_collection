#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=sample-1-plot-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load gcc/12.3 python/3.11 arrow

source ~/project/.venv-mace/bin/activate

python -u - <<'PY'
from pathlib import Path
import pandas as pd

summary_dir = Path("array_summaries")

for dtype_str in ["float32", "float64"]:
    for calculator in ["mace", "uma"]:
        files = sorted(summary_dir.glob(f"{dtype_str}_{calculator}_*_summary.csv"))
        if not files:
            raise SystemExit(f"ERROR: no {dtype_str} {calculator} summary files found in {summary_dir}")

        combined = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

        output_dir = Path(f"outputs_{dtype_str}") / calculator
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "summary.csv"
        combined.to_csv(output_path, index=False)
        print(f"Wrote {len(combined)} rows to {output_path}", flush=True)
PY

for dtype_str in float32 float64; do
  python -u scripts_python/run_comprehensive.py \
    --mace-dir "outputs_${dtype_str}/mace" \
    --uma-dir "outputs_${dtype_str}/uma" \
    --output-dir "outputs_comprehensive/float/${dtype_str}"

  if [ -f "outputs_${dtype_str}/mace/contour/summary.csv" ] || [ -f "outputs_${dtype_str}/uma/contour/summary.csv" ]; then
    python -u scripts_python/contour_comprehensive.py \
      --mace-contour-dir "outputs_${dtype_str}/mace/contour" \
      --uma-contour-dir "outputs_${dtype_str}/uma/contour" \
      --comprehensive-dir "outputs_comprehensive/float/${dtype_str}" \
      --output-dir "outputs_comprehensive/float/${dtype_str}/contour"
  else
    echo "No ${dtype_str} contour summaries found; skipping ${dtype_str} contour plots."
  fi
done

python -u scripts_python/float_comprehensive.py \
  --float32-dir outputs_comprehensive/float/float32 \
  --float64-dir outputs_comprehensive/float/float64 \
  --output-dir outputs_comprehensive/float/comparison

deactivate

echo "Smoke-test plotting complete."