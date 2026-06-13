#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

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

for calculator, output_dir in [
    ("mace", Path("outputs_mace")),
    ("uma", Path("outputs_uma")),
]:
    files = sorted(summary_dir.glob(f"{calculator}_*_summary.csv"))

    if not files:
        raise SystemExit(f"ERROR: no {calculator} summary files found in {summary_dir}")

    frames = [pd.read_csv(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)

    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "summary.csv"
    combined.to_csv(output_path, index=False)

    print(f"Wrote {len(combined)} rows to {output_path}", flush=True)
PY

python -u scripts_python/run_comprehensive.py --output-dir comprehensive_outputs

if [ -f outputs_mace/contour/summary.csv ] || [ -f outputs_uma/contour/summary.csv ]; then
  python -u scripts_python/contour_comprehensive.py \
    --mace-contour-dir outputs_mace/contour \
    --uma-contour-dir outputs_uma/contour \
    --comprehensive-dir comprehensive_outputs \
    --output-dir comprehensive_outputs/contour
else
  echo "No contour summaries found; skipping contour comparison plots."
fi

deactivate

echo "Plotting complete."