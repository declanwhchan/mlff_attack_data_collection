#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load gcc/12.3 python/3.11 cuda/12.6 arrow

source ~/project/.venv-mace/bin/activate

python - <<'PY'
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

    print(f"Wrote {len(combined)} rows to {output_path}")
PY

python scripts_python/run_comprehensive.py --output-dir comprehensive_outputs

deactivate

echo "Plotting complete."