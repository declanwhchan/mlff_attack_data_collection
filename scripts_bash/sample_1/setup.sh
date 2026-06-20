#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=sample-1-setup-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export PYTHONUNBUFFERED=1

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -z "${MP_API_KEY:-}" ]; then
  echo "ERROR: .env is missing MP_API_KEY"
  exit 1
fi

module load gcc/12.3 python/3.11 arrow

source ~/project/.venv-mace/bin/activate

python -u scripts_python/run_material_mpids.py \
  --materials test_1.csv \
  --config test_1.json \
  --tests-out generated_material_tests.csv

deactivate

echo "Setup smoke test complete."