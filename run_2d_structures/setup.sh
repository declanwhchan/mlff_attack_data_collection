#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=setup-%j.out

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

python -u pipeline/setup_mpids.py \
  --materials datasets/2d_structures/tests_materials.csv \
  --config datasets/2d_structures/tests_comprehensive.json \
  --tests-out generated_material_tests.csv

deactivate

echo "Setup complete."