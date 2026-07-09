#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=visualize-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load gcc/12.3 python/3.11 arrow

source ~/project/.venv-mace/bin/activate

python -u pipeline/visualize.py \
  --materials datasets/tests_materials.csv \
  --structures-dir mp_structures \
  --output-dir outputs_visuals \
  --dpi 600

deactivate

echo "Visualization complete."