#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-6
#SBATCH --output=sample-1-contour-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

module load gcc/12.3 python/3.11 arrow

if [ ! -f generated_material_tests.csv ]; then
  echo "ERROR: generated_material_tests.csv missing. Run run_<dataset>/sample_1/setup.sh first."
  exit 1
fi

source ~/project/.venv-mace/bin/activate
mapfile -t CONTOUR_JOBS < <(env -u SLURM_ARRAY_TASK_ID python -u pipeline/contour.py --tests generated_material_tests.csv --config datasets/2d_structures/test_1.json --list-jobs)
deactivate

TRIAL_NAME="trial1_seed42"
MLFF_SEED=42

JOB_COUNT="${#CONTOUR_JOBS[@]}"
TOTAL_COUNT=$((JOB_COUNT * 2))
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

if [ "$JOB_COUNT" -eq 0 ]; then
  echo "ERROR: contour.py --list-jobs returned zero jobs."
  exit 1
fi

if [ "$TASK_INDEX" -lt 0 ] || [ "$TASK_INDEX" -ge "$TOTAL_COUNT" ]; then
  echo "ERROR: no contour job for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  echo "Valid task IDs are 1..$TOTAL_COUNT"
  exit 1
fi

if [ "$TASK_INDEX" -lt "$JOB_COUNT" ]; then
  MLFF_DTYPE="float32"
  JOB_INDEX="$TASK_INDEX"
else
  MLFF_DTYPE="float64"
  JOB_INDEX=$((TASK_INDEX - JOB_COUNT))
fi
export MLFF_DTYPE
export MLFF_SEED
export MLFF_OUTPUT_ROOT="$TRIAL_NAME"

JOB_LINE="${CONTOUR_JOBS[$JOB_INDEX]}"
IFS=',' read -r JOB_NUMBER CALCULATOR MATERIAL_SLUG INPUT_PATH <<< "$JOB_LINE"

echo "Selected trial: $TRIAL_NAME"
echo "Selected dtype: $MLFF_DTYPE"
echo "Selected seed: $MLFF_SEED"
echo "Selected contour material: $MATERIAL_SLUG"
echo "Calculator: $CALCULATOR"

if [ "$CALCULATOR" = "uma" ] && [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  echo "ERROR: UMA requires HF_TOKEN in .env or HUGGINGFACE_HUB_TOKEN in the environment."
  exit 1
fi

if [ "$CALCULATOR" = "mace" ]; then
  source ~/project/.venv-mace/bin/activate
elif [ "$CALCULATOR" = "uma" ]; then
  source ~/project/.venv-uma/bin/activate
elif [ "$CALCULATOR" = "chgnet" ]; then
  source ~/project/.venv-chgnet/bin/activate
else
  echo "ERROR: unknown calculator $CALCULATOR"
  exit 1
fi

python -u pipeline/contour.py \
  --tests generated_material_tests.csv \
  --config datasets/2d_structures/test_1.json \
  --calculator "$CALCULATOR" \
  --material-slug "$MATERIAL_SLUG" \
  --dtype-str "$MLFF_DTYPE" \
  --seed "$MLFF_SEED"

deactivate

echo "Finished $MLFF_DTYPE contour smoke test for $CALCULATOR $MATERIAL_SLUG"