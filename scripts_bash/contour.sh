#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-600%150
#SBATCH --output=contour-%A_%a.out

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
  echo "ERROR: generated_material_tests.csv missing. Run setup.sh first."
  exit 1
fi

source ~/project/.venv-mace/bin/activate
mapfile -t CONTOUR_JOBS < <(env -u SLURM_ARRAY_TASK_ID python -u scripts_python/contour.py --list-jobs)
deactivate

TRIAL_NAMES=("trial1_seed42" "trial2_seed43" "trial3_seed44" "trial4_seed45" "trial5_seed46")
TRIAL_SEEDS=(42 43 44 45 46)

JOB_COUNT="${#CONTOUR_JOBS[@]}"
TASKS_PER_TRIAL=$((JOB_COUNT * 2))
TOTAL_COUNT=$((${#TRIAL_NAMES[@]} * TASKS_PER_TRIAL))
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

TRIAL_INDEX=$((TASK_INDEX / TASKS_PER_TRIAL))
WITHIN_TRIAL=$((TASK_INDEX % TASKS_PER_TRIAL))

PROJECT_OUTPUT_ROOT="${PROJECT_OUTPUT_ROOT:-$PWD}"
SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection}"

TRIAL_NAME="${TRIAL_NAMES[$TRIAL_INDEX]}"
TRIAL_DIR="$SCRATCH_OUTPUT_ROOT/$TRIAL_NAME"
MLFF_SEED="${TRIAL_SEEDS[$TRIAL_INDEX]}"

if [ "$WITHIN_TRIAL" -lt "$JOB_COUNT" ]; then
  MLFF_DTYPE="float32"
  JOB_INDEX="$WITHIN_TRIAL"
else
  MLFF_DTYPE="float64"
  JOB_INDEX=$((WITHIN_TRIAL - JOB_COUNT))
fi

export MLFF_DTYPE
export MLFF_SEED
export MLFF_OUTPUT_ROOT="$TRIAL_DIR"

mkdir -p "$TRIAL_DIR"

JOB_LINE="${CONTOUR_JOBS[$JOB_INDEX]}"
IFS=',' read -r JOB_NUMBER CALCULATOR MATERIAL_SLUG INPUT_PATH <<< "$JOB_LINE"

if [ -z "${CALCULATOR:-}" ] || [ -z "${MATERIAL_SLUG:-}" ]; then
  echo "ERROR: could not parse contour job line: $JOB_LINE"
  exit 1
fi

echo "Selected trial: $TRIAL_NAME"
echo "Selected dtype: $MLFF_DTYPE"
echo "Selected seed: $MLFF_SEED"
echo "Selected contour material: $MATERIAL_SLUG"
echo "Calculator: $CALCULATOR"
echo "Input path: ${INPUT_PATH:-}"
echo "CPU threads per task: $SLURM_CPUS_PER_TASK"

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

which python

python -u scripts_python/contour.py \
  --calculator "$CALCULATOR" \
  --material-slug "$MATERIAL_SLUG" \
  --dtype-str "$MLFF_DTYPE" \
  --seed "$MLFF_SEED"

deactivate

echo "Finished $MLFF_DTYPE contour exploration for $CALCULATOR $MATERIAL_SLUG"