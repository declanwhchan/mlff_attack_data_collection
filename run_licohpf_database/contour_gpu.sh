#!/bin/bash
#SBATCH --account=def-j3goals
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=1-200%40
#SBATCH --output=contour-gpu-%A_%a.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

module load gcc/12.3 python/3.11 arrow

TESTS_FILE="$REPO_ROOT/generated_licohpf_gpu_tests.csv"
CONFIG_FILE="$REPO_ROOT/datasets/licohpf_database/tests_comprehensive.json"

for path in "$TESTS_FILE" "$CONFIG_FILE"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: missing required file: $path"
        exit 1
    fi
done

TASK_INFO="$(
python - <<'PY'
import os

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

trials = [
    ("trial1_seed42", 42),
    ("trial2_seed43", 43),
    ("trial3_seed44", 44),
    ("trial4_seed45", 45),
    ("trial5_seed46", 46),
]

materials = [
    f"licohpf_{index:03d}"
    for index in range(1, 21)
]

dtypes = [
    "float32",
    "float64",
]

tasks_per_trial = len(materials) * len(dtypes)
maximum_task_id = len(trials) * tasks_per_trial

if task_id < 1 or task_id > maximum_task_id:
    raise SystemExit(
        f"ERROR: task ID must be 1..{maximum_task_id}, "
        f"got {task_id}"
    )

index = task_id - 1
trial_index = index // tasks_per_trial
within_trial = index % tasks_per_trial

dtype_index = within_trial // len(materials)
material_index = within_trial % len(materials)

trial_name, seed = trials[trial_index]
dtype_str = dtypes[dtype_index]
material_slug = materials[material_index]

print(
    trial_name,
    seed,
    dtype_str,
    material_slug,
)
PY
)"

read -r TRIAL_NAME MLFF_SEED MLFF_DTYPE MATERIAL_SLUG \
    <<< "$TASK_INFO"

MODEL_ID="mace_model"

export MODEL_ID
export MLFF_SEED
export MLFF_DTYPE
export MATERIAL_SLUG

SCRATCH_BASE="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/licohpf_database}"
TRIAL_DIR="$SCRATCH_BASE/$TRIAL_NAME"

export MLFF_OUTPUT_ROOT="$TRIAL_DIR"

CONTOUR_SUMMARY_DIR="$TRIAL_DIR/contour_array_summaries"
mkdir -p "$CONTOUR_SUMMARY_DIR"

export CONTOUR_SUMMARY_FILE="$CONTOUR_SUMMARY_DIR/${MLFF_DTYPE}_${MODEL_ID}_${MATERIAL_SLUG}_summary.csv"

echo "Trial: $TRIAL_NAME"
echo "Seed: $MLFF_SEED"
echo "Model: $MODEL_ID"
echo "Dtype: $MLFF_DTYPE"
echo "Structure: $MATERIAL_SLUG"
echo "Output root: $MLFF_OUTPUT_ROOT"
echo "Summary: $CONTOUR_SUMMARY_FILE"

ENVIRONMENT="$HOME/project/.venv-mace"

if [ ! -f "$ENVIRONMENT/bin/activate" ]; then
    echo "ERROR: missing environment: $ENVIRONMENT"
    exit 1
fi

source "$ENVIRONMENT/bin/activate"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available")

print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
print("CUDA device:", torch.cuda.get_device_name(0))
PY

python -u pipeline/contour.py \
    --tests generated_licohpf_gpu_tests.csv \
    --config datasets/licohpf_database/tests_comprehensive.json \
    --calculator mace_model \
    --dtype-str "$MLFF_DTYPE" \
    --material-slug "$MATERIAL_SLUG" \
    --seed "$MLFF_SEED"

if [ ! -s "$CONTOUR_SUMMARY_FILE" ]; then
    echo "ERROR: contour summary was not created."
    exit 1
fi

python - <<'PY'
import os
import pandas as pd

path = os.environ["CONTOUR_SUMMARY_FILE"]
expected_dtype = os.environ["MLFF_DTYPE"]
expected_material = os.environ["MATERIAL_SLUG"]

rows = pd.read_csv(path)

if len(rows) != 3:
    raise SystemExit(
        f"ERROR: contour summary has {len(rows)} rows; expected 3"
    )

if set(rows["status"]) != {"success"}:
    failed = rows[rows["status"] != "success"]
    print(failed.to_string(index=False))
    raise SystemExit("ERROR: contour calculations failed")

if set(rows["model_id"]) != {"mace_model"}:
    raise SystemExit("ERROR: incorrect model_id in contour summary")

if set(rows["dtype_str"]) != {expected_dtype}:
    raise SystemExit("ERROR: incorrect dtype in contour summary")

if set(rows["material_slug"]) != {expected_material}:
    raise SystemExit("ERROR: incorrect material in contour summary")

expected_betas = {0.0, 0.05, 0.1}
actual_betas = {
    round(float(value), 8)
    for value in rows["beta"]
}

if actual_betas != expected_betas:
    raise SystemExit(
        f"ERROR: expected betas {expected_betas}; "
        f"got {actual_betas}"
    )

print("All three CUDA contour calculations succeeded")
PY

deactivate

echo "Finished CUDA contour task successfully."