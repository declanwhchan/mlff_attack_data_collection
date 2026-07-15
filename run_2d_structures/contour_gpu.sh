#!/bin/bash
#SBATCH --account=def-j3goals
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=1-200%40
#SBATCH --output=contour-gpu-%A_%a.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

module load gcc/12.3 python/3.11 arrow

GPU_TESTS="$REPO_ROOT/generated_material_gpu_tests.csv"
CONFIG_FILE="$REPO_ROOT/datasets/2d_structures/tests_comprehensive.json"
PYTHON="$HOME/project/.venv-mace/bin/python"

for required_file in \
    "$GPU_TESTS" \
    "$CONFIG_FILE" \
    "$REPO_ROOT/pipeline/contour.py" \
    "$REPO_ROOT/MACE_model.model"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Missing required file:"
        echo "$required_file"
        echo "Run setup.sh before contour_gpu.sh."
        exit 1
    fi
done

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Missing MACE Python:"
    echo "$PYTHON"
    exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty."
    echo "Submit this script with an H100 GPU."
    exit 1
fi

TASK_INFO=$(
    "$PYTHON" - \
        "$GPU_TESTS" \
        "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv
import sys


tests_path = sys.argv[1]
task_id = int(sys.argv[2])

trials = [
    ("trial1_seed42", 42),
    ("trial2_seed43", 43),
    ("trial3_seed44", 44),
    ("trial4_seed45", 45),
    ("trial5_seed46", 46),
]

dtype_order = {
    "float32": 0,
    "float64": 1,
}

with open(
    tests_path,
    "r",
    encoding="utf-8-sig",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))

if not rows:
    raise SystemExit(
        f"ERROR: No GPU rows found in {tests_path}"
    )

models = {
    row["model_id"].strip().lower()
    for row in rows
}

if models != {"mace_model"}:
    raise SystemExit(
        "ERROR: GPU test database must contain only "
        f"mace_model. Found: {sorted(models)}"
    )

devices = {
    row["device"].strip().lower()
    for row in rows
}

if devices != {"cuda"}:
    raise SystemExit(
        "ERROR: GPU test rows must all use CUDA. "
        f"Found: {sorted(devices)}"
    )

jobs = sorted(
    {
        (
            row["dtype_str"].strip().lower(),
            row["material_slug"].strip(),
        )
        for row in rows
    },
    key=lambda item: (
        dtype_order[item[0]],
        item[1],
    ),
)

jobs_per_trial = len(jobs)
total_tasks = len(trials) * jobs_per_trial

if jobs_per_trial != 40:
    raise SystemExit(
        "ERROR: Expected 40 GPU contour jobs per "
        f"trial but found {jobs_per_trial}"
    )

if total_tasks != 200:
    raise SystemExit(
        "ERROR: Expected 200 total GPU contour tasks "
        f"but calculated {total_tasks}"
    )

if task_id < 1 or task_id > total_tasks:
    raise SystemExit(
        f"ERROR: SLURM_ARRAY_TASK_ID must be "
        f"1..{total_tasks}, got {task_id}"
    )

zero_based = task_id - 1
trial_index = zero_based // jobs_per_trial
job_index = zero_based % jobs_per_trial

trial_name, seed = trials[trial_index]
dtype_str, material_slug = jobs[job_index]

print(
    "|".join(
        [
            trial_name,
            str(seed),
            dtype_str,
            material_slug,
        ]
    )
)
PY
)

IFS='|' read -r \
    TRIAL_NAME \
    MLFF_SEED \
    MLFF_DTYPE \
    MATERIAL_SLUG \
    <<< "$TASK_INFO"

if [ -z "${TRIAL_NAME:-}" ] || \
   [ -z "${MLFF_SEED:-}" ] || \
   [ -z "${MLFF_DTYPE:-}" ] || \
   [ -z "${MATERIAL_SLUG:-}" ]; then
    echo "ERROR: Could not parse GPU contour task:"
    echo "$TASK_INFO"
    exit 1
fi

MODEL_ID="mace_model"
CALCULATOR_BACKEND="mace"

SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/2d_structures}"
TRIAL_DIR="$SCRATCH_OUTPUT_ROOT/$TRIAL_NAME"
CONTOUR_SUMMARY_DIR="$TRIAL_DIR/contour_array_summaries"

mkdir -p "$CONTOUR_SUMMARY_DIR"

export MODEL_ID
export CALCULATOR_BACKEND
export MATERIAL_SLUG
export MLFF_SEED
export MLFF_DTYPE
export MLFF_OUTPUT_ROOT="$TRIAL_DIR"

export CONTOUR_SUMMARY_FILE="$CONTOUR_SUMMARY_DIR/${MLFF_DTYPE}_${MODEL_ID}_${MATERIAL_SLUG}_summary.csv"

echo "Trial: $TRIAL_NAME"
echo "Seed: $MLFF_SEED"
echo "Model: $MODEL_ID"
echo "Calculator backend: $CALCULATOR_BACKEND"
echo "Dtype: $MLFF_DTYPE"
echo "Material: $MATERIAL_SLUG"
echo "Output root: $MLFF_OUTPUT_ROOT"
echo "Python: $PYTHON"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Summary: $CONTOUR_SUMMARY_FILE"
echo "CPU threads: ${SLURM_CPUS_PER_TASK:-8}"

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: PyTorch reports that CUDA is unavailable"
    )

print("PyTorch version:", torch.__version__)
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device:", torch.cuda.get_device_name(0))
PY

"$PYTHON" -u pipeline/contour.py \
    --tests generated_material_gpu_tests.csv \
    --config datasets/2d_structures/tests_comprehensive.json \
    --calculator "$MODEL_ID" \
    --dtype-str "$MLFF_DTYPE" \
    --material-slug "$MATERIAL_SLUG" \
    --seed "$MLFF_SEED"

if [ ! -s "$CONTOUR_SUMMARY_FILE" ]; then
    echo "ERROR: GPU contour summary was not generated:"
    echo "$CONTOUR_SUMMARY_FILE"
    exit 1
fi

"$PYTHON" - \
    "$CONTOUR_SUMMARY_FILE" \
    "$CONFIG_FILE" \
    "$MODEL_ID" \
    "$MLFF_DTYPE" \
    "$MATERIAL_SLUG" <<'PY'
import csv
import json
import sys
from pathlib import Path


summary_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
expected_model = sys.argv[3]
expected_dtype = sys.argv[4]
expected_material = sys.argv[5]

with config_path.open(
    "r",
    encoding="utf-8",
) as handle:
    config = json.load(handle)

expected_betas = {
    round(float(value), 10)
    for value in config["contour_betas"]
}

with summary_path.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))

if len(rows) != len(expected_betas):
    raise SystemExit(
        "ERROR: GPU contour summary contains "
        f"{len(rows)} rows; expected "
        f"{len(expected_betas)}"
    )

statuses = {
    row["status"].strip().lower()
    for row in rows
}

if statuses != {"success"}:
    for row in rows:
        if row["status"].strip().lower() != "success":
            print(
                "FAILED GPU CONTOUR:",
                row.get("error", ""),
            )

    raise SystemExit(
        "ERROR: One or more GPU contour "
        "calculations failed"
    )

models = {
    row["model_id"].strip().lower()
    for row in rows
}

if models != {expected_model}:
    raise SystemExit(
        f"ERROR: Incorrect model_id values: {models}"
    )

dtypes = {
    row["dtype_str"].strip().lower()
    for row in rows
}

if dtypes != {expected_dtype}:
    raise SystemExit(
        f"ERROR: Incorrect dtype values: {dtypes}"
    )

materials = {
    row["material_slug"].strip()
    for row in rows
}

if materials != {expected_material}:
    raise SystemExit(
        "ERROR: Incorrect material_slug values: "
        f"{materials}"
    )

actual_betas = {
    round(float(row["beta"]), 10)
    for row in rows
}

if actual_betas != expected_betas:
    raise SystemExit(
        "ERROR: Incorrect contour beta values. "
        f"Expected {sorted(expected_betas)}, "
        f"found {sorted(actual_betas)}"
    )

print(
    f"All {len(rows)} GPU contour calculations "
    "succeeded"
)
PY

echo "Finished GPU contour task successfully"
echo "Model: $MODEL_ID"
echo "Dtype: $MLFF_DTYPE"
echo "Material: $MATERIAL_SLUG"
echo "Trial: $TRIAL_NAME"
