#!/bin/bash
#SBATCH --account=def-j3goals
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=1-200%40
#SBATCH --output=main-gpu-%A_%a.out

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
PYTHON="$HOME/project/.venv-mace/bin/python"

if [ ! -f "$GPU_TESTS" ]; then
    echo "ERROR: Missing GPU test database:"
    echo "$GPU_TESTS"
    echo "Run this first:"
    echo "bash run_2d_structures/setup.sh"
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: MACE Python executable was not found:"
    echo "$PYTHON"
    exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty."
    echo "This script must be submitted with an H100 GPU."
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
        f"ERROR: No test rows found in {tests_path}"
    )

model_ids = {
    row["model_id"].strip().lower()
    for row in rows
}

if model_ids != {"mace_model"}:
    raise SystemExit(
        "ERROR: GPU database must contain only "
        f"mace_model. Found: {sorted(model_ids)}"
    )

devices = {
    row["device"].strip().lower()
    for row in rows
}

if devices != {"cuda"}:
    raise SystemExit(
        "ERROR: All GPU database rows must use CUDA. "
        f"Found devices: {sorted(devices)}"
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
        "ERROR: Expected 40 GPU jobs per trial "
        f"but found {jobs_per_trial}. "
        "Expected 20 materials with float32 and "
        "float64."
    )

if total_tasks != 200:
    raise SystemExit(
        "ERROR: Expected 200 total GPU array tasks "
        f"but calculated {total_tasks}"
    )

if task_id < 1 or task_id > total_tasks:
    raise SystemExit(
        f"ERROR: SLURM_ARRAY_TASK_ID must be 1.."
        f"{total_tasks}, got {task_id}"
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
    echo "ERROR: Could not parse GPU task information:"
    echo "$TASK_INFO"
    exit 1
fi

MODEL_ID="mace_model"
CALCULATOR_BACKEND="mace"

SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/2d_structures}"
TRIAL_DIR="$SCRATCH_OUTPUT_ROOT/$TRIAL_NAME"
SUMMARY_DIR="$TRIAL_DIR/array_summaries"

mkdir -p "$SUMMARY_DIR"

export MLFF_SEED
export MLFF_DTYPE
export MLFF_OUTPUT_ROOT="$TRIAL_DIR"

TASK_TMP_ROOT="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$TASK_TMP_ROOT"

TASK_TESTS="$TASK_TMP_ROOT/tests.csv"

"$PYTHON" - \
    "$GPU_TESTS" \
    "$TASK_TESTS" \
    "$MLFF_DTYPE" \
    "$MATERIAL_SLUG" <<'PY'
import csv
import sys
from pathlib import Path


source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
dtype_str = sys.argv[3]
material_slug = sys.argv[4]

with source_path.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames or []
    rows = list(reader)

selected = [
    row
    for row in rows
    if row["model_id"].strip().lower()
    == "mace_model"
    and row["dtype_str"].strip().lower()
    == dtype_str
    and row["material_slug"].strip()
    == material_slug
]

if not selected:
    raise SystemExit(
        "ERROR: No rows selected for "
        f"mace_model {dtype_str} {material_slug}"
    )

for row in selected:
    if row["device"].strip().lower() != "cuda":
        raise SystemExit(
            "ERROR: GPU task selected a non-CUDA row"
        )

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with output_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    writer.writerows(selected)

print(
    f"Selected {len(selected):,} test rows",
    flush=True,
)
PY

SUMMARY_FILE="$SUMMARY_DIR/${MLFF_DTYPE}_${MODEL_ID}_${MATERIAL_SLUG}_summary.csv"

echo "Trial: $TRIAL_NAME"
echo "Seed: $MLFF_SEED"
echo "Model: $MODEL_ID"
echo "Calculator backend: $CALCULATOR_BACKEND"
echo "Dtype: $MLFF_DTYPE"
echo "Material: $MATERIAL_SLUG"
echo "Output root: $MLFF_OUTPUT_ROOT"
echo "Python: $PYTHON"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Temporary test file: $TASK_TESTS"
echo "Summary file: $SUMMARY_FILE"
echo "CPU threads: ${SLURM_CPUS_PER_TASK:-8}"

"$PYTHON" - <<'PY'
import os
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: PyTorch reports that CUDA is unavailable"
    )

print("Python executable:", sys.executable)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device:", torch.cuda.get_device_name(0))
print("MLFF_DTYPE:", os.environ["MLFF_DTYPE"])
print("MLFF_SEED:", os.environ["MLFF_SEED"])
print("MLFF_OUTPUT_ROOT:", os.environ["MLFF_OUTPUT_ROOT"])
PY

SUMMARY_FILE="$SUMMARY_FILE" \
    "$PYTHON" -u pipeline/run_tests.py \
    --tests "$TASK_TESTS"

if [ ! -s "$SUMMARY_FILE" ]; then
    echo "ERROR: Expected GPU summary was not generated:"
    echo "$SUMMARY_FILE"
    exit 1
fi

echo "Finished GPU task successfully"
echo "Model: $MODEL_ID"
echo "Dtype: $MLFF_DTYPE"
echo "Material: $MATERIAL_SLUG"
echo "Trial: $TRIAL_NAME"
