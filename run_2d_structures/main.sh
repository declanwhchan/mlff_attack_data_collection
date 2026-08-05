#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-600%15
#SBATCH --output=main-cpu-%A_%a.out

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

if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

module load gcc/12.3 python/3.11 arrow

CPU_TESTS="$REPO_ROOT/generated_material_cpu_tests.csv"

if [ ! -f "$CPU_TESTS" ]; then
    echo "ERROR: Missing CPU test database:"
    echo "$CPU_TESTS"
    echo "Run this first:"
    echo "bash run_2d_structures/setup.sh"
    exit 1
fi

TASK_INFO=$(
    "$HOME/project/.venv-mace/bin/python" - \
        "$CPU_TESTS" \
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

model_order = {
    "mace_mh": 0,
    "uma": 1,
    "chgnet": 2,
}

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

jobs = sorted(
    {
        (
            row["model_id"].strip().lower(),
            row["dtype_str"].strip().lower(),
            row["material_slug"].strip(),
        )
        for row in rows
    },
    key=lambda item: (
        model_order[item[0]],
        dtype_order[item[1]],
        item[2],
    ),
)

expected_models = {
    "mace_mh",
    "uma",
    "chgnet",
}

present_models = {
    model_id
    for model_id, _, _ in jobs
}

if present_models != expected_models:
    raise SystemExit(
        "ERROR: CPU database model set is incorrect. "
        f"Found: {sorted(present_models)}"
    )

jobs_per_trial = len(jobs)
total_tasks = len(trials) * jobs_per_trial

if jobs_per_trial != 120:
    raise SystemExit(
        "ERROR: Expected 120 CPU jobs per trial "
        f"but found {jobs_per_trial}. "
        "Expected 20 materials and six model/dtype "
        "combinations."
    )

if total_tasks != 600:
    raise SystemExit(
        "ERROR: Expected 600 total CPU array tasks "
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
model_id, dtype_str, material_slug = jobs[job_index]

print(
    "|".join(
        [
            trial_name,
            str(seed),
            model_id,
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
    MODEL_ID \
    MLFF_DTYPE \
    MATERIAL_SLUG \
    <<< "$TASK_INFO"

if [ -z "${TRIAL_NAME:-}" ] || \
   [ -z "${MLFF_SEED:-}" ] || \
   [ -z "${MODEL_ID:-}" ] || \
   [ -z "${MLFF_DTYPE:-}" ] || \
   [ -z "${MATERIAL_SLUG:-}" ]; then
    echo "ERROR: Could not parse task information:"
    echo "$TASK_INFO"
    exit 1
fi

case "$MODEL_ID" in
    mace_mh)
        CALCULATOR_BACKEND="mace"
        PYTHON="$HOME/project/.venv-mace/bin/python"
        ;;
    uma)
        CALCULATOR_BACKEND="uma"
        PYTHON="$HOME/project/.venv-uma/bin/python"
        ;;
    chgnet)
        CALCULATOR_BACKEND="chgnet"
        PYTHON="$HOME/project/.venv-chgnet/bin/python"
        ;;
    *)
        echo "ERROR: Unsupported CPU model_id: $MODEL_ID"
        exit 1
        ;;
esac

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python executable was not found:"
    echo "$PYTHON"
    exit 1
fi

if [ "$MODEL_ID" = "uma" ] && \
   [ -z "${HF_TOKEN:-}" ] && \
   [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "ERROR: UMA requires HF_TOKEN in .env or"
    echo "HUGGINGFACE_HUB_TOKEN in the environment."
    exit 1
fi

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
    "$CPU_TESTS" \
    "$TASK_TESTS" \
    "$MODEL_ID" \
    "$MLFF_DTYPE" \
    "$MATERIAL_SLUG" <<'PY'
import csv
import sys
from pathlib import Path


source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
model_id = sys.argv[3]
dtype_str = sys.argv[4]
material_slug = sys.argv[5]

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
    if row["model_id"].strip().lower() == model_id
    and row["dtype_str"].strip().lower() == dtype_str
    and row["material_slug"].strip() == material_slug
]

if not selected:
    raise SystemExit(
        "ERROR: No rows selected for "
        f"{model_id} {dtype_str} {material_slug}"
    )

for row in selected:
    if row["device"].strip().lower() != "cpu":
        raise SystemExit(
            "ERROR: CPU task selected a non-CPU row"
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
echo "Temporary test file: $TASK_TESTS"
echo "Summary file: $SUMMARY_FILE"
echo "CPU threads: ${SLURM_CPUS_PER_TASK:-8}"

"$PYTHON" - <<'PY'
import os
import sys

print("Python executable:", sys.executable)
print("MLFF_DTYPE:", os.environ["MLFF_DTYPE"])
print("MLFF_SEED:", os.environ["MLFF_SEED"])
print("MLFF_OUTPUT_ROOT:", os.environ["MLFF_OUTPUT_ROOT"])
PY

SUMMARY_FILE="$SUMMARY_FILE" \
    "$PYTHON" -u pipeline/run_tests.py \
    --tests "$TASK_TESTS"

if [ ! -s "$SUMMARY_FILE" ]; then
    echo "ERROR: Expected summary was not generated:"
    echo "$SUMMARY_FILE"
    exit 1
fi

echo "Finished CPU task successfully"
echo "Model: $MODEL_ID"
echo "Dtype: $MLFF_DTYPE"
echo "Material: $MATERIAL_SLUG"
echo "Trial: $TRIAL_NAME"
