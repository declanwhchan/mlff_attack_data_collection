#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=2-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-600%15
#SBATCH --output=contour-cpu-%A_%a.out

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
CONFIG_FILE="$REPO_ROOT/datasets/2d_structures/tests_comprehensive.json"
MACE_PYTHON="$HOME/project/.venv-mace/bin/python"

for required_file in \
    "$CPU_TESTS" \
    "$CONFIG_FILE" \
    "$REPO_ROOT/pipeline/contour.py"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Missing required file:"
        echo "$required_file"
        echo "Run setup.sh before submitting contour.sh."
        exit 1
    fi
done

if [ ! -x "$MACE_PYTHON" ]; then
    echo "ERROR: Missing MACE Python:"
    echo "$MACE_PYTHON"
    exit 1
fi

TASK_INFO=$(
    "$MACE_PYTHON" - \
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

present_models = {
    model_id
    for model_id, _, _ in jobs
}

expected_models = {
    "mace_mh",
    "uma",
    "chgnet",
}

if present_models != expected_models:
    raise SystemExit(
        "ERROR: CPU contour database contains the "
        f"wrong models: {sorted(present_models)}"
    )

jobs_per_trial = len(jobs)
total_tasks = len(trials) * jobs_per_trial

if jobs_per_trial != 120:
    raise SystemExit(
        "ERROR: Expected 120 CPU contour jobs per "
        f"trial but found {jobs_per_trial}"
    )

if total_tasks != 600:
    raise SystemExit(
        "ERROR: Expected 600 total CPU contour tasks "
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
    echo "ERROR: Could not parse contour task:"
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
        echo "ERROR: Unsupported CPU model: $MODEL_ID"
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
echo "Summary: $CONTOUR_SUMMARY_FILE"
echo "CPU threads: ${SLURM_CPUS_PER_TASK:-8}"

"$PYTHON" -u pipeline/contour.py \
    --tests generated_material_cpu_tests.csv \
    --config datasets/2d_structures/tests_comprehensive.json \
    --calculator "$MODEL_ID" \
    --dtype-str "$MLFF_DTYPE" \
    --material-slug "$MATERIAL_SLUG" \
    --seed "$MLFF_SEED"

if [ ! -s "$CONTOUR_SUMMARY_FILE" ]; then
    echo "ERROR: Contour summary was not generated:"
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
        "ERROR: Contour summary contains "
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
                "FAILED CONTOUR:",
                row.get("error", ""),
            )

    raise SystemExit(
        "ERROR: One or more contour calculations failed"
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
    f"All {len(rows)} CPU contour calculations "
    "succeeded"
)
PY

echo "Finished CPU contour task successfully"
echo "Model: $MODEL_ID"
echo "Dtype: $MLFF_DTYPE"
echo "Material: $MATERIAL_SLUG"
echo "Trial: $TRIAL_NAME"
