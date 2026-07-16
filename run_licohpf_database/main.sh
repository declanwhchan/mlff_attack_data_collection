#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-700%100
#SBATCH --output=main-cpu-%A_%a.out

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

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

TESTS_FILE="$REPO_ROOT/generated_licohpf_cpu_tests.csv"

if [ ! -f "$TESTS_FILE" ]; then
    echo "ERROR: $TESTS_FILE does not exist."
    echo "Run: bash run_licohpf_database/setup.sh"
    exit 1
fi

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

model_dtype_pairs = [
    ("mace_mh", "float32"),
    ("mace_mh", "float64"),
    ("uma", "float32"),
    ("uma", "float64"),
    ("mtp", "float64"),
    ("chgnet", "float32"),
    ("chgnet", "float64"),
]

tasks_per_trial = len(materials) * len(model_dtype_pairs)
maximum_task_id = len(trials) * tasks_per_trial

if task_id < 1 or task_id > maximum_task_id:
    raise SystemExit(
        f"ERROR: task ID must be 1..{maximum_task_id}, "
        f"got {task_id}"
    )

index = task_id - 1
trial_index = index // tasks_per_trial
within_trial = index % tasks_per_trial

pair_index = within_trial // len(materials)
material_index = within_trial % len(materials)

trial_name, seed = trials[trial_index]
model_id, dtype_str = model_dtype_pairs[pair_index]
material_slug = materials[material_index]

print(
    trial_name,
    seed,
    model_id,
    dtype_str,
    material_slug,
)
PY
)"

read -r TRIAL_NAME MLFF_SEED MODEL_ID MLFF_DTYPE MATERIAL_SLUG \
    <<< "$TASK_INFO"

export MLFF_SEED
export MLFF_DTYPE

SCRATCH_BASE="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/licohpf_database}"
TRIAL_DIR="$SCRATCH_BASE/$TRIAL_NAME"

export MLFF_OUTPUT_ROOT="$TRIAL_DIR"

mkdir -p "$TRIAL_DIR/array_summaries"

echo "Trial: $TRIAL_NAME"
echo "Seed: $MLFF_SEED"
echo "Model: $MODEL_ID"
echo "Dtype: $MLFF_DTYPE"
echo "Structure: $MATERIAL_SLUG"
echo "Output root: $MLFF_OUTPUT_ROOT"

case "$MODEL_ID" in
    mace_mh)
        ENVIRONMENT="$HOME/project/.venv-mace"
        ;;

    uma)
        ENVIRONMENT="$HOME/project/.venv-uma"

        if [ -z "${HF_TOKEN:-}" ] \
            && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
            echo "ERROR: UMA requires HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
            exit 1
        fi
        ;;

    mtp)
        ENVIRONMENT="$HOME/project/.venv-mtp"
        ;;

    chgnet)
        ENVIRONMENT="$HOME/project/.venv-chgnet"
        ;;

    *)
        echo "ERROR: unsupported CPU model: $MODEL_ID"
        exit 1
        ;;
esac

if [ ! -f "$ENVIRONMENT/bin/activate" ]; then
    echo "ERROR: missing environment: $ENVIRONMENT"
    exit 1
fi

set +u
source "$ENVIRONMENT/bin/activate"
set -u

echo "Python: $(which python)"

if [ "$MODEL_ID" = "mtp" ]; then
    if ! command -v mlp >/dev/null 2>&1; then
        echo "ERROR: mlp is not available in .venv-mtp"
        exit 1
    fi

    mlp list | head -n 3
fi

TASK_DIRECTORY="${SLURM_TMPDIR:-/tmp/$USER/mlff_attack_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$TASK_DIRECTORY"

TASK_CSV="$TASK_DIRECTORY/tests.csv"

export TESTS_FILE
export TASK_CSV
export MODEL_ID
export MATERIAL_SLUG

python - <<'PY'
import csv
import os
from pathlib import Path

source_path = Path(os.environ["TESTS_FILE"])
output_path = Path(os.environ["TASK_CSV"])
model_id = os.environ["MODEL_ID"]
dtype_str = os.environ["MLFF_DTYPE"]
material_slug = os.environ["MATERIAL_SLUG"]

with source_path.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames
    rows = [
        row
        for row in reader
        if row["model_id"] == model_id
        and row["dtype_str"] == dtype_str
        and row["material_slug"] == material_slug
    ]

if not rows:
    raise SystemExit(
        "ERROR: no rows selected for "
        f"{model_id} {dtype_str} {material_slug}"
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
    writer.writerows(rows)

print(f"Temporary test rows: {len(rows)}")
print(f"Temporary test file: {output_path}")
PY

SUMMARY_FILE="$TRIAL_DIR/array_summaries/${MLFF_DTYPE}_${MODEL_ID}_${MATERIAL_SLUG}_summary.csv"
export SUMMARY_FILE

python -u pipeline/run_tests.py --tests "$TASK_CSV"

if [ ! -s "$SUMMARY_FILE" ]; then
    echo "ERROR: summary was not created: $SUMMARY_FILE"
    exit 1
fi

python - <<'PY'
import os
import pandas as pd

summary_path = os.environ["SUMMARY_FILE"]
rows = pd.read_csv(summary_path)

tests = pd.read_csv(
    os.environ["TASK_CSV"]
)

if len(rows) != len(tests):
    raise SystemExit(
        f"ERROR: summary has {len(rows)} rows, "
        f"but the task CSV has {len(tests)} rows"
    )

failed = rows[rows["status"] != "success"]

if not failed.empty:
    print(
        failed[
            [
                column
                for column in [
                    "run_id",
                    "status",
                    "error",
                    "reason",
                ]
                if column in failed.columns
            ]
        ].to_string(index=False)
    )
    raise SystemExit(
        f"ERROR: {len(failed)} runs did not succeed"
    )

print(f"All {len(rows)} configured CPU runs succeeded")
PY

echo "Finished CPU task successfully."
