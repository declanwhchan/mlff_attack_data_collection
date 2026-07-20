#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=2-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-700%150
#SBATCH --output=contour-cpu-%A_%a.out

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

export MODEL_ID
export MATERIAL_SLUG
export MLFF_SEED
export MLFF_DTYPE

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

    mlp list >/dev/null
fi

python -u pipeline/contour.py \
    --tests generated_licohpf_cpu_tests.csv \
    --config datasets/licohpf_database/tests_comprehensive.json \
    --calculator "$MODEL_ID" \
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
expected_model = os.environ["MODEL_ID"]
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

if set(rows["model_id"]) != {expected_model}:
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

print("All three CPU contour calculations succeeded")
PY

echo "Finished CPU contour task successfully."
