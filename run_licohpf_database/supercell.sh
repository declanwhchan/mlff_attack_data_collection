#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=2-00:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-640%150
#SBATCH --output=supercell-cpu-%A_%a.out

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

SUPERCELL_ROOT="${SUPERCELL_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/licohpf_database/supercell}"

if [ ! -f "$SUPERCELL_ROOT/generated_supercell_tests.csv" ]; then
    echo "ERROR: missing $SUPERCELL_ROOT/generated_supercell_tests.csv"
    echo "Generate the supercell database before submitting this job."
    exit 1
fi

CPU_TASK_ID="${SLURM_ARRAY_TASK_ID}"

if [ "$CPU_TASK_ID" -lt 1 ] || [ "$CPU_TASK_ID" -gt 640 ]; then
    echo "ERROR: CPU array task must be 1..640"
    exit 1
fi

TASK_ZERO=$((CPU_TASK_ID - 1))
CELL_INDEX=$((TASK_ZERO / 4))
CPU_MODEL_INDEX=$((TASK_ZERO % 4))

FULL_TASK_ID=$((CELL_INDEX * 5 + CPU_MODEL_INDEX + 1))

TASK_INFO="$(
"$HOME/project/.venv-mace/bin/python" pipeline/supercell.py task-info \
    --output-root "$SUPERCELL_ROOT" \
    --task-id "$FULL_TASK_ID"
)"

echo "$TASK_INFO"

eval "$TASK_INFO"

export MATERIAL
export REPEAT
export MODEL_ID
export CALCULATOR_BACKEND
export DEVICE
export TEST_CSV
export SUMMARY_FILE

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
        echo "ERROR: GPU model appeared in CPU task: $MODEL_ID"
        exit 1
        ;;
esac

if [ "$DEVICE" != "cpu" ]; then
    echo "ERROR: CPU model $MODEL_ID has device=$DEVICE"
    exit 1
fi

if [ ! -f "$ENVIRONMENT/bin/activate" ]; then
    echo "ERROR: missing environment: $ENVIRONMENT"
    exit 1
fi

set +u
source "$ENVIRONMENT/bin/activate"
set -u

if [ "$MODEL_ID" = "mtp" ]; then
    if ! command -v mlp >/dev/null 2>&1; then
        echo "ERROR: mlp is unavailable"
        exit 1
    fi

    mlp list | head -n 3
fi

export MLFF_OUTPUT_ROOT="$SUPERCELL_ROOT"
export MLFF_DTYPE="float64"
export MLFF_SEED="42"

python -u pipeline/runtime.py run \
    --tests "$TEST_CSV" \
    --summary-file "$SUMMARY_FILE"

python - <<'PY'
import os
import pandas as pd

path = os.environ["SUMMARY_FILE"]
model_id = os.environ["MODEL_ID"]

rows = pd.read_csv(path)

if len(rows) != 3:
    raise SystemExit(
        f"ERROR: expected 3 supercell results, got {len(rows)}"
    )

failed = rows[rows["status"] != "success"]

if not failed.empty:
    print(failed.to_string(index=False))
    raise SystemExit(
        f"ERROR: {len(failed)} supercell runs failed"
    )

if set(rows["model_id"]) != {model_id}:
    raise SystemExit(
        "ERROR: incorrect model identity in summary"
    )

print("All three CPU supercell attacks succeeded")
PY

rm -f "$TEST_CSV"

deactivate

echo "Finished CPU supercell task."
