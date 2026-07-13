#!/bin/bash
#SBATCH --account=def-j3goals
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=1-160%40
#SBATCH --output=supercell-gpu-%A_%a.out

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

SUPERCELL_ROOT="${SUPERCELL_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/licohpf_database/supercell}"

if [ ! -f "$SUPERCELL_ROOT/generated_supercell_tests.csv" ]; then
    echo "ERROR: missing $SUPERCELL_ROOT/generated_supercell_tests.csv"
    echo "Generate the supercell database before submitting this job."
    exit 1
fi

GPU_TASK_ID="${SLURM_ARRAY_TASK_ID}"

if [ "$GPU_TASK_ID" -lt 1 ] || [ "$GPU_TASK_ID" -gt 160 ]; then
    echo "ERROR: GPU array task must be 1..160"
    exit 1
fi

FULL_TASK_ID=$((GPU_TASK_ID * 5))

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

if [ "$MODEL_ID" != "mace_model" ]; then
    echo "ERROR: expected mace_model, got $MODEL_ID"
    exit 1
fi

if [[ "$DEVICE" != cuda* ]]; then
    echo "ERROR: mace_model requires CUDA, got device=$DEVICE"
    exit 1
fi

ENVIRONMENT="$HOME/project/.venv-mace"

if [ ! -f "$ENVIRONMENT/bin/activate" ]; then
    echo "ERROR: missing environment: $ENVIRONMENT"
    exit 1
fi

source "$ENVIRONMENT/bin/activate"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable")

print("CUDA device:", torch.cuda.get_device_name(0))
PY

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

if set(rows["model_id"]) != {"mace_model"}:
    raise SystemExit(
        "ERROR: incorrect GPU model identity"
    )

print("All three CUDA supercell attacks succeeded")
PY

rm -f "$TEST_CSV"

deactivate

echo "Finished CUDA supercell task."
