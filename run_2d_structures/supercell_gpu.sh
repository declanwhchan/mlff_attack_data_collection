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

SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/2d_structures}"
SUPERCELL_ROOT="${SUPERCELL_OUTPUT_ROOT:-$SCRATCH_OUTPUT_ROOT/supercell}"

SUPERCELL_TESTS="$SUPERCELL_ROOT/generated_supercell_tests.csv"
SUPERCELL_METADATA="$SUPERCELL_ROOT/supercell_metadata.csv"
PYTHON="$HOME/project/.venv-mace/bin/python"

for required_file in \
    "$SUPERCELL_TESTS" \
    "$SUPERCELL_METADATA" \
    "$REPO_ROOT/pipeline/supercell.py" \
    "$REPO_ROOT/pipeline/runtime.py" \
    "$REPO_ROOT/MACE_model.model"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Missing required file:"
        echo "$required_file"
        echo "Run this first:"
        echo "bash run_2d_structures/setup.sh"
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

GPU_TASK_ID="$SLURM_ARRAY_TASK_ID"

if [ "$GPU_TASK_ID" -lt 1 ] || \
   [ "$GPU_TASK_ID" -gt 160 ]; then
    echo "ERROR: GPU supercell task must be 1..160"
    exit 1
fi

FULL_TASK_ID=$((GPU_TASK_ID * 5))

TASK_INFO=$(
    "$PYTHON" -u pipeline/supercell.py \
        task-info \
        --output-root "$SUPERCELL_ROOT" \
        --task-id "$FULL_TASK_ID"
)

echo "$TASK_INFO"

eval "$TASK_INFO"

export MATERIAL
export REPEAT
export MODEL_ID
export CALCULATOR_BACKEND
export DEVICE
export TEST_CSV
export SUMMARY_FILE

cleanup_task_csv() {
    if [ -n "${TEST_CSV:-}" ] && \
       [ -f "$TEST_CSV" ]; then
        rm -f -- "$TEST_CSV"
    fi
}

trap cleanup_task_csv EXIT

if [ "$MODEL_ID" != "mace_model" ]; then
    echo "ERROR: Expected mace_model, found:"
    echo "$MODEL_ID"
    exit 1
fi

if [ "$CALCULATOR_BACKEND" != "mace" ]; then
    echo "ERROR: mace_model requires backend=mace"
    echo "Found: $CALCULATOR_BACKEND"
    exit 1
fi

if [[ "$DEVICE" != cuda* ]]; then
    echo "ERROR: mace_model requires CUDA."
    echo "Found device=$DEVICE"
    exit 1
fi

export MLFF_OUTPUT_ROOT="$SUPERCELL_ROOT"
export MLFF_DTYPE="float64"
export MLFF_SEED="42"

echo "GPU supercell task"
echo "Array task: $GPU_TASK_ID"
echo "Full task: $FULL_TASK_ID"
echo "Material: $MATERIAL"
echo "Repeat: $REPEAT"
echo "Model: $MODEL_ID"
echo "Calculator backend: $CALCULATOR_BACKEND"
echo "Device: $DEVICE"
echo "Dtype: $MLFF_DTYPE"
echo "Seed: $MLFF_SEED"
echo "Python: $PYTHON"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Test CSV: $TEST_CSV"
echo "Summary: $SUMMARY_FILE"
echo "Output root: $MLFF_OUTPUT_ROOT"

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

"$PYTHON" -u pipeline/runtime.py run \
    --tests "$TEST_CSV" \
    --summary-file "$SUMMARY_FILE"

if [ ! -s "$SUMMARY_FILE" ]; then
    echo "ERROR: GPU supercell summary was not generated:"
    echo "$SUMMARY_FILE"
    exit 1
fi

"$PYTHON" - \
    "$SUMMARY_FILE" \
    "$MATERIAL" \
    "$REPEAT" <<'PY'
import csv
import sys
from pathlib import Path


summary_path = Path(sys.argv[1])
expected_material = sys.argv[2]
expected_repeat = sys.argv[3]

with summary_path.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))

if len(rows) != 3:
    raise SystemExit(
        "ERROR: Expected three GPU supercell "
        f"results, found {len(rows)}"
    )

failed = [
    row
    for row in rows
    if row["status"].strip().lower()
    != "success"
]

if failed:
    for row in failed:
        print(
            "FAILED GPU SUPERCELL:",
            row.get("run_id", ""),
            row.get("error", ""),
            row.get("reason", ""),
        )

    raise SystemExit(
        f"ERROR: {len(failed)} GPU supercell "
        "calculations failed"
    )

models = {
    row["model_id"].strip().lower()
    for row in rows
}

if models != {"mace_model"}:
    raise SystemExit(
        f"ERROR: Incorrect GPU model identities: "
        f"{models}"
    )

backends = {
    row["calculator_backend"].strip().lower()
    for row in rows
}

if backends != {"mace"}:
    raise SystemExit(
        f"ERROR: Incorrect GPU backends: {backends}"
    )

devices = {
    row["device"].strip().lower()
    for row in rows
}

if not all(
    device.startswith("cuda")
    for device in devices
):
    raise SystemExit(
        f"ERROR: Incorrect GPU devices: {devices}"
    )

dtypes = {
    row["dtype_str"].strip().lower()
    for row in rows
}

if dtypes != {"float64"}:
    raise SystemExit(
        f"ERROR: Incorrect GPU supercell dtypes: "
        f"{dtypes}"
    )

base_materials = {
    row["base_material_slug"].strip()
    for row in rows
}

if base_materials != {expected_material}:
    raise SystemExit(
        "ERROR: Incorrect base material values: "
        f"{base_materials}"
    )

repeat_values = {
    row["supercell_repeat_tuple"].strip()
    for row in rows
}

if repeat_values != {expected_repeat}:
    raise SystemExit(
        f"ERROR: Incorrect repeat values: "
        f"{repeat_values}"
    )

print(
    "All three CUDA supercell attacks succeeded"
)
PY

echo "Finished GPU supercell task successfully"
echo "Model: mace_model"
echo "Material: $MATERIAL"
echo "Repeat: $REPEAT"
