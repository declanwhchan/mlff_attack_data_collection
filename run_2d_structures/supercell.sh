#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=2-00:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-480%15
#SBATCH --output=supercell-cpu-%A_%a.out

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

SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/2d_structures}"
SUPERCELL_ROOT="${SUPERCELL_OUTPUT_ROOT:-$SCRATCH_OUTPUT_ROOT/supercell}"

SUPERCELL_TESTS="$SUPERCELL_ROOT/generated_supercell_tests.csv"
SUPERCELL_METADATA="$SUPERCELL_ROOT/supercell_metadata.csv"
MACE_PYTHON="$HOME/project/.venv-mace/bin/python"

for required_file in \
    "$SUPERCELL_TESTS" \
    "$SUPERCELL_METADATA" \
    "$REPO_ROOT/pipeline/supercell.py" \
    "$REPO_ROOT/pipeline/runtime.py"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Missing required file:"
        echo "$required_file"
        echo "Run this first:"
        echo "bash run_2d_structures/setup.sh"
        exit 1
    fi
done

if [ ! -x "$MACE_PYTHON" ]; then
    echo "ERROR: Missing MACE Python:"
    echo "$MACE_PYTHON"
    exit 1
fi

CPU_TASK_ID="$SLURM_ARRAY_TASK_ID"

if [ "$CPU_TASK_ID" -lt 1 ] || \
   [ "$CPU_TASK_ID" -gt 480 ]; then
    echo "ERROR: CPU supercell task must be 1..480"
    exit 1
fi

FULL_TASK_ID="$CPU_TASK_ID"

TASK_INFO=$(
    "$MACE_PYTHON" -u pipeline/supercell.py \
        task-info \
        --output-root "$SUPERCELL_ROOT" \
        --task-id "$FULL_TASK_ID" \
        --models mace_mh uma chgnet
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

case "$MODEL_ID" in
    mace_mh)
        EXPECTED_BACKEND="mace"
        PYTHON="$HOME/project/.venv-mace/bin/python"
        ;;
    uma)
        EXPECTED_BACKEND="uma"
        PYTHON="$HOME/project/.venv-uma/bin/python"
        ;;
    chgnet)
        EXPECTED_BACKEND="chgnet"
        PYTHON="$HOME/project/.venv-chgnet/bin/python"
        ;;
    *)
        echo "ERROR: GPU or unknown model appeared in"
        echo "the CPU supercell workflow: $MODEL_ID"
        exit 1
        ;;
esac

if [ "$CALCULATOR_BACKEND" != "$EXPECTED_BACKEND" ]; then
    echo "ERROR: Incorrect calculator backend."
    echo "Model: $MODEL_ID"
    echo "Expected: $EXPECTED_BACKEND"
    echo "Found: $CALCULATOR_BACKEND"
    exit 1
fi

if [ "$DEVICE" != "cpu" ]; then
    echo "ERROR: CPU model $MODEL_ID has"
    echo "device=$DEVICE"
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Missing Python executable:"
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

export MLFF_OUTPUT_ROOT="$SUPERCELL_ROOT"
export MLFF_DTYPE="float64"
export MLFF_SEED="42"

echo "CPU supercell task"
echo "Array task: $CPU_TASK_ID"
echo "Full task: $FULL_TASK_ID"
echo "Material: $MATERIAL"
echo "Repeat: $REPEAT"
echo "Model: $MODEL_ID"
echo "Calculator backend: $CALCULATOR_BACKEND"
echo "Device: $DEVICE"
echo "Dtype: $MLFF_DTYPE"
echo "Seed: $MLFF_SEED"
echo "Python: $PYTHON"
echo "Test CSV: $TEST_CSV"
echo "Summary: $SUMMARY_FILE"
echo "Output root: $MLFF_OUTPUT_ROOT"

"$PYTHON" - <<'PY'
import os
import sys

print("Python executable:", sys.executable)
print("MLFF_DTYPE:", os.environ["MLFF_DTYPE"])
print("MLFF_SEED:", os.environ["MLFF_SEED"])
print("MLFF_OUTPUT_ROOT:", os.environ["MLFF_OUTPUT_ROOT"])
PY

"$PYTHON" -u pipeline/runtime.py run \
    --tests "$TEST_CSV" \
    --summary-file "$SUMMARY_FILE"

if [ ! -s "$SUMMARY_FILE" ]; then
    echo "ERROR: Supercell summary was not generated:"
    echo "$SUMMARY_FILE"
    exit 1
fi

"$PYTHON" - \
    "$SUMMARY_FILE" \
    "$MODEL_ID" \
    "$MATERIAL" \
    "$REPEAT" <<'PY'
import csv
import sys
from pathlib import Path


summary_path = Path(sys.argv[1])
expected_model = sys.argv[2]
expected_material = sys.argv[3]
expected_repeat = sys.argv[4]

with summary_path.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))

if len(rows) != 3:
    raise SystemExit(
        "ERROR: Expected three CPU supercell "
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
            "FAILED SUPERCELL:",
            row.get("run_id", ""),
            row.get("error", ""),
            row.get("reason", ""),
        )

    raise SystemExit(
        f"ERROR: {len(failed)} CPU supercell "
        "calculations failed"
    )

models = {
    row["model_id"].strip().lower()
    for row in rows
}

if models != {expected_model}:
    raise SystemExit(
        f"ERROR: Incorrect model identities: {models}"
    )

dtypes = {
    row["dtype_str"].strip().lower()
    for row in rows
}

if dtypes != {"float64"}:
    raise SystemExit(
        f"ERROR: Incorrect supercell dtypes: {dtypes}"
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
    "All three CPU supercell attacks succeeded"
)
PY

echo "Finished CPU supercell task successfully"
echo "Model: $MODEL_ID"
echo "Material: $MATERIAL"
echo "Repeat: $REPEAT"
