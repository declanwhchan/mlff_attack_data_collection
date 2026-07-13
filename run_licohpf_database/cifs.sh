#!/bin/bash

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

CPU_MODELS=(mace_mh uma mtp chgnet)
MATERIALS=20
OUTPUT_ROOT="${CIFS_OUTPUT_ROOT:-$REPO_ROOT}"

if [ "${1:-}" != "worker" ]; then
    for required in \
        generated_licohpf_cpu_tests.csv \
        generated_licohpf_gpu_tests.csv
    do
        if [ ! -f "$required" ]; then
            echo "ERROR: missing $required" >&2
            echo "Run: bash run_licohpf_database/setup.sh" >&2
            exit 1
        fi
    done

    CPU_JOB=$(sbatch --parsable \
        --account=rrg-j3goals \
        --time=5:00:00 \
        --mem=16G \
        --cpus-per-task=8 \
        --array=1-80%40 \
        --output=cifs-cpu-%A_%a.out \
        run_licohpf_database/cifs.sh worker cpu)

    GPU_JOB=$(sbatch --parsable \
        --account=def-j3goals \
        --time=5:00:00 \
        --mem=32G \
        --cpus-per-task=8 \
        --gpus-per-node=h100:1 \
        --array=1-20%20 \
        --output=cifs-gpu-%A_%a.out \
        run_licohpf_database/cifs.sh worker gpu)

    echo "CIFS_CPU_JOB=$CPU_JOB"
    echo "CIFS_GPU_JOB=$GPU_JOB"
    echo "Only these output folders are used:"
    echo "  $OUTPUT_ROOT/structures"
    echo "  $OUTPUT_ROOT/structures_perturbed"
    exit 0
fi

WORKER_KIND="${2:?ERROR: worker type must be cpu or gpu}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?ERROR: SLURM_ARRAY_TASK_ID is unset}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export TORCH_NUM_THREADS="$OMP_NUM_THREADS"
export MLFF_SEED=42

module load gcc/12.3 python/3.11 arrow

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

if [ "$WORKER_KIND" = "cpu" ]; then
    if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 80 ]; then
        echo "ERROR: CPU task ID must be 1..80" >&2
        exit 1
    fi

    MODEL_INDEX=$(( (TASK_ID - 1) / MATERIALS ))
    MATERIAL_INDEX=$(( (TASK_ID - 1) % MATERIALS + 1 ))
    MODEL_ID="${CPU_MODELS[$MODEL_INDEX]}"
    TESTS_FILE="$REPO_ROOT/generated_licohpf_cpu_tests.csv"

    case "$MODEL_ID" in
        mace_mh) ENVIRONMENT="$HOME/project/.venv-mace" ;;
        uma) ENVIRONMENT="$HOME/project/.venv-uma" ;;
        mtp) ENVIRONMENT="$HOME/project/.venv-mtp" ;;
        chgnet) ENVIRONMENT="$HOME/project/.venv-chgnet" ;;
    esac
elif [ "$WORKER_KIND" = "gpu" ]; then
    if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 20 ]; then
        echo "ERROR: GPU task ID must be 1..20" >&2
        exit 1
    fi

    MODEL_ID="mace_model"
    MATERIAL_INDEX="$TASK_ID"
    TESTS_FILE="$REPO_ROOT/generated_licohpf_gpu_tests.csv"
    ENVIRONMENT="$HOME/project/.venv-mace"
else
    echo "ERROR: worker type must be cpu or gpu" >&2
    exit 1
fi

MATERIAL_SLUG=$(printf 'licohpf_%03d' "$MATERIAL_INDEX")

if [ ! -f "$TESTS_FILE" ]; then
    echo "ERROR: missing setup-generated file: $TESTS_FILE" >&2
    exit 1
fi

if [ ! -f "$ENVIRONMENT/bin/activate" ]; then
    echo "ERROR: missing environment: $ENVIRONMENT" >&2
    exit 1
fi

source "$ENVIRONMENT/bin/activate"

if [ "$WORKER_KIND" = "gpu" ]; then
    python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable")
print("CUDA device:", torch.cuda.get_device_name(0))
PY
fi

echo "Trial: trial1_seed42"
echo "Dtype: float64"
echo "Model: $MODEL_ID"
echo "Structure: $MATERIAL_SLUG"

python -u pipeline/cifs.py \
    --tests "$TESTS_FILE" \
    --model-id "$MODEL_ID" \
    --material-slug "$MATERIAL_SLUG" \
    --output-root "$OUTPUT_ROOT"

deactivate
echo "Finished relax-then-perturb task successfully."
