#!/bin/bash

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

CPU_MODELS=(mace_mh uma mtp chgnet)
MATERIALS=20
OUTPUT_ROOT="${CIFS_OUTPUT_ROOT:-$REPO_ROOT}"

if [ "${1:-}" != "worker" ]; then
    for required in \
        20_licohpf.xyz \
        generated_licohpf_cpu_tests.csv \
        generated_licohpf_gpu_tests.csv
    do
        if [ ! -f "$required" ]; then
            echo "ERROR: missing $required" >&2
            echo "Run: bash run_licohpf_database/setup.sh" >&2
            exit 1
        fi
    done

    INITIAL_JOB=$(sbatch --parsable \
        --account=rrg-j3goals \
        --time=00:15:00 \
        --mem=2G \
        --cpus-per-task=1 \
        --output=cifs-initial-%j.out \
        run_licohpf_database/cifs.sh worker initial)

    CPU_JOB=$(sbatch --parsable \
        --account=rrg-j3goals \
        --dependency="afterok:${INITIAL_JOB}" \
        --time=1-00:00:00 \
        --mem=16G \
        --cpus-per-task=8 \
        --array=1-80%40 \
        --output=cifs-cpu-%A_%a.out \
        run_licohpf_database/cifs.sh worker cpu)

    GPU_JOB=$(sbatch --parsable \
        --account=def-j3goals \
        --dependency="afterok:${INITIAL_JOB}" \
        --time=1:00:00 \
        --mem=32G \
        --cpus-per-task=4 \
        --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1 \
        --array=1-20%1 \
        --output=cifs-gpu-%A_%a.out \
        run_licohpf_database/cifs.sh worker gpu)

    echo "CIFS_INITIAL_JOB=$INITIAL_JOB"
    echo "CIFS_CPU_JOB=$CPU_JOB"
    echo "CIFS_GPU_JOB=$GPU_JOB"
    echo "Final output folders:"
    echo "  $OUTPUT_ROOT/structures"
    echo "  $OUTPUT_ROOT/structures_perturbed"
    exit 0
fi

WORKER_KIND="${2:?ERROR: worker type must be initial, cpu, or gpu}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
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

if [ "$WORKER_KIND" = "initial" ]; then
    ENVIRONMENT="$HOME/project/.venv-mace"

    if [ ! -f "$ENVIRONMENT/bin/activate" ]; then
        echo "ERROR: missing environment: $ENVIRONMENT" >&2
        exit 1
    fi

    set +u
    source "$ENVIRONMENT/bin/activate"
    set -u

    mkdir -p "$OUTPUT_ROOT"

    TEMP_STRUCTURES="$OUTPUT_ROOT/.initial_structures_${SLURM_JOB_ID}"
    BACKUP_STRUCTURES="$OUTPUT_ROOT/structures_model_relaxed_${SLURM_JOB_ID}"

    export TEMP_STRUCTURES

    python - <<'PY'
import os
from pathlib import Path

from ase.io import read, write


source = Path("20_licohpf.xyz")
destination = Path(os.environ["TEMP_STRUCTURES"])

frames = read(source, index=":")

if len(frames) != 20:
    raise SystemExit(
        f"ERROR: expected 20 initial structures, found {len(frames)}"
    )

destination.mkdir(parents=True, exist_ok=False)

for index, atoms in enumerate(frames, start=1):
    output_path = destination / f"licohpf_{index:03d}.cif"
    write(output_path, atoms, format="cif")

print(f"Created {len(frames)} initial CIF structures.")
PY

    INITIAL_COUNT=$(
        find "$TEMP_STRUCTURES" \
            -maxdepth 1 \
            -type f \
            -name '*.cif' |
        wc -l
    )

    if [ "$INITIAL_COUNT" -ne 20 ]; then
        echo "ERROR: created $INITIAL_COUNT initial CIFs; expected 20" >&2
        exit 1
    fi

    if [ -d "$OUTPUT_ROOT/structures" ]; then
        mv "$OUTPUT_ROOT/structures" "$BACKUP_STRUCTURES"
        echo "Previous structures backed up to:"
        echo "  $BACKUP_STRUCTURES"
    fi

    mv "$TEMP_STRUCTURES" "$OUTPUT_ROOT/structures"

    echo "Initial CIF generation complete."
    echo "Created:"
    echo "  $OUTPUT_ROOT/structures/licohpf_001.cif"
    echo "  ..."
    echo "  $OUTPUT_ROOT/structures/licohpf_020.cif"
    exit 0
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?ERROR: SLURM_ARRAY_TASK_ID is unset}"

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
    echo "ERROR: worker type must be initial, cpu, or gpu" >&2
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

set +u
source "$ENVIRONMENT/bin/activate"
set -u

if [ "$WORKER_KIND" = "gpu" ]; then
    python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable")

print("CUDA device:", torch.cuda.get_device_name(0))
PY
fi

WORKER_OUTPUT_ROOT="$SLURM_TMPDIR/cifs_${WORKER_KIND}_${TASK_ID}"

echo "Trial: trial1_seed42"
echo "Dtype: float64"
echo "Model: $MODEL_ID"
echo "Structure: $MATERIAL_SLUG"
echo "Temporary worker output: $WORKER_OUTPUT_ROOT"

python -u pipeline/cifs.py \
    --tests "$TESTS_FILE" \
    --model-id "$MODEL_ID" \
    --material-slug "$MATERIAL_SLUG" \
    --output-root "$WORKER_OUTPUT_ROOT"

SOURCE_PERTURBED="$WORKER_OUTPUT_ROOT/structures_perturbed/$MODEL_ID"
FINAL_PERTURBED="$OUTPUT_ROOT/structures_perturbed/$MODEL_ID"

if [ ! -d "$SOURCE_PERTURBED" ]; then
    echo "ERROR: missing generated perturbed structures:" >&2
    echo "$SOURCE_PERTURBED" >&2
    exit 1
fi

mkdir -p "$FINAL_PERTURBED"

cp -a \
    "$SOURCE_PERTURBED/." \
    "$FINAL_PERTURBED/"

PERTURBED_COUNT=$(
    find "$SOURCE_PERTURBED" \
        -maxdepth 1 \
        -type f \
        -name '*.cif' |
    wc -l
)

if [ "$PERTURBED_COUNT" -ne 82 ]; then
    echo "ERROR: generated $PERTURBED_COUNT unique perturbed CIFs; expected 82" >&2
    exit 1
fi

echo "Copied $PERTURBED_COUNT perturbed CIFs to:"
echo "  $FINAL_PERTURBED"
echo "Finished CIF task successfully."
