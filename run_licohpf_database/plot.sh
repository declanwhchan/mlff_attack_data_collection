#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-5%5
#SBATCH --output=plot-%A_%a.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

module load gcc/12.3 python/3.11 arrow

TRIALS=(
    "trial1_seed42"
    "trial2_seed43"
    "trial3_seed44"
    "trial4_seed45"
    "trial5_seed46"
)

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

if [ "$TASK_INDEX" -lt 0 ] \
    || [ "$TASK_INDEX" -ge "${#TRIALS[@]}" ]; then
    echo "ERROR: plot task must be 1..${#TRIALS[@]}"
    exit 1
fi

TRIAL_NAME="${TRIALS[$TASK_INDEX]}"

SCRATCH_BASE="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/licohpf_database}"
SCRATCH_TRIAL="$SCRATCH_BASE/$TRIAL_NAME"
SUPERCELL_ROOT="${SUPERCELL_OUTPUT_ROOT:-$SCRATCH_BASE/supercell}"

PROJECT_BASE="${PROJECT_OUTPUT_ROOT:-$REPO_ROOT}"
PROJECT_RESULTS="$PROJECT_BASE/licohpf_database_results"
PROJECT_TRIAL="$PROJECT_RESULTS/$TRIAL_NAME"

mkdir -p "$PROJECT_TRIAL"

if [ ! -d "$SCRATCH_TRIAL/array_summaries" ]; then
    echo "ERROR: missing main summaries:"
    echo "$SCRATCH_TRIAL/array_summaries"
    exit 1
fi

if [ ! -d "$SCRATCH_TRIAL/contour_array_summaries" ]; then
    echo "ERROR: missing contour summaries:"
    echo "$SCRATCH_TRIAL/contour_array_summaries"
    exit 1
fi

ENVIRONMENT="$HOME/project/.venv-mace"

if [ ! -f "$ENVIRONMENT/bin/activate" ]; then
    echo "ERROR: missing plotting environment: $ENVIRONMENT"
    exit 1
fi

source "$ENVIRONMENT/bin/activate"

export SCRATCH_TRIAL
export PROJECT_TRIAL

python - <<'PY'
from pathlib import Path
import os

import pandas as pd


scratch_trial = Path(os.environ["SCRATCH_TRIAL"])
project_trial = Path(os.environ["PROJECT_TRIAL"])

main_summary_dir = (
    scratch_trial
    / "array_summaries"
)

contour_summary_dir = (
    scratch_trial
    / "contour_array_summaries"
)

configured_tests = pd.read_csv(
    "generated_licohpf_tests.csv"
)

if configured_tests.empty:
    raise SystemExit(
        "ERROR: generated_licohpf_tests.csv is empty"
    )

configured_materials = sorted(
    configured_tests["material_slug"]
    .dropna()
    .unique()
)

expected_structures = len(configured_materials)

if expected_structures == 0:
    raise SystemExit(
        "ERROR: no configured structures were found"
    )

models_by_dtype = {
    "float32": [
        "mace_mh",
        "uma",
        "chgnet",
        "mace_model",
    ],
    "float64": [
        "mace_mh",
        "uma",
        "mtp",
        "chgnet",
        "mace_model",
    ],
}

for dtype_str, models in models_by_dtype.items():
    dtype_output = (
        project_trial
        / "outputs_comprehensive"
        / dtype_str
    )
    dtype_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_id in models:
        main_paths = sorted(
            main_summary_dir.glob(
                f"{dtype_str}_{model_id}_"
                "licohpf_*_summary.csv"
            )
        )

        if len(main_paths) != expected_structures:
            print(
                f"WARNING: expected {expected_structures} "
                f"{dtype_str} {model_id} main summaries, "
                f"found {len(main_paths)}"
            )

        main_frames = [
            pd.read_csv(path)
            for path in main_paths
        ]
        main = pd.concat(
            main_frames,
            ignore_index=True,
            sort=False,
        )

        configured_subset = configured_tests[
            (
                configured_tests["dtype_str"]
                == dtype_str
            )
            & (
                configured_tests["model_id"]
                == model_id
            )
        ]

        expected_main_rows = len(
            configured_subset
        )

        if expected_main_rows == 0:
            raise SystemExit(
                f"ERROR: no configured rows for "
                f"{dtype_str} {model_id}"
            )

        if len(main) != expected_main_rows:
            print(
                f"WARNING: {dtype_str} {model_id} main "
                f"summary has {len(main)} rows; "
                f"expected {expected_main_rows}"
            )

        if set(main["status"]) != {"success"}:
            failed = main[
                main["status"] != "success"
            ]
            print(failed.to_string(index=False))
            print(
                f"WARNING: failed main rows for "
                f"{dtype_str} {model_id}; continuing"
            )

        main["calculator"] = model_id
        main["model_id"] = model_id

        model_output = (
            dtype_output
            / model_id
        )
        model_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        main.to_csv(
            model_output / "summary.csv",
            index=False,
        )

        contour_paths = sorted(
            contour_summary_dir.glob(
                f"{dtype_str}_{model_id}_"
                "licohpf_*_summary.csv"
            )
        )

        if len(contour_paths) != expected_structures:
            print(
                f"WARNING: expected {expected_structures} "
                f"{dtype_str} {model_id} contour summaries, "
                f"found {len(contour_paths)}"
            )

        contour_frames = [
            pd.read_csv(path)
            for path in contour_paths
        ]
        contour = pd.concat(
            contour_frames,
            ignore_index=True,
            sort=False,
        )

        contour_betas = {
            round(float(value), 12)
            for value in contour["beta"]
        }

        expected_contour_rows = (
            len(contour_betas)
            * expected_structures
        )

        if len(contour) != expected_contour_rows:
            print(
                f"WARNING: {dtype_str} {model_id} contour "
                f"summary has {len(contour)} rows; "
                f"expected {expected_contour_rows}"
            )

        if set(contour["status"]) != {"success"}:
            failed = contour[
                contour["status"] != "success"
            ]
            print(failed.to_string(index=False))
            print(
                f"WARNING: failed contour rows for "
                f"{dtype_str} {model_id}; continuing"
            )

        contour["calculator"] = model_id
        contour["model_id"] = model_id

        contour_output = (
            model_output
            / "contour"
        )
        contour_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        contour.to_csv(
            contour_output / "summary.csv",
            index=False,
        )

print("Main and contour summaries combined directly into project.")
PY

run_dtype_plots() {
    local dtype_str="$1"
    local dtype_root="$PROJECT_TRIAL/outputs_comprehensive/$dtype_str"

    local mace_mh_dir="$dtype_root/mace_mh"
    local uma_dir="$dtype_root/uma"
    local mtp_dir="$dtype_root/mtp"
    local chgnet_dir="$dtype_root/chgnet"
    local mace_model_dir="$dtype_root/mace_model"

    if [ "$dtype_str" = "float32" ]; then
        mkdir -p "$mtp_dir"
    fi

    python -u pipeline/run_comprehensive.py \
        --mace-mh-dir "$mace_mh_dir" \
        --uma-dir "$uma_dir" \
        --mtp-dir "$mtp_dir" \
        --chgnet-dir "$chgnet_dir" \
        --mace-model-dir "$mace_model_dir" \
        --output-dir "$dtype_root" \
        --materials generated_licohpf_tests.csv \
        --structures-dir datasets/licohpf_database/structures

    python -u pipeline/contour_comprehensive.py \
        --mace-mh-contour-dir "$mace_mh_dir/contour" \
        --uma-contour-dir "$uma_dir/contour" \
        --mtp-contour-dir "$mtp_dir/contour" \
        --chgnet-contour-dir "$chgnet_dir/contour" \
        --mace-model-contour-dir "$mace_model_dir/contour" \
        --comprehensive-dir "$dtype_root" \
        --output-dir "$dtype_root/contour"
}

run_dtype_plots float32 &
FLOAT32_PID=$!

run_dtype_plots float64 &
FLOAT64_PID=$!

wait "$FLOAT32_PID"
wait "$FLOAT64_PID"

python -u pipeline/float_comprehensive.py \
    --float32-dir "$PROJECT_TRIAL/outputs_comprehensive/float32" \
    --float64-dir "$PROJECT_TRIAL/outputs_comprehensive/float64" \
    --output-dir "$PROJECT_TRIAL/outputs_comprehensive/comparison"

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    if [ ! -d "$SUPERCELL_ROOT/array_summaries" ]; then
        echo "ERROR: missing supercell array summaries:"
        echo "$SUPERCELL_ROOT/array_summaries"
        exit 1
    fi

    python pipeline/supercell.py combine \
        --output-root "$SUPERCELL_ROOT"

    SUPERCELL_PROJECT="$PROJECT_RESULTS/supercell"
    mkdir -p "$SUPERCELL_PROJECT/summaries"

    for model_id in \
        mace_mh \
        uma \
        mtp \
        chgnet \
        mace_model
    do
        source_summary="$SUPERCELL_ROOT/outputs_float64/$model_id/summary.csv"
        destination_summary="$SUPERCELL_PROJECT/summaries/${model_id}_summary.csv"

        if [ ! -s "$source_summary" ]; then
            echo "ERROR: missing supercell summary: $source_summary"
            exit 1
        fi

        cp "$source_summary" "$destination_summary"
    done

    python -u pipeline/runtime.py plot \
        --mace-mh-summary "$SUPERCELL_PROJECT/summaries/mace_mh_summary.csv" \
        --uma-summary "$SUPERCELL_PROJECT/summaries/uma_summary.csv" \
        --mtp-summary "$SUPERCELL_PROJECT/summaries/mtp_summary.csv" \
        --chgnet-summary "$SUPERCELL_PROJECT/summaries/chgnet_summary.csv" \
        --mace-model-summary "$SUPERCELL_PROJECT/summaries/mace_model_summary.csv" \
        --output-dir "$SUPERCELL_PROJECT/plots" \
        --epsilon 0.01

    for model_id in \
        mace_mh \
        uma \
        mtp \
        chgnet \
        mace_model
    do
        rm -f "$SUPERCELL_ROOT/outputs_float64/$model_id/summary.csv"
    done

    RANDOM_SEED_JOB="$(
        sbatch \
            --parsable \
            --account=rrg-j3goals \
            --dependency="afterok:${SLURM_ARRAY_JOB_ID}" \
            --time=04:00:00 \
            --mem=24G \
            --cpus-per-task=8 \
            --output=random-seed-%j.out \
            --export=ALL,PROJECT_RESULTS="$PROJECT_RESULTS" \
            --wrap="cd '$REPO_ROOT' && '$HOME/project/.venv-mace/bin/python' -u pipeline/random_seed_comprehensive.py --project-root '$PROJECT_RESULTS' --output-dir '$PROJECT_RESULTS/random_seed'"
    )"

    echo "Submitted random-seed plot job: $RANDOM_SEED_JOB"
fi

rm -f "$SCRATCH_TRIAL/array_summaries/"*.csv
rm -f "$SCRATCH_TRIAL/contour_array_summaries/"*.csv

rmdir "$SCRATCH_TRIAL/array_summaries" 2>/dev/null || true
rmdir "$SCRATCH_TRIAL/contour_array_summaries" 2>/dev/null || true

deactivate

echo "Plotting complete for $TRIAL_NAME"
echo "Project output: $PROJECT_TRIAL"
