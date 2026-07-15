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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

module load gcc/12.3 python/3.11 arrow

PYTHON="$HOME/project/.venv-mace/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Missing plotting Python:"
    echo "$PYTHON"
    exit 1
fi

TRIALS=(
    "trial1_seed42"
    "trial2_seed43"
    "trial3_seed44"
    "trial4_seed45"
    "trial5_seed46"
)

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

if [ "$TASK_INDEX" -lt 0 ] || \
   [ "$TASK_INDEX" -ge "${#TRIALS[@]}" ]; then
    echo "ERROR: Plot task must be 1..5"
    exit 1
fi

TRIAL_NAME="${TRIALS[$TASK_INDEX]}"

SCRATCH_BASE="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/2d_structures}"
SCRATCH_TRIAL="$SCRATCH_BASE/$TRIAL_NAME"
SUPERCELL_ROOT="${SUPERCELL_OUTPUT_ROOT:-$SCRATCH_BASE/supercell}"

PROJECT_BASE="${PROJECT_OUTPUT_ROOT:-$REPO_ROOT}"
PROJECT_RESULTS="$PROJECT_BASE/2d_structures_results"
PROJECT_TRIAL="$PROJECT_RESULTS/$TRIAL_NAME"

CONFIG_FILE="$REPO_ROOT/datasets/2d_structures/tests_comprehensive.json"
GENERATED_TESTS="$REPO_ROOT/generated_material_tests.csv"

for required_file in \
    "$CONFIG_FILE" \
    "$GENERATED_TESTS" \
    "$REPO_ROOT/pipeline/run_comprehensive.py" \
    "$REPO_ROOT/pipeline/contour_comprehensive.py" \
    "$REPO_ROOT/pipeline/float_comprehensive.py" \
    "$REPO_ROOT/pipeline/runtime.py" \
    "$REPO_ROOT/pipeline/supercell.py"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Missing required file:"
        echo "$required_file"
        exit 1
    fi
done

MAIN_SUMMARY_DIR="$SCRATCH_TRIAL/array_summaries"
CONTOUR_SUMMARY_DIR="$SCRATCH_TRIAL/contour_array_summaries"

if [ ! -d "$MAIN_SUMMARY_DIR" ]; then
    echo "ERROR: Missing main summaries:"
    echo "$MAIN_SUMMARY_DIR"
    exit 1
fi

if [ ! -d "$CONTOUR_SUMMARY_DIR" ]; then
    echo "ERROR: Missing contour summaries:"
    echo "$CONTOUR_SUMMARY_DIR"
    exit 1
fi

mkdir -p "$PROJECT_TRIAL"

export SCRATCH_TRIAL
export PROJECT_TRIAL
export GENERATED_TESTS
export CONFIG_FILE

"$PYTHON" - <<'PY'
from pathlib import Path
import json
import os

import pandas as pd


scratch_trial = Path(
    os.environ["SCRATCH_TRIAL"]
)
project_trial = Path(
    os.environ["PROJECT_TRIAL"]
)
generated_tests_path = Path(
    os.environ["GENERATED_TESTS"]
)
config_path = Path(
    os.environ["CONFIG_FILE"]
)

main_summary_dir = (
    scratch_trial / "array_summaries"
)
contour_summary_dir = (
    scratch_trial / "contour_array_summaries"
)

configured_tests = pd.read_csv(
    generated_tests_path,
    keep_default_na=False,
)

if configured_tests.empty:
    raise SystemExit(
        "ERROR: generated_material_tests.csv is empty"
    )

with config_path.open(
    "r",
    encoding="utf-8",
) as handle:
    config = json.load(handle)

expected_betas = {
    round(float(value), 12)
    for value in config["contour_betas"]
}

configured_materials = sorted(
    configured_tests["material_slug"]
    .dropna()
    .astype(str)
    .unique()
)

expected_materials = len(configured_materials)

if expected_materials != 20:
    raise SystemExit(
        "ERROR: Expected 20 configured materials, "
        f"found {expected_materials}"
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
                "*_summary.csv"
            )
        )

        if len(main_paths) != expected_materials:
            raise SystemExit(
                f"ERROR: Expected {expected_materials} "
                f"{dtype_str} {model_id} main summaries, "
                f"found {len(main_paths)}"
            )

        main = pd.concat(
            [
                pd.read_csv(
                    path,
                    keep_default_na=False,
                )
                for path in main_paths
            ],
            ignore_index=True,
            sort=False,
        )

        configured_subset = configured_tests[
            (
                configured_tests["dtype_str"]
                .astype(str)
                == dtype_str
            )
            & (
                configured_tests["model_id"]
                .astype(str)
                == model_id
            )
        ]

        expected_main_rows = len(
            configured_subset
        )

        if expected_main_rows == 0:
            raise SystemExit(
                f"ERROR: No configured rows for "
                f"{dtype_str} {model_id}"
            )

        if len(main) != expected_main_rows:
            raise SystemExit(
                f"ERROR: {dtype_str} {model_id} main "
                f"summary contains {len(main)} rows; "
                f"expected {expected_main_rows}"
            )

        statuses = {
            str(value).strip().lower()
            for value in main["status"]
        }

        if statuses != {"success"}:
            failed = main[
                main["status"]
                .astype(str)
                .str.lower()
                != "success"
            ]
            print(failed.to_string(index=False))
            raise SystemExit(
                f"ERROR: Failed main rows for "
                f"{dtype_str} {model_id}"
            )

        main_models = {
            str(value).strip().lower()
            for value in main["model_id"]
        }

        if main_models != {model_id}:
            raise SystemExit(
                f"ERROR: Incorrect model identities "
                f"in {dtype_str} {model_id}: "
                f"{main_models}"
            )

        main["calculator"] = model_id
        main["model_id"] = model_id

        model_output = (
            dtype_output / model_id
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
                "*_summary.csv"
            )
        )

        if len(contour_paths) != expected_materials:
            raise SystemExit(
                f"ERROR: Expected {expected_materials} "
                f"{dtype_str} {model_id} contour "
                f"summaries, found {len(contour_paths)}"
            )

        contour = pd.concat(
            [
                pd.read_csv(
                    path,
                    keep_default_na=False,
                )
                for path in contour_paths
            ],
            ignore_index=True,
            sort=False,
        )

        expected_contour_rows = (
            expected_materials
            * len(expected_betas)
        )

        if len(contour) != expected_contour_rows:
            raise SystemExit(
                f"ERROR: {dtype_str} {model_id} contour "
                f"summary contains {len(contour)} rows; "
                f"expected {expected_contour_rows}"
            )

        actual_betas = {
            round(float(value), 12)
            for value in contour["beta"]
        }

        if actual_betas != expected_betas:
            raise SystemExit(
                f"ERROR: Incorrect contour betas for "
                f"{dtype_str} {model_id}. "
                f"Expected {sorted(expected_betas)}, "
                f"found {sorted(actual_betas)}"
            )

        contour_statuses = {
            str(value).strip().lower()
            for value in contour["status"]
        }

        if contour_statuses != {"success"}:
            failed = contour[
                contour["status"]
                .astype(str)
                .str.lower()
                != "success"
            ]
            print(failed.to_string(index=False))
            raise SystemExit(
                f"ERROR: Failed contour rows for "
                f"{dtype_str} {model_id}"
            )

        contour_models = {
            str(value).strip().lower()
            for value in contour["model_id"]
        }

        if contour_models != {model_id}:
            raise SystemExit(
                f"ERROR: Incorrect contour model IDs "
                f"for {dtype_str} {model_id}: "
                f"{contour_models}"
            )

        contour["calculator"] = model_id
        contour["model_id"] = model_id

        contour_output = (
            model_output / "contour"
        )
        contour_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        contour.to_csv(
            contour_output / "summary.csv",
            index=False,
        )

print(
    "Main and contour summaries combined "
    "into the project directory."
)
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

    "$PYTHON" -u pipeline/run_comprehensive.py \
        --mace-mh-dir "$mace_mh_dir" \
        --uma-dir "$uma_dir" \
        --mtp-dir "$mtp_dir" \
        --chgnet-dir "$chgnet_dir" \
        --mace-model-dir "$mace_model_dir" \
        --output-dir "$dtype_root" \
        --materials generated_material_tests.csv \
        --structures-dir mp_structures

    "$PYTHON" -u pipeline/contour_comprehensive.py \
        --mace-mh-contour-dir "$mace_mh_dir/contour" \
        --uma-contour-dir "$uma_dir/contour" \
        --mtp-contour-dir "$mtp_dir/contour" \
        --chgnet-contour-dir "$chgnet_dir/contour" \
        --mace-model-contour-dir "$mace_model_dir/contour" \
        --comprehensive-dir "$dtype_root" \
        --output-dir "$dtype_root/contour"
}

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

run_dtype_plots float32 &
FLOAT32_PID=$!

run_dtype_plots float64 &
FLOAT64_PID=$!

wait "$FLOAT32_PID"
wait "$FLOAT64_PID"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

"$PYTHON" -u pipeline/float_comprehensive.py \
    --float32-dir "$PROJECT_TRIAL/outputs_comprehensive/float32" \
    --float64-dir "$PROJECT_TRIAL/outputs_comprehensive/float64" \
    --output-dir "$PROJECT_TRIAL/outputs_comprehensive/comparison"

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    SUPERCELL_ARRAY_SUMMARIES="$SUPERCELL_ROOT/array_summaries"

    if [ ! -d "$SUPERCELL_ARRAY_SUMMARIES" ]; then
        echo "ERROR: Missing supercell summaries:"
        echo "$SUPERCELL_ARRAY_SUMMARIES"
        exit 1
    fi

    "$PYTHON" -u pipeline/supercell.py combine \
        --output-root "$SUPERCELL_ROOT"

    SUPERCELL_PROJECT="$PROJECT_RESULTS/supercell"

    mkdir -p \
        "$SUPERCELL_PROJECT/summaries" \
        "$SUPERCELL_PROJECT/plots"

    for model_id in \
        mace_mh \
        uma \
        mtp \
        chgnet \
        mace_model; do
        source_summary="$SUPERCELL_ROOT/outputs_float64/$model_id/summary.csv"
        destination_summary="$SUPERCELL_PROJECT/summaries/${model_id}_summary.csv"

        if [ ! -s "$source_summary" ]; then
            echo "ERROR: Missing supercell summary:"
            echo "$source_summary"
            exit 1
        fi

        cp "$source_summary" "$destination_summary"
    done

    "$PYTHON" -u pipeline/runtime.py plot \
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
        mace_model; do
        rm -f \
            "$SUPERCELL_ROOT/outputs_float64/$model_id/summary.csv"
    done

    RANDOM_SEED_JOB=$(
        sbatch \
            --parsable \
            --account=rrg-j3goals \
            --dependency="afterok:${SLURM_ARRAY_JOB_ID}" \
            --time=04:00:00 \
            --mem=24G \
            --cpus-per-task=8 \
            --output=random-seed-%j.out \
            --export=ALL,PROJECT_RESULTS="$PROJECT_RESULTS" \
            --wrap="cd '$REPO_ROOT' && '$PYTHON' -u pipeline/random_seed_comprehensive.py --project-root '$PROJECT_RESULTS' --output-dir '$PROJECT_RESULTS/random_seed'"
    )

    echo "Submitted random-seed plot job:"
    echo "$RANDOM_SEED_JOB"
fi

rm -f "$MAIN_SUMMARY_DIR/"*.csv
rm -f "$CONTOUR_SUMMARY_DIR/"*.csv

rmdir "$MAIN_SUMMARY_DIR" 2>/dev/null || true
rmdir "$CONTOUR_SUMMARY_DIR" 2>/dev/null || true

echo "Plotting complete"
echo "Trial: $TRIAL_NAME"
echo "Project output: $PROJECT_TRIAL"
