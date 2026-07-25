#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-5%5
#SBATCH --output=plot-%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

module load gcc/12.3 python/3.11 arrow

ENVIRONMENT="$HOME/project/.venv-mace"
PYTHON="$ENVIRONMENT/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: missing plotting Python:"
    echo "$PYTHON"
    exit 1
fi

SCRATCH_BASE="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/licohpf_database}"
SUPERCELL_ROOT="${SUPERCELL_ROOT:-${SUPERCELL_OUTPUT_ROOT:-$SCRATCH_BASE/supercell}}"

PROJECT_BASE="${PROJECT_OUTPUT_ROOT:-$REPO_ROOT}"
PROJECT_RESULTS="${PROJECT_RESULTS:-$PROJECT_BASE/licohpf_database_results}"


run_partial_supercell_plots() {
    local supercell_project="$PROJECT_RESULTS/supercell"
    local summary_dir="$supercell_project/summaries"
    local plot_dir="$supercell_project/plots"
    local available_summaries=0
    local model_id
    local source_summary
    local destination_summary

    mkdir -p \
        "$summary_dir" \
        "$plot_dir"

    if [ ! -d "$SUPERCELL_ROOT/array_summaries" ]; then
        echo "WARNING: no supercell array summaries are available:"
        echo "$SUPERCELL_ROOT/array_summaries"
        echo "No supercell plots can be generated yet."
        return 0
    fi

    "$PYTHON" -u pipeline/supercell.py combine \
        --output-root "$SUPERCELL_ROOT"

    for model_id in \
        mace_mh \
        uma \
        mtp \
        chgnet \
        mace_model
    do
        source_summary="$SUPERCELL_ROOT/outputs_float64/$model_id/summary.csv"
        destination_summary="$summary_dir/${model_id}_summary.csv"

        if [ -s "$source_summary" ]; then
            cp \
                "$source_summary" \
                "$destination_summary"
        else
            echo "WARNING: no current combined supercell summary for $model_id"
        fi

        if [ -s "$destination_summary" ]; then
            available_summaries=$((available_summaries + 1))
        fi
    done

    if [ "$available_summaries" -eq 0 ]; then
        echo "WARNING: no usable supercell summaries are available."
        echo "No supercell plots can be generated yet."
        return 0
    fi

    echo "Generating supercell plots from $available_summaries available model summaries."

    "$PYTHON" -u pipeline/runtime.py plot \
        --mace-mh-summary "$summary_dir/mace_mh_summary.csv" \
        --uma-summary "$summary_dir/uma_summary.csv" \
        --mtp-summary "$summary_dir/mtp_summary.csv" \
        --chgnet-summary "$summary_dir/chgnet_summary.csv" \
        --mace-model-summary "$summary_dir/mace_model_summary.csv" \
        --output-dir "$plot_dir" \
        --epsilon 0.01

    echo "Partial-data supercell plots written to:"
    echo "$plot_dir"
}


# A separate afterany Slurm job calls this same plot.sh file with
# PLOT_SUPERCELL_ONLY=1. This prevents the need for another script.
if [ "${PLOT_SUPERCELL_ONLY:-0}" = "1" ]; then
    run_partial_supercell_plots
    exit 0
fi


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
SCRATCH_TRIAL="$SCRATCH_BASE/$TRIAL_NAME"
PROJECT_TRIAL="$PROJECT_RESULTS/$TRIAL_NAME"

mkdir -p "$PROJECT_RESULTS"


# Task 1 submits the random-seed and supercell plotting jobs immediately.
# afterany allows them to run when the plot array finishes even if one or
# more trial plotting tasks fail or contain incomplete data.
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    RANDOM_SEED_JOB=$(
        sbatch \
            --parsable \
            --account=rrg-j3goals \
            --dependency="afterany:${SLURM_ARRAY_JOB_ID}" \
            --time=04:00:00 \
            --mem=24G \
            --cpus-per-task=8 \
            --output=random-seed-%j.out \
            --export=ALL,PROJECT_RESULTS="$PROJECT_RESULTS" \
            --wrap="cd '$REPO_ROOT' && '$PYTHON' -u pipeline/random_seed_comprehensive.py --project-root '$PROJECT_RESULTS' --output-dir '$PROJECT_RESULTS/random_seed'"
    )

    echo "Submitted partial-data-safe random-seed plot job:"
    echo "$RANDOM_SEED_JOB"

    SUPERCELL_PLOT_JOB=$(
        sbatch \
            --parsable \
            --account=rrg-j3goals \
            --dependency="afterany:${SLURM_ARRAY_JOB_ID}" \
            --time=04:00:00 \
            --mem=24G \
            --cpus-per-task=8 \
            --output=supercell-plot-%j.out \
            --export=ALL,PLOT_SUPERCELL_ONLY=1,REPO_ROOT="$REPO_ROOT",SUPERCELL_ROOT="$SUPERCELL_ROOT",PROJECT_RESULTS="$PROJECT_RESULTS" \
            --wrap="cd '$REPO_ROOT' && bash run_licohpf_database/plot.sh"
    )

    echo "Submitted partial-data-safe supercell plot job:"
    echo "$SUPERCELL_PLOT_JOB"
fi


mkdir -p "$PROJECT_TRIAL"

if [ ! -d "$SCRATCH_TRIAL/array_summaries" ]; then
    echo "WARNING: missing main summaries:"
    echo "$SCRATCH_TRIAL/array_summaries"
    exit 0
fi

if [ ! -d "$SCRATCH_TRIAL/contour_array_summaries" ]; then
    echo "WARNING: missing contour summaries:"
    echo "$SCRATCH_TRIAL/contour_array_summaries"
fi

if [ ! -f "$ENVIRONMENT/bin/activate" ]; then
    echo "ERROR: missing plotting environment:"
    echo "$ENVIRONMENT"
    exit 1
fi

source "$ENVIRONMENT/bin/activate"

export SCRATCH_TRIAL
export PROJECT_TRIAL


"$PYTHON" - <<'PY'
from pathlib import Path
import os

import pandas as pd


scratch_trial = Path(
    os.environ["SCRATCH_TRIAL"]
)
project_trial = Path(
    os.environ["PROJECT_TRIAL"]
)

main_summary_dir = (
    scratch_trial / "array_summaries"
)
contour_summary_dir = (
    scratch_trial / "contour_array_summaries"
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
    .astype(str)
    .unique()
)

expected_structures = len(
    configured_materials
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


def read_available_frames(
    paths,
    description,
):
    frames = []

    for path in paths:
        try:
            frame = pd.read_csv(
                path,
                keep_default_na=False,
            )
        except Exception as error:
            print(
                f"WARNING: could not read "
                f"{description} {path}: {error}"
            )
            continue

        if frame.empty:
            print(
                f"WARNING: empty {description}: "
                f"{path}"
            )
            continue

        frames.append(frame)

    if not frames:
        return None

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def report_failed_rows(
    frame,
    description,
):
    if "status" not in frame.columns:
        print(
            f"WARNING: {description} has no "
            "status column"
        )
        return

    successful = (
        frame["status"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "success"
    )

    failed = frame[~successful]

    if not failed.empty:
        print(
            f"WARNING: {len(failed)} failed rows "
            f"in {description}; continuing"
        )
        print(
            failed.to_string(index=False)
        )


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
        model_output = (
            dtype_output / model_id
        )

        model_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        main_paths = sorted(
            main_summary_dir.glob(
                f"{dtype_str}_{model_id}_"
                "licohpf_*_summary.csv"
            )
        )

        if len(main_paths) != expected_structures:
            print(
                f"WARNING: expected "
                f"{expected_structures} "
                f"{dtype_str} {model_id} main "
                f"summaries, found "
                f"{len(main_paths)}"
            )

        main = read_available_frames(
            main_paths,
            (
                f"{dtype_str} {model_id} "
                "main summary"
            ),
        )

        if main is None:
            print(
                f"WARNING: no usable "
                f"{dtype_str} {model_id} "
                "main summaries; skipping model"
            )
            continue

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

        if len(main) != expected_main_rows:
            print(
                f"WARNING: {dtype_str} "
                f"{model_id} main summary has "
                f"{len(main)} rows; expected "
                f"{expected_main_rows}"
            )

        report_failed_rows(
            main,
            (
                f"{dtype_str} {model_id} "
                "main summary"
            ),
        )

        main["calculator"] = model_id
        main["model_id"] = model_id

        if "run_id" in main.columns:
            main = main.drop_duplicates(
                subset=["run_id"],
                keep="last",
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
                f"WARNING: expected "
                f"{expected_structures} "
                f"{dtype_str} {model_id} contour "
                f"summaries, found "
                f"{len(contour_paths)}"
            )

        contour = read_available_frames(
            contour_paths,
            (
                f"{dtype_str} {model_id} "
                "contour summary"
            ),
        )

        if contour is None:
            print(
                f"WARNING: no usable "
                f"{dtype_str} {model_id} contour "
                "summaries; skipping contour"
            )
            continue

        report_failed_rows(
            contour,
            (
                f"{dtype_str} {model_id} "
                "contour summary"
            ),
        )

        contour["calculator"] = model_id
        contour["model_id"] = model_id

        duplicate_columns = [
            column
            for column in [
                "run_id",
                "beta",
            ]
            if column in contour.columns
        ]

        if duplicate_columns:
            contour = contour.drop_duplicates(
                subset=duplicate_columns,
                keep="last",
            )

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
    "Available main and contour summaries "
    "combined into the project directory."
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

    mkdir -p \
        "$mace_mh_dir" \
        "$uma_dir" \
        "$mtp_dir" \
        "$chgnet_dir" \
        "$mace_model_dir"

    "$PYTHON" -u pipeline/run_comprehensive.py \
        --mace-mh-dir "$mace_mh_dir" \
        --uma-dir "$uma_dir" \
        --mtp-dir "$mtp_dir" \
        --chgnet-dir "$chgnet_dir" \
        --mace-model-dir "$mace_model_dir" \
        --output-dir "$dtype_root" \
        --materials generated_licohpf_tests.csv \
        --structures-dir datasets/licohpf_database/structures

    "$PYTHON" -u pipeline/contour_comprehensive.py \
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

PLOT_FAILURE=0

if ! wait "$FLOAT32_PID"; then
    echo "WARNING: float32 plotting was incomplete."
    PLOT_FAILURE=1
fi

if ! wait "$FLOAT64_PID"; then
    echo "WARNING: float64 plotting was incomplete."
    PLOT_FAILURE=1
fi

if ! "$PYTHON" -u pipeline/float_comprehensive.py \
    --float32-dir "$PROJECT_TRIAL/outputs_comprehensive/float32" \
    --float64-dir "$PROJECT_TRIAL/outputs_comprehensive/float64" \
    --output-dir "$PROJECT_TRIAL/outputs_comprehensive/comparison"
then
    echo "WARNING: float32/float64 comparison could not be completed."
    PLOT_FAILURE=1
fi

deactivate

if [ "$PLOT_FAILURE" -ne 0 ]; then
    echo "Plotting completed with partial or missing data."
else
    echo "Plotting completed successfully."
fi

echo "Trial: $TRIAL_NAME"
echo "Project output: $PROJECT_TRIAL"
