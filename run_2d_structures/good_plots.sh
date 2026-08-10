#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=good-plots-%A_%a.out

# Rebuild only the presentation summaries.  This script assumes all five
# trial-level comprehensive datasets already exist under 2d_structures_results.
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
PROJECT_BASE="${PROJECT_OUTPUT_ROOT:-$REPO_ROOT}"
PROJECT_RESULTS="$PROJECT_BASE/2d_structures_results"
RANDOM_SEED_DIR="$PROJECT_RESULTS/random_seed"
WHY_PLOTS_DIR="$PROJECT_RESULTS/why_plots"

for required_path in \
    "$PYTHON" \
    "$REPO_ROOT/pipeline/random_seed_comprehensive.py" \
    "$REPO_ROOT/pipeline/why_plots.py"; do
    if [ ! -e "$required_path" ]; then
        echo "ERROR: Missing required path: $required_path"
        exit 1
    fi
done

for trial in trial1_seed42 trial2_seed43 trial3_seed44 trial4_seed45 trial5_seed46; do
    dataset="$PROJECT_RESULTS/$trial/outputs_comprehensive/float64/combined_dataset.csv"
    if [ ! -s "$dataset" ]; then
        echo "ERROR: Missing trial dataset: $dataset"
        exit 1
    fi
done

mkdir -p "$RANDOM_SEED_DIR" "$WHY_PLOTS_DIR"

# The former all-material Jaccard overview was a non-causal spaghetti plot.
# Remove any stale copies so this job cannot leave it in the presentation set.
rm -f "$WHY_PLOTS_DIR/02_all_material_jaccard_overview.png" \
      "$WHY_PLOTS_DIR/02_jaccard_seed_summary.csv"

echo "Rebuilding random-seed figures: $RANDOM_SEED_DIR"
"$PYTHON" -u pipeline/random_seed_comprehensive.py \
    --project-root "$PROJECT_RESULTS" \
    --output-dir "$RANDOM_SEED_DIR" \
    --models mace_mh uma

echo "Generating presentation why-plots: $WHY_PLOTS_DIR"
"$PYTHON" -u pipeline/why_plots.py \
    --project-root "$PROJECT_RESULTS" \
    --output-dir "$WHY_PLOTS_DIR" \
    --with-pes \
    --with-phonons

for required_output in \
    "$RANDOM_SEED_DIR/random_seed_combined.csv" \
    "$WHY_PLOTS_DIR/01_material_attribution.png"; do
    if [ ! -s "$required_output" ]; then
        echo "ERROR: Expected output was not created: $required_output"
        exit 1
    fi
done

echo "Completed random_seed and why_plots only."
