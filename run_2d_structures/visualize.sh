#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=visualize-2d-%j.out

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

MATERIALS_FILE="$REPO_ROOT/datasets/2d_structures/tests_materials.csv"
STRUCTURES_DIR="$REPO_ROOT/mp_structures"
OUTPUT_DIR="$REPO_ROOT/2d_structures_results/visualizations"

for required_path in \
    "$MATERIALS_FILE" \
    "$STRUCTURES_DIR" \
    "$REPO_ROOT/pipeline/visualize.py"; do
    if [ ! -e "$required_path" ]; then
        echo "ERROR: Missing required path:"
        echo "$required_path"
        echo "Run setup.sh before visualize.sh."
        exit 1
    fi
done

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Missing MACE Python:"
    echo "$PYTHON"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Generating 2D structure visualizations"
echo "Materials: $MATERIALS_FILE"
echo "Structures: $STRUCTURES_DIR"
echo "Output: $OUTPUT_DIR"
echo "Python: $PYTHON"

"$PYTHON" -u pipeline/visualize.py \
    --materials datasets/2d_structures/tests_materials.csv \
    --structures-dir mp_structures \
    --output-dir 2d_structures_results/visualizations \
    --dpi 600 \
    --rotation "10x,-20y,0z" \
    --scale 0.85 \
    --radii-scale 0.85 \
    --suptitle "Initial 2D Material Structures"

PNG_COUNT=$(
    find "$OUTPUT_DIR" \
        -type f \
        -name '*.png' \
        | wc -l
)

if [ "$PNG_COUNT" -ne 21 ]; then
    echo "ERROR: Expected 21 PNG files, generated:"
    echo "$PNG_COUNT"
    exit 1
fi

required_outputs=(
    "$OUTPUT_DIR/initial_structures_5x4.png"
    "$OUTPUT_DIR/structure_diagnostics.csv"
    "$OUTPUT_DIR/structure_diagnostics.md"
)

for output_file in "${required_outputs[@]}"; do
    if [ ! -s "$output_file" ]; then
        echo "ERROR: Expected visualization output"
        echo "was not generated:"
        echo "$output_file"
        exit 1
    fi
done

echo
echo "Visualization completed successfully."
echo "PNG files: $PNG_COUNT"
echo "Output directory:"
echo "$OUTPUT_DIR"
