#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=supercell-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export PYTHONUNBUFFERED=1

MODE="${SUPERCELL_MODE:-controller}"

PROJECT_OUTPUT_ROOT="${PROJECT_OUTPUT_ROOT:-$PWD}"
SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection}"

# Raw/intermediate data goes here, like main.sh scratch trial outputs.
SUPER_ROOT="$SCRATCH_OUTPUT_ROOT/outputs_supercell"

# Final publication/comprehensive output goes here, like plot.sh project outputs.
SUPER_COMPREHENSIVE_DIR="$PROJECT_OUTPUT_ROOT/supercell"

load_env() {
  if [ -f .env ]; then
    set -a
    source .env
    set +a
  fi

  if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  fi

  module load gcc/12.3 python/3.11 arrow
}

controller_mode() {
  load_env

  mkdir -p "$SUPER_ROOT"
  mkdir -p "$SUPER_COMPREHENSIVE_DIR"

  echo "Controller job"
  echo "Scratch supercell root: $SUPER_ROOT"
  echo "Project comprehensive output: $SUPER_COMPREHENSIVE_DIR"

  source ~/project/.venv-mace/bin/activate
  python -u scripts_python/supercell.py generate --output-root "$SUPER_ROOT"
  deactivate

  ARRAY_JOB_ID=$(sbatch --parsable \
    --array=1-216%40 \
    --time=7-00:00:00 \
    --mem=16G \
    --cpus-per-task=8 \
    --output=supercell-%A_%a.out \
    --export=ALL,SUPERCELL_MODE=run,PROJECT_OUTPUT_ROOT="$PROJECT_OUTPUT_ROOT",SCRATCH_OUTPUT_ROOT="$SCRATCH_OUTPUT_ROOT" \
    scripts_bash/supercell.sh)

  PLOT_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:$ARRAY_JOB_ID \
    --time=1-00:00:00 \
    --mem=16G \
    --cpus-per-task=8 \
    --output=supercell-plot-%j.out \
    --export=ALL,SUPERCELL_MODE=plot,PROJECT_OUTPUT_ROOT="$PROJECT_OUTPUT_ROOT",SCRATCH_OUTPUT_ROOT="$SCRATCH_OUTPUT_ROOT" \
    scripts_bash/supercell.sh)

  echo "Submitted supercell run array: $ARRAY_JOB_ID"
  echo "Submitted dependent supercell plot job: $PLOT_JOB_ID"
}

run_mode() {
  load_env

  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
  export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
  export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
  export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
  export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

  source ~/project/.venv-mace/bin/activate

  TASK_ENV=$(python -u scripts_python/supercell.py task-info \
    --output-root "$SUPER_ROOT" \
    --task-id "$SLURM_ARRAY_TASK_ID")

  eval "$TASK_ENV"

  deactivate

  export MLFF_DTYPE="float64"
  export MLFF_SEED="42"

  # This is the important part: run_tests.py writes outputs_float64/mace and outputs_float64/uma under scratch.
  export MLFF_OUTPUT_ROOT="$SUPER_ROOT"

  echo "Supercell run task"
  echo "Scratch output root: $MLFF_OUTPUT_ROOT"
  echo "Material: $MATERIAL"
  echo "Repeat: $REPEAT"
  echo "Calculator: $CALCULATOR"
  echo "Test CSV: $TEST_CSV"
  echo "Summary file: $SUMMARY_FILE"

  if [ "$CALCULATOR" = "uma" ] && [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "ERROR: UMA requires HF_TOKEN in .env or HUGGINGFACE_HUB_TOKEN."
    exit 1
  fi

  if [ "$CALCULATOR" = "mace" ]; then
    source ~/project/.venv-mace/bin/activate
  elif [ "$CALCULATOR" = "uma" ]; then
    source ~/project/.venv-uma/bin/activate
  else
    echo "ERROR: unknown calculator $CALCULATOR"
    exit 1
  fi

  python -u scripts_python/runtime.py run \
  --tests "$TEST_CSV" \
  --summary-file "$SUMMARY_FILE"

  deactivate

  echo "Finished supercell task: float64 $CALCULATOR $MATERIAL r$REPEAT"
}

plot_mode() {
  load_env

  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
  export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
  export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
  export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
  export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

  echo "Supercell plot job"
  echo "Reading scratch outputs from: $SUPER_ROOT"
  echo "Writing project comprehensive outputs to: $SUPER_COMPREHENSIVE_DIR"

  source ~/project/.venv-mace/bin/activate

  python -u scripts_python/supercell.py combine --output-root "$SUPER_ROOT"

  mkdir -p "$SUPER_COMPREHENSIVE_DIR"

  python -u scripts_python/run_comprehensive.py \
    --mace-dir "$SUPER_ROOT/outputs_float64/mace" \
    --uma-dir "$SUPER_ROOT/outputs_float64/uma" \
    --output-dir "$SUPER_COMPREHENSIVE_DIR"

  python -u scripts_python/runtime.py plot \
    --mace-summary "$SUPER_ROOT/outputs_float64/mace/summary.csv" \
    --uma-summary "$SUPER_ROOT/outputs_float64/uma/summary.csv" \
    --output-dir "$SUPER_COMPREHENSIVE_DIR" \
    --epsilon 0.1

  deactivate

  echo "Finished supercell comprehensive plots:"
  echo "$SUPER_COMPREHENSIVE_DIR"
}

case "$MODE" in
  controller)
    controller_mode
    ;;
  run)
    run_mode
    ;;
  plot)
    plot_mode
    ;;
  *)
    echo "ERROR: unknown SUPERCELL_MODE=$MODE"
    exit 1
    ;;
esac