#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-400%40
#SBATCH --output=main-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

module load gcc/12.3 python/3.11 arrow

if [ ! -f generated_material_tests.csv ]; then
  echo "ERROR: generated_material_tests.csv missing. Run setup.sh first."
  exit 1
fi

TASK_INFO=$(python -u - <<'PY'
import csv
import os

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

trials = [
    ("Trial 1 - 42", 42),
    ("Trial 2 - 43", 43),
    ("Trial 3 - 44", 44),
    ("Trial 4 - 45", 45),
    ("Trial 5 - 46", 46),
]

with open("generated_material_tests.csv", newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))

materials = []
seen = set()

for row in rows:
    slug = row["material_slug"]
    if slug not in seen:
        seen.add(slug)
        materials.append(slug)

n_materials = len(materials)
dtypes = ["float32", "float64"]
calculators = ["mace", "uma"]

tasks_per_trial = n_materials * len(calculators) * len(dtypes)
max_task_id = len(trials) * tasks_per_trial

if task_id < 1 or task_id > max_task_id:
    raise SystemExit(f"ERROR: SLURM_ARRAY_TASK_ID must be 1..{max_task_id}, got {task_id}")

index = task_id - 1
trial_index = index // tasks_per_trial
within_trial = index % tasks_per_trial

trial_name, seed = trials[trial_index]
dtype_str = dtypes[within_trial // (n_materials * len(calculators))]
within_dtype = within_trial % (n_materials * len(calculators))
calculator = calculators[within_dtype // n_materials]
material_slug = materials[within_dtype % n_materials]

print(f"{dtype_str} {calculator} {material_slug} {seed} {trial_name}")
PY
)

MLFF_DTYPE=$(echo "$TASK_INFO" | awk '{print $1}')
CALCULATOR=$(echo "$TASK_INFO" | awk '{print $2}')
MATERIAL_SLUG=$(echo "$TASK_INFO" | awk '{print $3}')
MLFF_SEED=$(echo "$TASK_INFO" | awk '{print $4}')
TRIAL_NAME=$(echo "$TASK_INFO" | cut -d' ' -f5-)

export MLFF_DTYPE
export MLFF_SEED
export MLFF_OUTPUT_ROOT="$TRIAL_NAME"

mkdir -p "$TRIAL_NAME/material_tests" "$TRIAL_NAME/array_summaries"

echo "Selected trial: $TRIAL_NAME"
echo "Selected dtype: $MLFF_DTYPE"
echo "Selected seed: $MLFF_SEED"
echo "Selected material: $MATERIAL_SLUG"
echo "Calculator: $CALCULATOR"
echo "CPU threads per task: $SLURM_CPUS_PER_TASK"

if [ "$CALCULATOR" = "uma" ] && [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  echo "ERROR: UMA requires HF_TOKEN in .env or HUGGINGFACE_HUB_TOKEN in the environment."
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

which python

python -u - <<PY
import csv
import os
from pathlib import Path

material_slug = "$MATERIAL_SLUG"
calculator = "$CALCULATOR"
dtype_str = "$MLFF_DTYPE"

with open("generated_material_tests.csv", newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))

selected = [
    dict(row, dtype_str=dtype_str)
    for row in rows
    if row["material_slug"] == material_slug
    and row["model_path"].lower().startswith(calculator)
]

if not selected:
    raise SystemExit(f"ERROR: no rows selected for {calculator} {material_slug}")

trial_name = os.environ["MLFF_OUTPUT_ROOT"]
output_dir = Path(trial_name) / "material_tests" / dtype_str
output_dir.mkdir(parents=True, exist_ok=True)

fieldnames = list(rows[0].keys())
if "dtype_str" not in fieldnames:
    fieldnames.append("dtype_str")

output = output_dir / f"{calculator}_{material_slug}.csv"
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(selected)

print(f"Wrote {len(selected)} rows to {output}", flush=True)
PY

python -u - <<'PY'
import os
import torch

print("MLFF_DTYPE:", os.environ.get("MLFF_DTYPE"), flush=True)
print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"), flush=True)
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"), flush=True)
print("TORCH_NUM_THREADS:", os.environ.get("TORCH_NUM_THREADS"), flush=True)
print("torch num threads:", torch.get_num_threads(), flush=True)
print("cuda available:", torch.cuda.is_available(), flush=True)
print("HF auth configured:", bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")), flush=True)
PY

echo "Running $MLFF_DTYPE $CALCULATOR for $MATERIAL_SLUG"

SUMMARY_FILE="${TRIAL_NAME}/array_summaries/${MLFF_DTYPE}_${CALCULATOR}_${MATERIAL_SLUG}_summary.csv" \
  python -u scripts_python/run_tests.py --tests "${TRIAL_NAME}/material_tests/${MLFF_DTYPE}/${CALCULATOR}_${MATERIAL_SLUG}.csv"

deactivate

echo "Finished $MLFF_DTYPE $CALCULATOR for $MATERIAL_SLUG"