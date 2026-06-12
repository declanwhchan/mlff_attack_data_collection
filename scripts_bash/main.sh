#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-40%10

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

mkdir -p material_tests array_summaries

TASK_INFO=$(python -u - <<'PY'
import csv
import os

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

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
max_task_id = n_materials * 2

if task_id < 1 or task_id > max_task_id:
    raise SystemExit(f"ERROR: SLURM_ARRAY_TASK_ID must be 1..{max_task_id}, got {task_id}")

if task_id <= n_materials:
    calculator = "mace"
    material_slug = materials[task_id - 1]
else:
    calculator = "uma"
    material_slug = materials[task_id - n_materials - 1]

print(f"{calculator} {material_slug}")
PY
)

CALCULATOR=$(echo "$TASK_INFO" | awk '{print $1}')
MATERIAL_SLUG=$(echo "$TASK_INFO" | awk '{print $2}')

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
from pathlib import Path

material_slug = "$MATERIAL_SLUG"
calculator = "$CALCULATOR"

with open("generated_material_tests.csv", newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))

selected = [
    row for row in rows
    if row["material_slug"] == material_slug
    and row["model_path"].lower().startswith(calculator)
]

if not selected:
    raise SystemExit(f"ERROR: no rows selected for {calculator} {material_slug}")

Path("material_tests").mkdir(exist_ok=True)

output = Path("material_tests") / f"{calculator}_{material_slug}.csv"
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(selected)

print(f"Wrote {len(selected)} rows to {output}", flush=True)
PY

python -u - <<'PY'
import os
import torch

print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"), flush=True)
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"), flush=True)
print("TORCH_NUM_THREADS:", os.environ.get("TORCH_NUM_THREADS"), flush=True)
print("torch num threads:", torch.get_num_threads(), flush=True)
print("cuda available:", torch.cuda.is_available(), flush=True)
print("HF auth configured:", bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")), flush=True)
PY

echo "Running $CALCULATOR for $MATERIAL_SLUG"

SUMMARY_FILE="array_summaries/${CALCULATOR}_${MATERIAL_SLUG}_summary.csv" \
  python -u scripts_python/run_tests.py --tests "material_tests/${CALCULATOR}_${MATERIAL_SLUG}.csv"

deactivate

echo "Finished $CALCULATOR for $MATERIAL_SLUG"