#!/bin/bash
#SBATCH --account=def-j3goals
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-40%4

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load gcc/12.3 python/3.11 cuda/12.6 arrow

if [ ! -f generated_material_tests.csv ]; then
  echo "ERROR: generated_material_tests.csv missing. Run setup.sh first."
  exit 1
fi

mkdir -p material_tests array_summaries

TASK_INFO=$(python - <<'PY'
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

python - <<PY
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

print(f"Wrote {len(selected)} rows to {output}")
PY

echo "Running $CALCULATOR for $MATERIAL_SLUG"

if [ "$CALCULATOR" = "mace" ]; then
  source ~/project/.venv-mace/bin/activate
elif [ "$CALCULATOR" = "uma" ]; then
  source ~/project/.venv-uma/bin/activate
else
  echo "ERROR: unknown calculator $CALCULATOR"
  exit 1
fi

SUMMARY_FILE="array_summaries/${CALCULATOR}_${MATERIAL_SLUG}_summary.csv" \
  python scripts_python/run_tests.py --tests "material_tests/${CALCULATOR}_${MATERIAL_SLUG}.csv"

deactivate

echo "Finished $CALCULATOR for $MATERIAL_SLUG"