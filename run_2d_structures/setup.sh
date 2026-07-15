#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=setup-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

MACE_PYTHON="$HOME/project/.venv-mace/bin/python"

CONFIG_FILE="datasets/2d_structures/tests_comprehensive.json"
MATERIALS_FILE="datasets/2d_structures/tests_materials.csv"

ALL_TESTS="generated_material_tests.csv"
CPU_TESTS="generated_material_cpu_tests.csv"
GPU_TESTS="generated_material_gpu_tests.csv"

SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/2d_structures}"
SUPERCELL_ROOT="${SUPERCELL_OUTPUT_ROOT:-$SCRATCH_OUTPUT_ROOT/supercell}"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ -z "${MP_API_KEY:-}" ]; then
    echo "ERROR: .env is missing MP_API_KEY"
    exit 1
fi

module load gcc/12.3 python/3.11 arrow

if [ ! -x "$MACE_PYTHON" ]; then
    echo "ERROR: MACE Python was not found:"
    echo "$MACE_PYTHON"
    exit 1
fi

required_files=(
    "$CONFIG_FILE"
    "$MATERIALS_FILE"
    "pipeline/setup_mpids.py"
    "pipeline/supercell.py"
    "mace-mh-1.model"
    "uma-s-1p1.pt"
    "pot.almtp"
    "pot.almtp.elements"
    "MACE_model.model"
)

for required_file in "${required_files[@]}"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Required file is missing:"
        echo "$REPO_ROOT/$required_file"
        exit 1
    fi
done

echo "Repository root: $REPO_ROOT"
echo "Scratch output root: $SCRATCH_OUTPUT_ROOT"
echo "Supercell root: $SUPERCELL_ROOT"
echo
echo "Generating complete five-model 2D database."

"$MACE_PYTHON" -u pipeline/setup_mpids.py \
    --materials "$MATERIALS_FILE" \
    --config "$CONFIG_FILE" \
    --tests-out "$ALL_TESTS" \
    --structures-dir mp_structures

"$MACE_PYTHON" - \
    "$ALL_TESTS" \
    "$CPU_TESTS" \
    "$GPU_TESTS" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path


all_path = Path(sys.argv[1])
cpu_path = Path(sys.argv[2])
gpu_path = Path(sys.argv[3])

required_columns = {
    "run_id",
    "material_label",
    "material_slug",
    "run_folder",
    "input_path",
    "model_id",
    "display_name",
    "calculator",
    "calculator_backend",
    "model_path",
    "attack_type",
    "epsilon",
    "n_steps",
    "device",
    "dtype_str",
}

expected_models = {
    "mace_mh",
    "uma",
    "mtp",
    "chgnet",
    "mace_model",
}

expected_backends = {
    "mace_mh": "mace",
    "uma": "uma",
    "mtp": "mtp",
    "chgnet": "chgnet",
    "mace_model": "mace",
}

cpu_models = {
    "mace_mh",
    "uma",
    "mtp",
    "chgnet",
}

gpu_models = {
    "mace_model",
}

with all_path.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames or []
    rows = list(reader)

missing_columns = required_columns.difference(
    fieldnames
)

if missing_columns:
    raise SystemExit(
        "ERROR: Generated test CSV is missing: "
        + ", ".join(sorted(missing_columns))
    )

if not rows:
    raise SystemExit(
        f"ERROR: {all_path} contains no rows"
    )

material_slugs = {
    row["material_slug"].strip()
    for row in rows
}

if len(material_slugs) != 20:
    raise SystemExit(
        "ERROR: Expected 20 materials, found "
        f"{len(material_slugs)}"
    )

present_models = {
    row["model_id"].strip().lower()
    for row in rows
}

if present_models != expected_models:
    raise SystemExit(
        "ERROR: Expected exactly these models: "
        + ", ".join(sorted(expected_models))
        + ". Found: "
        + ", ".join(sorted(present_models))
    )

run_ids = [
    row["run_id"].strip()
    for row in rows
]

if len(run_ids) != len(set(run_ids)):
    raise SystemExit(
        "ERROR: Duplicate run_id values were generated"
    )

cpu_rows = []
gpu_rows = []

for row_number, row in enumerate(
    rows,
    start=2,
):
    model_id = (
        row["model_id"].strip().lower()
    )
    backend = (
        row["calculator_backend"]
        .strip()
        .lower()
    )
    device = row["device"].strip().lower()
    dtype_str = (
        row["dtype_str"].strip().lower()
    )

    row["model_id"] = model_id
    row["calculator_backend"] = backend
    row["calculator"] = backend
    row["device"] = device
    row["dtype_str"] = dtype_str

    if backend != expected_backends[model_id]:
        raise SystemExit(
            f"ERROR: Row {row_number}: "
            f"{model_id} requires backend "
            f"{expected_backends[model_id]}, "
            f"not {backend}"
        )

    if dtype_str not in {
        "float32",
        "float64",
    }:
        raise SystemExit(
            f"ERROR: Row {row_number} has invalid "
            f"dtype {dtype_str!r}"
        )

    if model_id == "mtp":
        if dtype_str != "float64":
            raise SystemExit(
                f"ERROR: Row {row_number}: "
                "MTP must use float64"
            )

        if device != "cpu":
            raise SystemExit(
                f"ERROR: Row {row_number}: "
                "MTP must use CPU"
            )

    if model_id == "mace_model":
        if device != "cuda":
            raise SystemExit(
                f"ERROR: Row {row_number}: "
                "mace_model must use CUDA"
            )

        gpu_rows.append(row)

    else:
        if device != "cpu":
            raise SystemExit(
                f"ERROR: Row {row_number}: "
                f"{model_id} must use CPU"
            )

        cpu_rows.append(row)

if {
    row["model_id"]
    for row in cpu_rows
} != cpu_models:
    raise SystemExit(
        "ERROR: CPU rows do not contain exactly: "
        + ", ".join(sorted(cpu_models))
    )

if {
    row["model_id"]
    for row in gpu_rows
} != gpu_models:
    raise SystemExit(
        "ERROR: GPU rows do not contain exactly "
        "mace_model"
    )

for output_path, output_rows in (
    (cpu_path, cpu_rows),
    (gpu_path, gpu_rows),
):
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(output_rows)

counts = Counter(
    (
        row["model_id"],
        row["dtype_str"],
        row["device"],
    )
    for row in rows
)

print()
print("2D main database validation passed")
print(f"Materials: {len(material_slugs):,}")
print(f"All rows: {len(rows):,}")
print(f"CPU rows: {len(cpu_rows):,}")
print(f"GPU rows: {len(gpu_rows):,}")
print()
print("Model/dtype/device row counts:")

for key in sorted(counts):
    model_id, dtype_str, device = key

    print(
        f"  {model_id:12s} "
        f"{dtype_str:7s} "
        f"{device:4s}: "
        f"{counts[key]:,}"
    )
PY

echo
echo "Generating 2D supercell database in:"
echo "$SUPERCELL_ROOT"

mkdir -p "$SUPERCELL_ROOT"

"$MACE_PYTHON" -u pipeline/supercell.py generate \
    --output-root "$SUPERCELL_ROOT" \
    --materials "$ALL_TESTS" \
    --config "$CONFIG_FILE" \
    --structures-dir mp_structures

"$MACE_PYTHON" - "$SUPERCELL_ROOT" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path


root = Path(sys.argv[1])

tests_path = (
    root / "generated_supercell_tests.csv"
)
metadata_path = (
    root / "supercell_metadata.csv"
)

for path in (
    tests_path,
    metadata_path,
):
    if not path.is_file():
        raise SystemExit(
            f"ERROR: Missing supercell file: {path}"
        )

with tests_path.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as handle:
    tests = list(csv.DictReader(handle))

with metadata_path.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as handle:
    metadata = list(csv.DictReader(handle))

if len(tests) != 2400:
    raise SystemExit(
        "ERROR: Expected 2,400 supercell test rows, "
        f"found {len(tests):,}"
    )

if len(metadata) != 160:
    raise SystemExit(
        "ERROR: Expected 160 supercell metadata rows, "
        f"found {len(metadata):,}"
    )

models = {
    row["model_id"].strip().lower()
    for row in tests
}

expected_models = {
    "mace_mh",
    "uma",
    "mtp",
    "chgnet",
    "mace_model",
}

if models != expected_models:
    raise SystemExit(
        "ERROR: Incorrect supercell model set: "
        f"{sorted(models)}"
    )

dtypes = {
    row["dtype_str"].strip().lower()
    for row in tests
}

if dtypes != {"float64"}:
    raise SystemExit(
        "ERROR: Supercell database must contain "
        "only float64"
    )

base_materials = {
    row["base_material_slug"].strip()
    for row in metadata
}

if len(base_materials) != 20:
    raise SystemExit(
        "ERROR: Expected 20 supercell base "
        f"materials, found {len(base_materials)}"
    )

repeat_tuples = {
    row["repeat_tuple"].strip()
    for row in metadata
}

expected_repeats = {
    "1x1x1",
    "1x1x2",
    "1x2x1",
    "1x2x2",
    "2x1x1",
    "2x1x2",
    "2x2x1",
    "2x2x2",
}

if repeat_tuples != expected_repeats:
    raise SystemExit(
        "ERROR: Incorrect repeat tuples: "
        f"{sorted(repeat_tuples)}"
    )

counts = Counter(
    row["model_id"].strip().lower()
    for row in tests
)

for model_id in sorted(expected_models):
    if counts[model_id] != 480:
        raise SystemExit(
            f"ERROR: Expected 480 {model_id} "
            f"supercell rows, found "
            f"{counts[model_id]}"
        )

print("2D supercell validation passed")
print(f"Supercell tests: {len(tests):,}")
print(f"Supercell metadata: {len(metadata):,}")

for model_id in sorted(counts):
    print(
        f"  {model_id:12s}: "
        f"{counts[model_id]:,}"
    )
PY

echo
echo "$ALL_TESTS: $(($(wc -l < "$ALL_TESTS") - 1)) rows"
echo "$CPU_TESTS: $(($(wc -l < "$CPU_TESTS") - 1)) rows"
echo "$GPU_TESTS: $(($(wc -l < "$GPU_TESTS") - 1)) rows"
echo
echo "Supercell tests: $(($(wc -l < "$SUPERCELL_ROOT/generated_supercell_tests.csv") - 1)) rows"
echo "Supercell metadata: $(($(wc -l < "$SUPERCELL_ROOT/supercell_metadata.csv") - 1)) rows"
echo
echo "2D-structure setup complete."
