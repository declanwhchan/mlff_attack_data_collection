#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Repository root: $REPO_ROOT"

required_files=(
    "20_licohpf.xyz"
    "mace-mh-1.model"
    "uma-s-1p1.pt"
    "pot.almtp"
    "pot.almtp.elements"
    "MACE_model.model"
    "datasets/licohpf_database/tests_comprehensive.json"
    "pipeline/run_tests.py"
    "pipeline/setup_licohpf.py"
)

for path in "${required_files[@]}"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: missing required file: $path"
        exit 1
    fi
done

if [ -d "$HOME/project/.venv-mace" ]; then
    source "$HOME/project/.venv-mace/bin/activate"
fi

python -m py_compile \
    pipeline/run_tests.py \
    pipeline/setup_licohpf.py

python -m json.tool \
    datasets/licohpf_database/tests_comprehensive.json \
    >/dev/null

python pipeline/setup_licohpf.py

python - <<'PY'
from pathlib import Path
import csv
import json

root = Path.cwd()

config_path = (
    root
    / "datasets"
    / "licohpf_database"
    / "tests_comprehensive.json"
)

with config_path.open("r", encoding="utf-8") as handle:
    config = json.load(handle)

expected_counts = {
    "generated_licohpf_tests.csv": 15300,
    "generated_licohpf_cpu_tests.csv": 11900,
    "generated_licohpf_gpu_tests.csv": 3400,
}

for filename, expected_count in expected_counts.items():
    path = root / filename

    if not path.is_file():
        raise SystemExit(f"ERROR: missing generated file: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != expected_count:
        raise SystemExit(
            f"ERROR: {filename} has {len(rows)} rows; "
            f"expected {expected_count}"
        )

all_path = root / "generated_licohpf_tests.csv"

with all_path.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))

expected_combinations = {
    ("mace_mh", "float32"),
    ("mace_mh", "float64"),
    ("uma", "float32"),
    ("uma", "float64"),
    ("mtp", "float64"),
    ("chgnet", "float32"),
    ("chgnet", "float64"),
    ("mace_model", "float32"),
    ("mace_model", "float64"),
}

actual_combinations = {
    (row["model_id"], row["dtype_str"])
    for row in rows
}

if actual_combinations != expected_combinations:
    raise SystemExit(
        "ERROR: incorrect model/dtype combinations:\n"
        f"{sorted(actual_combinations)}"
    )

materials = {
    row["material_slug"]
    for row in rows
}

expected_structures = int(config["expected_structures"])

if len(materials) != expected_structures:
    raise SystemExit(
        f"ERROR: found {len(materials)} structures; "
        f"expected {expected_structures}"
    )

run_ids = [row["run_id"] for row in rows]

if len(run_ids) != len(set(run_ids)):
    raise SystemExit("ERROR: duplicate run_id values were generated")

for row in rows:
    model_id = row["model_id"]
    dtype_str = row["dtype_str"]
    device = row["device"]

    if model_id == "mtp":
        if dtype_str != "float64":
            raise SystemExit("ERROR: found float32 MTP row")
        if device != "cpu":
            raise SystemExit("ERROR: found non-CPU MTP row")

    if model_id == "mace_model":
        if not device.startswith("cuda"):
            raise SystemExit(
                "ERROR: found non-CUDA mace_model row"
            )
    elif device != "cpu":
        raise SystemExit(
            f"ERROR: found non-CPU {model_id} row"
        )

print("SETUP VALIDATION PASSED")
print(f"Structures: {len(materials)}")
print(f"All rows: {len(rows)}")
print("Float32 models: mace_mh, uma, chgnet, mace_model")
print("Float64 models: mace_mh, uma, mtp, chgnet, mace_model")
PY

SUPERCELL_ROOT="${SUPERCELL_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/licohpf_database/supercell}"

echo "Generating LiCOHPF supercell database in:"
echo "$SUPERCELL_ROOT"

python pipeline/supercell.py generate \
    --output-root "$SUPERCELL_ROOT"

python - <<PY
from pathlib import Path
import csv

root = Path("$SUPERCELL_ROOT")

expected = {
    "generated_supercell_tests.csv": 2400,
    "supercell_metadata.csv": 160,
}

for filename, expected_rows in expected.items():
    path = root / filename

    if not path.is_file():
        raise SystemExit(
            f"ERROR: missing {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != expected_rows:
        raise SystemExit(
            f"ERROR: {filename} has {len(rows)} rows; "
            f"expected {expected_rows}"
        )

print("SUPERCELL SETUP VALIDATION PASSED")
PY

if command -v deactivate >/dev/null 2>&1; then
    deactivate
fi

echo "LiCOHPF setup complete."
