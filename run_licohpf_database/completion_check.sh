#!/bin/bash

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

SCRATCH_BASE="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/licohpf_database}"
LOG_DIR="${DEBUG_LOG_DIR:-$REPO_ROOT}"
PYTHON="${PYTHON:-$HOME/project/.venv-mace/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python was not found:"
    echo "$PYTHON"
    exit 1
fi

export REPO_ROOT
export SCRATCH_BASE
export LOG_DIR

"$PYTHON" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv
import os
import re


repo_root = Path(os.environ["REPO_ROOT"])
scratch_base = Path(os.environ["SCRATCH_BASE"])
log_dir = Path(os.environ["LOG_DIR"])

trials = [
    "trial1_seed42",
    "trial2_seed43",
    "trial3_seed44",
    "trial4_seed45",
    "trial5_seed46",
]

materials = [
    f"licohpf_{number:03d}"
    for number in range(1, 21)
]

cpu_pairs = [
    ("mace_mh", "float32"),
    ("mace_mh", "float64"),
    ("uma", "float32"),
    ("uma", "float64"),
    ("mtp", "float64"),
    ("chgnet", "float32"),
    ("chgnet", "float64"),
]

gpu_pairs = [
    ("mace_model", "float32"),
    ("mace_model", "float64"),
]


def load_expected_rows(filename):
    path = repo_root / filename

    if not path.is_file():
        raise SystemExit(
            f"ERROR: required configuration file is missing: {path}"
        )

    counts = defaultdict(int)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        required = {
            "model_id",
            "dtype_str",
            "material_slug",
        }
        missing_columns = required.difference(
            reader.fieldnames or []
        )

        if missing_columns:
            raise SystemExit(
                f"ERROR: {path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            key = (
                row["model_id"].strip(),
                row["dtype_str"].strip(),
                row["material_slug"].strip(),
            )
            counts[key] += 1

    return counts


def index_logs(kind):
    pattern = re.compile(
        rf"^main-{re.escape(kind)}-(\d+)_(\d+)\.out$"
    )
    indexed = defaultdict(list)

    for path in log_dir.rglob(f"main-{kind}-*.out"):
        match = pattern.match(path.name)

        if match:
            task_id = int(match.group(2))
            indexed[task_id].append(path)

    return indexed


def log_contains(path, text):
    try:
        return text in path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return False


def compact_ranges(numbers):
    numbers = sorted(set(numbers))

    if not numbers:
        return "none"

    ranges = []
    start = numbers[0]
    previous = numbers[0]

    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue

        if start == previous:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{previous}")

        start = number
        previous = number

    if start == previous:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{previous}")

    return ",".join(ranges)


def write_ids(path, task_ids):
    path.write_text(
        ",".join(str(task_id) for task_id in task_ids),
        encoding="utf-8",
    )


def inspect(
    kind,
    pairs,
    configuration,
    success_marker,
):
    expected_counts = load_expected_rows(configuration)
    logs = index_logs(kind)

    expected_tasks = (
        len(trials)
        * len(materials)
        * len(pairs)
    )

    valid = []
    missing = []
    invalid = []

    successful_logs = []
    failed_only_logs = []
    no_logs = []

    expected_total_rows = 0
    valid_total_rows = 0

    for task_id in range(1, expected_tasks + 1):
        zero_index = task_id - 1
        tasks_per_trial = len(materials) * len(pairs)

        trial_index = zero_index // tasks_per_trial
        within_trial = zero_index % tasks_per_trial
        pair_index = within_trial // len(materials)
        material_index = within_trial % len(materials)

        trial = trials[trial_index]
        model_id, dtype_str = pairs[pair_index]
        material_slug = materials[material_index]

        key = (
            model_id,
            dtype_str,
            material_slug,
        )
        expected_rows = expected_counts.get(key, 0)
        expected_total_rows += expected_rows

        summary = (
            scratch_base
            / trial
            / "array_summaries"
            / (
                f"{dtype_str}_{model_id}_"
                f"{material_slug}_summary.csv"
            )
        )

        task_logs = logs.get(task_id, [])

        if not task_logs:
            no_logs.append(task_id)
        elif any(
            log_contains(path, success_marker)
            for path in task_logs
        ):
            successful_logs.append(task_id)
        else:
            failed_only_logs.append(task_id)

        if not summary.is_file():
            missing.append(task_id)
            continue

        try:
            with summary.open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            invalid.append(task_id)
            continue

        failed_rows = sum(
            1
            for row in rows
            if row.get(
                "status",
                "",
            ).strip().lower() != "success"
        )

        if (
            expected_rows == 0
            or len(rows) != expected_rows
            or failed_rows != 0
        ):
            invalid.append(task_id)
        else:
            valid.append(task_id)
            valid_total_rows += len(rows)

    rerun = sorted(set(missing + invalid))

    rerun_file = (
        repo_root
        / f"debug_{kind}_rerun_tasks.txt"
    )
    missing_file = (
        repo_root
        / f"debug_{kind}_missing_tasks.txt"
    )
    invalid_file = (
        repo_root
        / f"debug_{kind}_invalid_tasks.txt"
    )

    write_ids(rerun_file, rerun)
    write_ids(missing_file, missing)
    write_ids(invalid_file, invalid)

    print()
    print("=" * 68)
    print(f"{kind.upper()} CHECK")
    print("=" * 68)
    print(f"Expected task IDs:          1-{expected_tasks}")
    print(f"Expected task count:        {expected_tasks}")
    print(f"Expected individual rows:   {expected_total_rows}")
    print(f"Valid summary tasks:        {len(valid)}")
    print(f"Valid individual rows:      {valid_total_rows}")
    print(f"Missing summary tasks:      {len(missing)}")
    print(f"Invalid/failed summaries:   {len(invalid)}")
    print(f"TOTAL REQUIRING RERUN:      {len(rerun)}")
    print()
    print(f"Tasks with successful log:  {len(successful_logs)}")
    print(f"Tasks with failed-only log: {len(failed_only_logs)}")
    print(f"Tasks with no log present:  {len(no_logs)}")
    print()
    print(
        "Missing summary IDs: "
        f"{compact_ranges(missing)}"
    )
    print(
        "Invalid summary IDs: "
        f"{compact_ranges(invalid)}"
    )
    print(
        "All rerun IDs:       "
        f"{compact_ranges(rerun)}"
    )
    print()
    print(f"Rerun list:   {rerun_file}")
    print(f"Missing list: {missing_file}")
    print(f"Invalid list: {invalid_file}")

    return rerun


cpu_rerun = inspect(
    kind="cpu",
    pairs=cpu_pairs,
    configuration="generated_licohpf_cpu_tests.csv",
    success_marker="Finished CPU task successfully.",
)

gpu_rerun = inspect(
    kind="gpu",
    pairs=gpu_pairs,
    configuration="generated_licohpf_gpu_tests.csv",
    success_marker="Finished CUDA task successfully.",
)

total_rerun = len(cpu_rerun) + len(gpu_rerun)

print()
print("=" * 68)
print("OVERALL")
print("=" * 68)
print("Expected task count:        900")
print(f"CPU tasks requiring rerun:  {len(cpu_rerun)}")
print(f"GPU tasks requiring rerun:  {len(gpu_rerun)}")
print(f"TOTAL REQUIRING RERUN:      {total_rerun}")

if cpu_rerun:
    print()
    print("CPU rerun command:")
    print(
        'sbatch --time=4-00:00:00 '
        '--array="$(cat debug_cpu_rerun_tasks.txt)%80" '
        'run_licohpf_database/main.sh'
    )

if gpu_rerun:
    print()
    print("GPU rerun command:")
    print(
        'sbatch --time=4-00:00:00 '
        '--array="$(cat debug_gpu_rerun_tasks.txt)%40" '
        'run_licohpf_database/main_gpu.sh'
    )

if not cpu_rerun and not gpu_rerun:
    print()
    print("All 900 task summaries are complete and successful.")
PY
