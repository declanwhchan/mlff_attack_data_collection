#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=02:00:00
#SBATCH --mem=16G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

module load gcc/12.3 python/3.11 arrow

PYTHON="${PYTHON:-$HOME/project/.venv-mace/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found: $PYTHON"
    exit 1
fi

export MPLBACKEND=Agg
export REPO_ROOT

SCRATCH_COLLECTION_ROOT="${SCRATCH_COLLECTION_ROOT:-/scratch/$USER/mlff_attack_data_collection}"
PROJECT_RESULTS_ROOT="${PROJECT_RESULTS_ROOT:-$SCRATCH_COLLECTION_ROOT/licohpf_database_results}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/presentation_visuals/contour_relaxation_checks}"
FIGURE_PREFIX="${FIGURE_PREFIX:-contour_relaxation_checks}"

mkdir -p "$OUTPUT_DIR"
export OUTPUT_DIR FIGURE_PREFIX PROJECT_RESULTS_ROOT

"$PYTHON" - <<'PY'
import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ase.io import read as ase_read
from ase.geometry import find_mic

project_results_root = Path(os.environ["PROJECT_RESULTS_ROOT"]).resolve()
output_dir = Path(os.environ["OUTPUT_DIR"]).resolve()
figure_prefix = os.environ["FIGURE_PREFIX"]
debug = os.environ.get("DEBUG_CONTOUR_CHECK", "").strip().lower() in {"1", "true", "yes", "on"}

ATTACK_SUMMARY_FILES = sorted(
    list(project_results_root.glob("trial*/outputs_comprehensive/*/combined_dataset.csv"))
)
ATTACK_RUN_SUMMARY_FILES = sorted(
    project_results_root.glob(
        "trial*/outputs_comprehensive/**/summary.csv"
    )
)
CONTOUR_SUMMARY_FILES = sorted(
    project_results_root.glob("trial*/outputs_comprehensive/*/*/contour/summary.csv")
)

if not ATTACK_SUMMARY_FILES and not ATTACK_RUN_SUMMARY_FILES and not CONTOUR_SUMMARY_FILES:
    raise SystemExit(
        f"ERROR: no plot.sh summaries found under:\n{project_results_root}"
    )

def log(*args):
    if debug:
        print("[DEBUG]", *args)

def clean(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()

def as_float(value):
    try:
        text = clean(value)
        return None if text == "" else float(text)
    except Exception:
        return None

def as_bool(value):
    text = clean(value).lower()
    if text in {"1", "true", "yes", "y", "success", "ok", "passed"}:
        return True
    if text in {"0", "false", "no", "n", "failed", "failure", "error", "timeout"}:
        return False
    return None

def read_rows(paths):
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path, keep_default_na=False)
        except Exception:
            continue
        if not frame.empty:
            frame["_source_file"] = str(path)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

attack = read_rows(ATTACK_SUMMARY_FILES)
attack_runs = read_rows(ATTACK_RUN_SUMMARY_FILES)
contour = read_rows(CONTOUR_SUMMARY_FILES)

if not attack_runs.empty:
    keep = [
        "run_id",
        "status",
        "before_force_csv",
        "after_force_csv",
        "before_relax_traj",
        "after_attack_relax_traj",
        "actual_output_dir",
        "output_dir",
        "run_folder",
        "relax_fmax",
        "relax_max_steps",
    ]

    keep = [c for c in keep if c in attack_runs.columns]

    attack = attack.merge(
        attack_runs[keep],
        on="run_id",
        how="left",
    )

    matched = attack["before_force_csv"].notna().sum()
    print(f"Matched attack runs: {matched}/{len(attack)}")

if attack.empty and contour.empty:
    raise SystemExit("ERROR: no usable rows were loaded")

log("attack rows:", len(attack), "contour rows:", len(contour))
log("attack columns:", list(attack.columns))
log("contour columns:", list(contour.columns))

def first_existing(row, *keys):
    for key in keys:
        value = clean(row.get(key))
        if value != "":
            return value
    return ""

def pick_column(row, include=(), exclude=()):
    keys = list(row.index) if hasattr(row, "index") else list(row.keys())
    for key in keys:
        lk = str(key).lower()
        if include and not all(token in lk for token in include):
            continue
        if any(token in lk for token in exclude):
            continue
        if clean(row.get(key)) != "":
            return key
    return None

def candidate_run_directories(row):
    source_value = clean(row.get("_source_file"))
    source_dir = (
        Path(source_value).resolve().parent
        if source_value
        else None
    )

    directories = []

    for key in (
        "run_dir",
        "actual_output_dir",
        "output_dir",
    ):
        value = clean(row.get(key))

        if not value:
            continue

        candidate = Path(value)

        if candidate.is_absolute():
            directories.append(candidate)
        else:
            if source_dir is not None:
                directories.append(
                    source_dir / candidate
                )

            directories.append(
                project_results_root / candidate
            )

    run_id = clean(row.get("run_id"))

    if source_dir is not None:
        if run_id:
            directories.append(
                source_dir / run_id
            )

        directories.append(source_dir)

    result = []
    seen = set()

    for directory in directories:
        try:
            directory = directory.resolve()
        except Exception:
            continue

        if directory not in seen:
            seen.add(directory)
            result.append(directory)

    return result


def resolve_path(
    row,
    *preferred_keys,
    fallback_names=(),
):
    source_value = clean(
        row.get("_source_file")
    )

    source_dir = (
        Path(source_value).resolve().parent
        if source_value
        else None
    )

    run_directories = (
        candidate_run_directories(row)
    )

    candidates = []

    for key in preferred_keys:
        value = clean(row.get(key))

        if not value:
            continue

        value_path = Path(value)

        if value_path.is_absolute():
            candidates.append(value_path)
        else:
            if source_dir is not None:
                candidates.extend([
                    source_dir / value_path,
                    source_dir / value_path.name,
                ])

            for directory in run_directories:
                candidates.extend([
                    directory / value_path,
                    directory / value_path.name,
                ])

    for directory in run_directories:
        for filename in fallback_names:
            candidates.append(
                directory / filename
            )

    seen = set()

    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue

        if candidate in seen:
            continue

        seen.add(candidate)

        if candidate.is_file():
            return candidate

    return None


def find_step_count(row, stage="after"):
    # Only explicitly named relaxation-step columns are allowed.
    # n_steps is an attack parameter and must never be used here.
    if stage == "after":
        keys = (
            "after_relax_steps",
            "after_attack_relax_steps",
            "after_attack_relaxation_steps",
            "post_relax_steps",
        )
    else:
        keys = (
            "contour_relaxed_steps",
            "contour_endpoint_relaxation_steps",
            "contour_endpoint_relax_steps",
        )

    for key in keys:
        value = as_float(row.get(key))

        if value is not None:
            return value

    return None


def relaxation_cap(row, stage="after"):
    if stage == "after":
        keys = (
            "relax_max_steps",
            "after_relax_max_steps",
            "max_relax_steps",
        )
    else:
        keys = (
            "contour_post_relax_max_steps",
            "relax_max_steps",
            "max_relax_steps",
        )

    for key in keys:
        value = as_float(row.get(key))

        if value is not None and value > 0:
            return value

    return 600.0


def relaxation_fmax(row):
    for key in (
        "relax_fmax",
        "fmax",
        "force_convergence_threshold",
    ):
        value = as_float(row.get(key))

        if value is not None and value > 0:
            return value

    return 0.01


def relaxation_csv_evidence(row):
    path = resolve_path(
        row,
        "after_attack_relaxation_data_csv",
        "after_relaxation_data_csv",
        "after_relax_data_csv",
        fallback_names=(
            "after_attack_relaxation_data.csv",
        ),
    )

    if path is None:
        return None, None

    try:
        frame = pd.read_csv(path)
    except Exception:
        return None, None

    if frame.empty:
        return None, None

    step_column = next(
        (
            column
            for column in frame.columns
            if str(column).strip().lower()
            == "step"
        ),
        None,
    )

    force_column = next(
        (
            column
            for column in frame.columns
            if str(column)
            .strip()
            .lower()
            .startswith("max force")
        ),
        None,
    )

    steps = None
    final_force = None

    if step_column is not None:
        values = pd.to_numeric(
            frame[step_column],
            errors="coerce",
        ).dropna()

        if not values.empty:
            steps = float(values.iloc[-1])

    if force_column is not None:
        values = pd.to_numeric(
            frame[force_column],
            errors="coerce",
        ).dropna()

        if not values.empty:
            final_force = float(
                values.iloc[-1]
            )

    return steps, final_force


def infer_attack_converged(row):
    csv_steps, final_force = (
        relaxation_csv_evidence(row)
    )

    # Final force is the strongest available evidence.
    if (
        final_force is not None
        and np.isfinite(final_force)
    ):
        return (
            final_force
            <= relaxation_fmax(row)
        )

    # Use only post-attack convergence flags.
    for key in (
        "after_relax_converged",
        "after_attack_relax_converged",
        "after_attack_relaxation_converged",
        "after_converged",
    ):
        explicit = as_bool(row.get(key))

        if explicit is not None:
            return explicit

    status = first_existing(
        row,
        "after_relax_status",
        "after_attack_relax_status",
        "after_attack_relaxation_status",
        "status",
    ).lower()

    if status in {
        "failed",
        "failure",
        "error",
        "timeout",
    }:
        return False

    steps = csv_steps

    if steps is None:
        steps = find_step_count(
            row,
            stage="after",
        )

    if (
        steps is not None
        and steps
        >= relaxation_cap(
            row,
            stage="after",
        )
    ):
        return False

    # Being below the cap is not, by itself,
    # proof that the force criterion was met.
    return None


def infer_contour_converged(row):
    final_force = as_float(
        first_existing(
            row,
            "contour_endpoint_relaxation_max_force_ev_a",
            "contour_relaxed_max_force_ev_a",
        )
    )

    if (
        final_force is not None
        and np.isfinite(final_force)
    ):
        return (
            final_force
            <= relaxation_fmax(row)
        )

    for key in (
        "contour_relaxed_converged",
        "contour_endpoint_relaxation_converged",
        "contour_converged",
    ):
        explicit = as_bool(row.get(key))

        if explicit is not None:
            return explicit

    status = first_existing(
        row,
        "contour_relaxed_status",
        "contour_endpoint_relaxation_status",
        "status",
    ).lower()

    if status in {
        "failed",
        "failure",
        "error",
        "timeout",
    }:
        return False

    steps = find_step_count(
        row,
        stage="contour",
    )

    if (
        steps is not None
        and steps
        >= relaxation_cap(
            row,
            stage="contour",
        )
    ):
        return False

    return None


def load_force_table(path):
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None

    required = {
        "atom_index",
        "fx",
        "fy",
        "fz",
    }

    if not required.issubset(frame.columns):
        return None

    frame = frame[
        [
            "atom_index",
            "fx",
            "fy",
            "fz",
        ]
    ].copy()

    frame["atom_index"] = pd.to_numeric(
        frame["atom_index"],
        errors="coerce",
    )

    for column in ("fx", "fy", "fz"):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    return frame


def force_delta_after_relax(row):
    # Reference: final initially relaxed structure.
    before_path = resolve_path(
        row,
        "before_force_csv",
        "before_forces_csv",
        "before_forces",
        fallback_names=(
            "before_forces.csv",
        ),
    )

    # Comparison: final post-attack relaxed structure.
    after_path = resolve_path(
        row,
        "after_force_csv",
        "after_forces_csv",
        "after_forces",
        fallback_names=(
            "after_forces.csv",
        ),
    )

    if (
        before_path is None
        or after_path is None
    ):
        return None

    before = load_force_table(before_path)
    after = load_force_table(after_path)

    if before is None or after is None:
        return None

    merged = before.merge(
        after,
        on="atom_index",
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )

    if merged.empty:
        return None

    before_vectors = merged[
        [
            "fx_before",
            "fy_before",
            "fz_before",
        ]
    ].to_numpy(dtype=float)

    after_vectors = merged[
        [
            "fx_after",
            "fy_after",
            "fz_after",
        ]
    ].to_numpy(dtype=float)

    delta = np.linalg.norm(
        after_vectors - before_vectors,
        axis=1,
    )

    delta = delta[np.isfinite(delta)]

    if delta.size == 0:
        return None

    return float(np.median(delta))


def displacement_after_relax(row):
    # Reference: last frame of the initial relaxation.
    before_path = resolve_path(
        row,
        "before_relax_traj",
        "before_attack_relax_traj",
        "before_attack_relaxation_traj",
        fallback_names=(
            "before_attack_relaxation.traj",
            "before_relaxation.traj",
            "before_relax.traj",
        ),
    )

    # Comparison: last frame of recovery relaxation.
    after_path = resolve_path(
        row,
        "after_attack_relax_traj",
        "after_attack_relaxation_traj",
        "after_relax_traj",
        "after_relaxation_traj",
        fallback_names=(
            "after_attack_relaxation.traj",
            "after_relaxation.traj",
        ),
    )

    if (
        before_path is None
        or after_path is None
    ):
        return None

    try:
        before = ase_read(
            before_path,
            index=-1,
        )
        after = ase_read(
            after_path,
            index=-1,
        )
    except Exception:
        return None

    if len(before) != len(after):
        return None

    if (
        before.get_chemical_symbols()
        != after.get_chemical_symbols()
    ):
        return None

    vectors = np.asarray(
        after.positions - before.positions,
        dtype=float,
    )

    # Correct periodic boundary crossings.
    if (
        bool(np.any(before.pbc))
        and abs(
            float(
                np.linalg.det(
                    before.cell.array
                )
            )
        )
        > 1.0e-12
    ):
        try:
            vectors, _ = find_mic(
                vectors,
                cell=before.cell,
                pbc=before.pbc,
            )
        except Exception:
            return None

    displacement = np.linalg.norm(
        vectors,
        axis=1,
    )

    displacement = displacement[
        np.isfinite(displacement)
    ]

    if displacement.size == 0:
        return None

    return float(
        np.median(displacement)
    )


def contour_metric(row, *keys):
    for key in keys:
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None

def attack_metric(row, *keys):
    for key in keys:
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None

def setup_ax(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1F2328")
    ax.spines["bottom"].set_color("#1F2328")
    ax.tick_params(direction="out", length=6, width=1.3, colors="#1F2328", pad=5)
    ax.grid(True, axis="y", color="#D9E1E8", linewidth=1.0, alpha=0.85)
    ax.set_axisbelow(True)

def save_fig(fig, name):
    path = output_dir / f"{figure_prefix}_{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none", dpi=300)
    plt.close(fig)
    print(f"Created: {path}")

PLOT_GROUPS = [
    ("Contour", "#2166AC"),
    ("FGSM", "#E41A1C"),
    ("I-FGSM", "#FF7F00"),
    ("PGD", "#984EA3"),
]


def attack_group_label(row):
    label = clean(row.get("attack_label")).upper()

    if label in {"FGSM", "I-FGSM", "PGD"}:
        return label

    attack_type = clean(row.get("attack_type")).strip().lower()
    n_steps = as_float(row.get("n_steps"))

    if attack_type == "fgsm":
        if n_steps is not None and n_steps > 1:
            return "I-FGSM"
        return "FGSM"

    if attack_type == "pgd":
        return "PGD"

    return label if label in {"FGSM", "I-FGSM", "PGD"} else ""


def empty_group_map():
    return {label: [] for label, _ in PLOT_GROUPS}


def collect_grouped_values(attack_value_fn, contour_value_fn):
    grouped = empty_group_map()

    for _, row in contour.iterrows():
        value = contour_value_fn(row)
        if value is not None and np.isfinite(value):
            grouped["Contour"].append(value)

    for _, row in attack.iterrows():
        label = attack_group_label(row)
        if label not in grouped:
            continue

        value = attack_value_fn(row)
        if value is not None and np.isfinite(value):
            grouped[label].append(value)

    return grouped


def plot_bar(name, grouped_vals, ylabel, title):
    fig, ax = plt.subplots(figsize=(11.2, 6.8), facecolor="white")
    setup_ax(ax)

    labels = [label for label, _ in PLOT_GROUPS]
    colors = [color for _, color in PLOT_GROUPS]
    vals = [
        np.asarray(grouped_vals.get(label, []), dtype=float)
        for label in labels
    ]
    vals = [v[np.isfinite(v)] for v in vals]
    means = [np.nan if v.size == 0 else float(np.nanmean(v)) for v in vals]

    positions = np.arange(len(labels))
    bars = ax.bar(positions, means, color=colors, width=0.62, alpha=0.94)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Stress Tests")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ymax = 100.0 if "Converged" in ylabel else max([m for m in means if np.isfinite(m)] + [1.0]) * 1.25
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    ax.set_ylim(0, ymax)

    for rect, mean in zip(bars, means):
        if np.isfinite(mean):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                min(mean + 0.02 * ymax, ymax * 0.98),
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color="#1F2328",
            )

    save_fig(fig, name)


def plot_box(name, grouped_vals, ylabel, title, logy=False):
    fig, ax = plt.subplots(figsize=(11.2, 6.8), facecolor="white")
    setup_ax(ax)

    labels = [label for label, _ in PLOT_GROUPS]
    colors = [color for _, color in PLOT_GROUPS]
    series = [
        np.asarray(
            [
                v
                for v in grouped_vals.get(label, [])
                if v is not None and np.isfinite(v)
            ],
            dtype=float,
        )
        for label in labels
    ]

    if logy:
        series = [vals[vals > 0] for vals in series]

    if all(vals.size == 0 for vals in series):
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        save_fig(fig, name)
        return

    data = [vals if vals.size else np.array([np.nan]) for vals in series]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        zorder=3,
        medianprops={"color": "white", "linewidth": 2.0},
        whiskerprops={"linewidth": 1.7},
        capprops={"linewidth": 1.7},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)
        patch.set_linewidth(2.0)
        patch.set_zorder(3)

    rng = np.random.default_rng(7)
    for i, vals in enumerate(series):
        if vals.size:
            jitter = rng.uniform(-0.09, 0.09, size=len(vals))
            ax.scatter(
                np.full(len(vals), i + 1) + jitter,
                vals,
                s=14,
                alpha=0.25,
                color=colors[i],
                linewidths=0,
                zorder=2,
            )
        else:
            ax.text(
                i + 1,
                0.05,
                "no data",
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                fontsize=11,
            )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Stress Tests")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if logy and any(vals.size for vals in series):
        ax.set_yscale("log")

    save_fig(fig, name)


def log(*args):
    if debug:
        print("[DEBUG]", *args)


if debug:
    if not attack.empty:
        r = attack.iloc[0]
        log("attack sample source:", r.get("_source_file"))
        for k in (
            "after_relax_steps",
            "before_force_csv",
            "after_force_csv",
            "before_relax_traj",
            "after_attack_relax_traj",
            "neighbor_jaccard_distance",
            "rdf_l1_distance",
            "coordination_change_max",
        ):
            log("attack", k, "=", clean(r.get(k)))
    if not contour.empty:
        r = contour.iloc[0]
        log("contour sample source:", r.get("_source_file"))
        for k in (
            "contour_relaxed_steps",
            "contour_endpoint_relaxation_steps",
            "contour_relaxed_median_force_delta_ev_a",
            "contour_relaxed_median_displacement_a",
            "contour_relaxed_coordination_change_max",
            "contour_relaxed_neighbor_jaccard_distance",
            "contour_relaxed_rdf_l1_distance",
        ):
            log("contour", k, "=", clean(r.get(k)))

attack_status = [
    infer_attack_converged(row)
    for _, row in attack.iterrows()
]

contour_status = [
    infer_contour_converged(row)
    for _, row in contour.iterrows()
]

attack_known = [
    value
    for value in attack_status
    if value is not None
]

contour_known = [
    value
    for value in contour_status
    if value is not None
]

convergence_groups = empty_group_map()

for _, row in attack.iterrows():
    label = attack_group_label(row)
    value = infer_attack_converged(row)

    if label not in convergence_groups or value is None:
        continue

    convergence_groups[label].append(100.0 if value else 0.0)

for value in [100.0 if value else 0.0 for value in contour_known]:
    convergence_groups["Contour"].append(value)

print(
    "Convergence values:",
    ", ".join(
        f"{label}={len(convergence_groups[label])}"
        for label, _ in PLOT_GROUPS
    ),
)

attack_steps = [
    find_step_count(
        row,
        stage="after",
    )
    for _, row in attack.iterrows()
]

attack_steps = np.asarray(
    [
        value
        for value in attack_steps
        if value is not None
        and np.isfinite(value)
    ],
    dtype=float,
)

if attack_steps.size:
    print(
        "Attack relaxation steps "
        f"(min/median/max): "
        f"{np.min(attack_steps):g} / "
        f"{np.median(attack_steps):g} / "
        f"{np.max(attack_steps):g}"
    )

    print(
        "Attack rows at or above their "
        "reported/default cap:",
        sum(
            steps
            >= relaxation_cap(
                row,
                stage="after",
            )
            for steps, (_, row)
            in zip(
                [
                    find_step_count(
                        row,
                        stage="after",
                    )
                    for _, row
                    in attack.iterrows()
                    if find_step_count(
                        row,
                        stage="after",
                    )
                    is not None
                ],
                [
                    item
                    for item
                    in attack.iterrows()
                    if find_step_count(
                        item[1],
                        stage="after",
                    )
                    is not None
                ],
            )
        ),
    )
else:
    print(
        "Attack relaxation steps: "
        "no explicit post-attack step data"
    )

plot_bar(
    "01_convergence_rate",
    convergence_groups,
    "Converged cases (%)",
    "Convergence rate after relaxation",
)

def attack_force_delta_value(row):
    value = attack_metric(
        row,
        "attack_median_force_delta_ev_a",
        "after_attack_after_relaxation_median_force_delta_ev_a",
    )
    return value if value is not None else force_delta_after_relax(row)


def attack_displacement_value(row):
    value = attack_metric(
        row,
        "attack_median_displacement_a",
        "after_attack_after_relaxation_median_displacement_a",
    )
    return value if value is not None else displacement_after_relax(row)


delta_force_groups = collect_grouped_values(
    attack_force_delta_value,
    lambda row: contour_metric(
        row,
        "contour_relaxed_median_force_delta_ev_a",
        "contour_relaxed_force_delta_ev_a",
        "contour_relaxed_delta_force_ev_a",
        "contour_median_force_delta_ev_a",
    ),
)

disp_groups = collect_grouped_values(
    attack_displacement_value,
    lambda row: contour_metric(
        row,
        "contour_relaxed_median_displacement_a",
        "contour_relaxed_displacement_a",
        "contour_median_displacement_a",
    ),
)

print(
    "Delta force values:",
    ", ".join(
        f"{label}={len(delta_force_groups[label])}"
        for label, _ in PLOT_GROUPS
    ),
)
print(
    "Displacement values:",
    ", ".join(
        f"{label}={len(disp_groups[label])}"
        for label, _ in PLOT_GROUPS
    ),
)

plot_box(
    "02_delta_force",
    delta_force_groups,
    r"Median $\Delta$ force (eV/$\AA$)",
    "Force change after relaxation",
    logy=True,
)

plot_box(
    "03_displacement",
    disp_groups,
    r"Median displacement ($\AA$)",
    "Displacement after relaxation",
    logy=True,
)

topology_specs = [
    (
        "04_topology_coordination_number",
        "Maximum coordination change",
        ("coordination_change_max",),
        ("contour_relaxed_coordination_change_max",),
    ),
    (
        "05_topology_jaccard",
        "Neighbor Jaccard distance",
        ("neighbor_jaccard_distance",),
        ("contour_relaxed_neighbor_jaccard_distance",),
    ),
    (
        "06_topology_rdf",
        "RDF L1 distance",
        ("rdf_l1_distance",),
        ("contour_relaxed_rdf_l1_distance",),
    ),
]

for fname, ylabel, attack_keys, contour_keys in topology_specs:
    grouped_vals = collect_grouped_values(
        lambda row, keys=attack_keys: attack_metric(row, *keys),
        lambda row, keys=contour_keys: contour_metric(row, *keys),
    )
    plot_box(fname, grouped_vals, ylabel, ylabel, logy=False)

print("Done.")

PY

