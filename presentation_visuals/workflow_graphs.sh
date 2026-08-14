#!/bin/bash

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

module load gcc/12.3 python/3.11 arrow

PYTHON="${PYTHON:-$HOME/project/.venv-mace/bin/python}"
TRIAL="${TRIAL:-trial1_seed42}"
SCRATCH_ROOT="${SCRATCH_OUTPUT_ROOT:-/scratch/$USER/mlff_attack_data_collection/2d_structures}"
MATERIAL_SLUG="${MATERIAL_SLUG:-berlinite_alpo4}"
MODEL_ID="${MODEL_ID:-mace_mh}"
MLFF_DTYPE="${MLFF_DTYPE:-float64}"
OUTPUT_DIR="${OUTPUT_DIR:-visualize_${MATERIAL_SLUG}}"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found: $PYTHON"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR/02_delta_force_vs_epsilon.png"

export MPLBACKEND=Agg
export REPO_ROOT TRIAL SCRATCH_ROOT MATERIAL_SLUG MODEL_ID MLFF_DTYPE OUTPUT_DIR

"$PYTHON" - <<'PY'
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ase.io import read
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

repo_root = Path(os.environ["REPO_ROOT"]).resolve()
trial = os.environ["TRIAL"]
material_slug = os.environ["MATERIAL_SLUG"]
model_id = os.environ["MODEL_ID"]
mlff_dtype = os.environ["MLFF_DTYPE"]

scratch_trial = (
    Path(os.environ["SCRATCH_ROOT"])
    / trial
).resolve()

output_dir = Path(os.environ["OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

summary_directory = scratch_trial / "array_summaries"

# ``mace`` was used as the MACE-MH identifier by early 2D runs. Newer
# configuration and summaries use the explicit ``mace_mh`` identifier.
model_id_aliases = {
    "mace": {"mace", "mace_mh", "mace-mh"},
    "mace_mh": {"mace", "mace_mh", "mace-mh"},
    "mace-mh": {"mace", "mace_mh", "mace-mh"},
}
model_ids_to_match = model_id_aliases.get(
    model_id.lower(),
    {model_id.lower()},
)

summary_candidates = [
    summary_directory / f"{mlff_dtype}_{candidate}_{material_slug}_summary.csv"
    for candidate in [
        model_id.lower(),
        *sorted(model_ids_to_match - {model_id.lower()}),
    ]
]

summary_path = next(
    (candidate for candidate in summary_candidates if candidate.is_file()),
    summary_candidates[0],
)

fmax = 0.05

BLUE = "#2166AC"
PERTURB_RED = "#C43C39"
THRESHOLD_COLOR = "#C44E52"
TEXT = "#202326"
SECONDARY = "#5A6168"

INITIAL_BACKGROUND = "#EAF2FA"
PERTURB_BACKGROUND = "#F8E4E2"
RECOVERY_BACKGROUND = "#EDF6F1"

# Reversed Viridis:
# lighter colours = smaller epsilon
# darker colours = larger epsilon
RECOVERY_CMAP = plt.get_cmap("viridis_r")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 22,
    "axes.labelsize": 30,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 19,
    "axes.linewidth": 2.1,
    "savefig.dpi": 300,
})


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------

def clean_string(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def candidate_directories(row):
    directories = []

    for column in (
        "actual_output_dir",
        "output_dir",
    ):
        value = clean_string(
            row.get(column, "")
        )

        if not value:
            continue

        path = Path(value)

        if path.is_absolute():
            directories.append(path)
        else:
            directories.extend([
                scratch_trial / path,
                repo_root / path,
            ])

    # Reconstruct the standard 2D-workflow location. This covers legacy
    # summaries whose stored paths were relative to a different submit
    # directory, as well as the historical ``mace`` output folder.
    row_dtype = clean_string(
        row.get("dtype_str", mlff_dtype)
    ).lower() or mlff_dtype.lower()
    row_material = clean_string(
        row.get("material_slug", material_slug)
    ).lower() or material_slug.lower()
    run_folder = clean_string(
        row.get("run_folder", row.get("run_id", ""))
    )

    if run_folder:
        for candidate_model_id in model_ids_to_match:
            directories.append(
                scratch_trial
                / f"outputs_{row_dtype}"
                / candidate_model_id
                / row_material
                / run_folder
            )

    return directories


def resolve_trajectory(
    row,
    column,
    default_name,
):
    candidates = []

    raw = clean_string(
        row.get(column, "")
    )

    if raw:
        raw_path = Path(raw)

        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend([
                repo_root / raw_path,
                scratch_trial / raw_path,
                summary_path.parent / raw_path,
            ])

            for directory in candidate_directories(row):
                candidates.append(
                    directory / raw_path
                )

    for directory in candidate_directories(row):
        candidates.append(
            directory / default_name
        )

    checked = set()

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate in checked:
            continue

        checked.add(candidate)

        if candidate.is_file():
            return candidate

    return None


def trajectory_maximum_forces(path):
    frames = read(
        str(path),
        index=":",
    )

    if not isinstance(frames, list):
        frames = [frames]

    values = []

    for frame in frames:
        try:
            forces = np.asarray(
                frame.get_forces(),
                dtype=float,
            )
        except Exception:
            values.append(np.nan)
            continue

        if (
            forces.ndim != 2
            or forces.shape[1] != 3
        ):
            values.append(np.nan)
            continue

        magnitudes = np.linalg.norm(
            forces,
            axis=1,
        )

        finite = magnitudes[
            np.isfinite(magnitudes)
        ]

        values.append(
            float(np.max(finite))
            if finite.size
            else np.nan
        )

    return np.asarray(
        values,
        dtype=float,
    )


def truncate_at_threshold(values, threshold):
    """End a trajectory at its first finite force at or below threshold."""
    values = np.asarray(values, dtype=float)
    reached = np.flatnonzero(
        np.isfinite(values)
        & (values <= threshold)
    )

    if not reached.size:
        return values

    # Show the endpoint on the displayed convergence line, rather than
    # continuing to lower values recorded after the stopping criterion.
    return np.concatenate([
        values[:int(reached[0])],
        np.asarray([threshold], dtype=float),
    ])

def final_finite_value(values):
    finite = np.flatnonzero(
        np.isfinite(values)
    )

    if not finite.size:
        return np.nan, None

    index = int(finite[-1])

    return float(values[index]), index


def decade_limits(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
        & (values > 0)
    ]

    if not values.size:
        return 1.0e-2, 1.0

    lower = 10.0 ** math.floor(
        math.log10(
            float(np.min(values))
        )
    )

    upper = 10.0 ** math.ceil(
        math.log10(
            float(np.max(values))
        )
    )

    if upper <= lower:
        upper = lower * 10.0

    return lower, upper


# ---------------------------------------------------------------------
# Load applicable 2D-material MACE-MH rows
# ---------------------------------------------------------------------

if not summary_path.is_file():
    raise SystemExit(
        "ERROR: missing summary:\n"
        f"{summary_path}"
    )

summary = pd.read_csv(summary_path)

material_slugs = summary.get(
    "material_slug",
    pd.Series(
        "",
        index=summary.index,
    ),
).astype(str).str.lower()

model_ids = summary.get(
    "model_id",
    pd.Series(
        "",
        index=summary.index,
    ),
).astype(str).str.lower()

# Older summaries sometimes recorded only the calculator backend. For
# MACE-MH this is ``mace``, so use it as an additional identity field.
calculator_ids = summary.get(
    "calculator",
    pd.Series(
        "",
        index=summary.index,
    ),
).astype(str).str.lower()

dtype_values = summary.get(
    "dtype_str",
    pd.Series(
        "",
        index=summary.index,
    ),
).astype(str).str.lower()

attack_types = summary.get(
    "attack_type",
    pd.Series(
        "",
        index=summary.index,
    ),
).astype(str).str.lower()

mask = (
    material_slugs.eq(material_slug.lower())
    & (
        model_ids.isin(model_ids_to_match)
        | calculator_ids.isin(model_ids_to_match)
    )
    & dtype_values.eq(mlff_dtype.lower())
    & attack_types.eq("fgsm")
)

summary = summary.loc[mask].copy()

if "status" in summary.columns:
    failed = (
        summary["status"]
        .astype(str)
        .str.lower()
        .eq("failed")
    )

    summary = summary.loc[
        ~failed
    ].copy()

# Prefer the one-step FGSM cases.
if "n_steps" in summary.columns:
    steps = pd.to_numeric(
        summary["n_steps"],
        errors="coerce",
    )

    single_step = summary.loc[
        steps.eq(1)
    ].copy()

    if not single_step.empty:
        summary = single_step

summary["epsilon_numeric"] = pd.to_numeric(
    summary["epsilon"],
    errors="coerce",
)

summary = summary.dropna(
    subset=["epsilon_numeric"]
).sort_values(
    "epsilon_numeric"
)

if summary.empty:
    raise SystemExit(
        "ERROR: no applicable MACE-model "
        "FGSM rows were found"
    )


# ---------------------------------------------------------------------
# Load initial and recovery trajectories
# ---------------------------------------------------------------------

initial_force = None
recovery_records = []
seen_epsilons = set()

for _, row in summary.iterrows():
    epsilon = float(
        row["epsilon_numeric"]
    )

    epsilon_key = round(
        epsilon,
        12,
    )

    if epsilon_key in seen_epsilons:
        continue

    before_path = resolve_trajectory(
        row,
        "before_relax_traj",
        "before_attack_relaxation.traj",
    )

    after_path = resolve_trajectory(
        row,
        "after_attack_relax_traj",
        "after_attack_relaxation.traj",
    )

    if (
        before_path is None
        or after_path is None
    ):
        print(
            "WARNING: missing trajectories "
            f"for epsilon={epsilon:g}"
        )
        continue

    try:
        if initial_force is None:
            initial_force = truncate_at_threshold(
                trajectory_maximum_forces(
                    before_path
                ),
                fmax,
            )

        recovery_force = truncate_at_threshold(
            trajectory_maximum_forces(
                after_path
            ),
            fmax,
        )
    except Exception as error:
        print(
            "WARNING: could not load "
            f"epsilon={epsilon:g}: {error}"
        )
        continue

    final_force, final_index = (
        final_finite_value(
            recovery_force
        )
    )

    if final_index is None:
        continue

    recovery_records.append({
        "epsilon": epsilon,
        "force": recovery_force,
        "converged": (
            np.isfinite(final_force)
            and final_force <= fmax
        ),
    })

    seen_epsilons.add(
        epsilon_key
    )


if initial_force is None:
    raise SystemExit(
        "ERROR: initial relaxation could not be loaded"
    )

if not recovery_records:
    raise SystemExit(
        "ERROR: no recovery trajectories could be loaded"
    )

recovery_records.sort(
    key=lambda record: record["epsilon"]
)

epsilon_values = np.asarray([
    record["epsilon"]
    for record in recovery_records
], dtype=float)

positive_epsilons = epsilon_values[
    epsilon_values > 0
]

if not positive_epsilons.size:
    raise SystemExit(
        "ERROR: epsilon values must be positive"
    )

minimum_epsilon = float(
    np.min(positive_epsilons)
)

maximum_epsilon = float(
    np.max(positive_epsilons)
)

color_norm = mcolors.LogNorm(
    vmin=minimum_epsilon,
    vmax=maximum_epsilon,
)


# ---------------------------------------------------------------------
# Construct separate initial and recovery axes
# ---------------------------------------------------------------------

initial_steps = np.arange(
    len(initial_force),
    dtype=float,
)

for record in recovery_records:
    # Recovery iterations restart at zero for every epsilon.
    record["steps"] = np.arange(
        len(record["force"]),
        dtype=float,
    )

maximum_initial_step = max(
    len(initial_force) - 1,
    1,
)

maximum_recovery_step = max(
    len(record["force"]) - 1
    for record in recovery_records
)


# ---------------------------------------------------------------------
# Determine shared force scale
# ---------------------------------------------------------------------

all_force_values = [
    initial_force[np.isfinite(initial_force)],
    np.asarray([fmax]),
]

for record in recovery_records:
    finite = record["force"][
        np.isfinite(record["force"])
        & (record["force"] > 0)
    ]

    if finite.size:
        all_force_values.append(finite)

force_lower, force_upper = decade_limits(
    np.concatenate(all_force_values)
)

force_lower = min(
    force_lower,
    1.0e-2,
)


# ---------------------------------------------------------------------
# Plot two separate panels
# ---------------------------------------------------------------------

figure = plt.figure(
    figsize=(16.0, 8.5),
    facecolor="white",
)

figure.supxlabel(
    "L-BFGS optimizer steps",
    x=0.49,
    y=0.035,
    fontsize=30,
    color=TEXT,
    ha="center",
)

grid = figure.add_gridspec(
    nrows=1,
    ncols=3,
    width_ratios=[1.0, 1.0, 0.035],
    left=0.105,
    right=0.91,
    bottom=0.19,
    top=0.95,
    wspace=0.17,
)

initial_axis = figure.add_subplot(
    grid[0, 0]
)

recovery_axis = figure.add_subplot(
    grid[0, 1],
    sharey=initial_axis,
)

colorbar_axis = figure.add_subplot(
    grid[0, 2]
)

for axis in (
    initial_axis,
    recovery_axis,
):
    axis.set_facecolor("white")
    axis.grid(False)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.spines["left"].set_color(TEXT)
    axis.spines["bottom"].set_color(TEXT)
    axis.spines["left"].set_linewidth(2.2)
    axis.spines["bottom"].set_linewidth(2.2)

    axis.tick_params(
        direction="out",
        length=8,
        width=2.0,
        colors=TEXT,
        pad=8,
    )

    axis.set_yscale("log")
    axis.set_ylim(
        force_lower,
        force_upper,
    )

    axis.axhline(
        fmax,
        color=THRESHOLD_COLOR,
        linewidth=3.0,
        linestyle=(0, (8, 5)),
        zorder=2,
    )


# ---------------------------------------------------------------------
# Left panel: initial relaxation
# ---------------------------------------------------------------------

initial_axis.plot(
    initial_steps,
    initial_force,
    color=BLUE,
    linewidth=4.0,
    linestyle="-",
    alpha=1.0,
    solid_capstyle="round",
    zorder=5,
)

# Keep the initial relaxation panel at its actual final optimizer step.
initial_x_upper = max(
    float(maximum_initial_step),
    1.0,
)
initial_tick_spacing = 1 if initial_x_upper <= 20 else 2

initial_axis.set_xlim(
    0,
    initial_x_upper,
)

initial_axis.set_xticks(
    np.arange(
        0,
        initial_x_upper + 1,
        initial_tick_spacing,
    )
)

initial_axis.set_ylabel(
    r"Max atomic force (eV/$\AA$)",
    labelpad=20,
)


# ---------------------------------------------------------------------
# Right panel: recovery trajectories
# ---------------------------------------------------------------------

for record in recovery_records:
    color = RECOVERY_CMAP(
        color_norm(record["epsilon"])
    )

    recovery_axis.plot(
        record["steps"],
        record["force"],
        color=color,
        linewidth=2.8,
        linestyle="-",
        alpha=0.82,
        solid_capstyle="round",
        zorder=5,
    )

recovery_x_upper = 300

recovery_axis.set_xlim(
    0,
    recovery_x_upper,
)

recovery_axis.set_xticks(
    [0, 100, 200, 300]
)

# The y-axis is shared, so labels are needed only on the left panel.
recovery_axis.tick_params(
    axis="y",
    which="both",
    left=False,
    labelleft=False,
)

recovery_axis.spines["left"].set_visible(
    False
)


# ---------------------------------------------------------------------
# Convergence-criterion annotation
# ---------------------------------------------------------------------

recovery_axis.text(
    1,
    fmax / 1.6,
    rf"$f_{{\max}}={fmax:g}\,\mathrm{{eV/\AA}}$",
    transform=recovery_axis.get_yaxis_transform(),
    ha="right",
    va="top",
    fontsize=20,
    color=THRESHOLD_COLOR,
    bbox={
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.88,
        "pad": 2.0,
    },
    zorder=10,
)


# ---------------------------------------------------------------------
# Epsilon colour bar
# ---------------------------------------------------------------------

scalar_mappable = ScalarMappable(
    norm=color_norm,
    cmap=RECOVERY_CMAP,
)

scalar_mappable.set_array([])

colorbar = figure.colorbar(
    scalar_mappable,
    cax=colorbar_axis,
    orientation="vertical",
)

colorbar.ax.set_title(
    r"$\epsilon$",
    fontsize=30,
    pad=12,
)

colorbar.ax.tick_params(
    labelsize=20,
    length=6,
    width=1.6,
)

colorbar.outline.set_linewidth(1.5)
colorbar.outline.set_edgecolor(TEXT)

candidate_ticks = np.asarray([
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
])

colorbar_ticks = candidate_ticks[
    (candidate_ticks >= minimum_epsilon)
    & (candidate_ticks <= maximum_epsilon)
]

if colorbar_ticks.size:
    colorbar.set_ticks(
        colorbar_ticks
    )

    colorbar.set_ticklabels([
        f"{value:g}"
        for value in colorbar_ticks
    ])

# ---------------------------------------------------------------------
# Save with a solid white background
# ---------------------------------------------------------------------

output_path = (
    output_dir
    / "01_relaxation_trajectory.png"
)

figure.savefig(
    output_path,
    dpi=300,
    facecolor="white",
    edgecolor="none",
    transparent=False,
    bbox_inches="tight",
    pad_inches=0.08,
)

plt.close(figure)


converged_count = sum(
    record["converged"]
    for record in recovery_records
)

print()
print(f"Created: {output_path}")
print(f"Material: {material_slug}")
print(f"Model: {model_id}")
print(f"Dtype: {mlff_dtype}")
print("Attack: FGSM")
print(
    "Recovery epsilon values: "
    + ", ".join(
        f"{value:g}"
        for value in epsilon_values
    )
)
print(
    f"Recovery trajectories: "
    f"{len(recovery_records)}"
)
PY
