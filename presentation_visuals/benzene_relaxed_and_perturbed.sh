#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=00:01:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$HOME/project/.venv-mace/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python environment not found: $PYTHON"
    exit 1
fi

mkdir -p visualize_workflow
export MPLBACKEND=Agg

rm -f \
    visualize_workflow/00_initial_structure.png \
    visualize_workflow/01_first_relaxation.png \
    visualize_workflow/02_perturbation.png \
    visualize_workflow/03_second_relaxation.png

"$PYTHON" - <<'PY'
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms


output_dir = Path("visualize_workflow")
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

# Publication-style colours.
CARBON_FACE = "#4B4F54"
CARBON_EDGE = "#202326"

HYDROGEN_FACE = "#F8F8F8"
HYDROGEN_EDGE = "#6C737A"

BOND_COLOR = "#737A82"
GHOST_COLOR = "#B8BDC3"

RELAX_COLOR = "#2166AC"
RELAX_LIGHT = "#82B9DA"

PERTURB_COLOR = "#D95F02"

# Construct the connected benzene-like structure.
angles = np.linspace(
    0.0,
    2.0 * np.pi,
    6,
    endpoint=False,
)

carbon_radius = 1.40
hydrogen_radius = 2.48

carbon_positions = np.column_stack([
    carbon_radius * np.cos(angles),
    carbon_radius * np.sin(angles),
    np.zeros(6),
])

hydrogen_positions = np.column_stack([
    hydrogen_radius * np.cos(angles),
    hydrogen_radius * np.sin(angles),
    np.zeros(6),
])

regular_positions = np.vstack([
    carbon_positions,
    hydrogen_positions,
])

first_relaxed = Atoms(
    symbols=["C"] * 6 + ["H"] * 6,
    positions=regular_positions,
)

# Initial distorted configuration.
initial = first_relaxed.copy()

initial_offsets = {
    0: np.array([-0.74,  0.34, 0.0]),
    2: np.array([ 0.42,  0.70, 0.0]),
    4: np.array([ 0.46, -0.65, 0.0]),
}

for atom_index, offset in initial_offsets.items():
    initial.positions[atom_index] += offset
    initial.positions[atom_index + 6] += offset

# Representative signs of an MLFF loss gradient.
#
# This creates an FGSM-style displacement:
#
#     R_adv = R + epsilon * sign(gradient)
#
# Directions are deliberately heterogeneous rather than radial.
carbon_gradient = np.array([
    [ 0.84,  0.31],
    [-0.42,  0.91],
    [ 0.67, -0.73],
    [-0.88, -0.26],
    [-0.35,  0.79],
    [ 0.76, -0.58],
])

hydrogen_gradient = np.array([
    [-0.62,  0.77],
    [ 0.48,  0.86],
    [-0.81, -0.39],
    [ 0.93,  0.24],
    [-0.56, -0.74],
    [ 0.38, -0.92],
])

epsilon = 0.52

carbon_attack_offsets = (
    epsilon * np.sign(carbon_gradient)
)

hydrogen_attack_offsets = (
    epsilon * np.sign(hydrogen_gradient)
)

perturbed = first_relaxed.copy()

for atom_index in range(6):
    perturbed.positions[
        atom_index,
        :2,
    ] += carbon_attack_offsets[atom_index]

    perturbed.positions[
        atom_index + 6,
        :2,
    ] += hydrogen_attack_offsets[atom_index]

# Second relaxation returns close to the original relaxed state.
second_relaxed = first_relaxed.copy()

residual_fraction = 0.08

for atom_index in range(6):
    second_relaxed.positions[
        atom_index,
        :2,
    ] += (
        residual_fraction
        * carbon_attack_offsets[atom_index]
    )

    second_relaxed.positions[
        atom_index + 6,
        :2,
    ] += (
        residual_fraction
        * hydrogen_attack_offsets[atom_index]
    )

# Six carbon-carbon and six carbon-hydrogen bonds.
bond_pairs = []

for atom_index in range(6):
    bond_pairs.append(
        (
            atom_index,
            (atom_index + 1) % 6,
        )
    )

    bond_pairs.append(
        (
            atom_index,
            atom_index + 6,
        )
    )

carbon_indices = np.arange(6)
hydrogen_indices = np.arange(6, 12)

all_coordinates = np.vstack([
    initial.positions[:, :2],
    first_relaxed.positions[:, :2],
    perturbed.positions[:, :2],
    second_relaxed.positions[:, :2],
])

minimum = all_coordinates.min(axis=0)
maximum = all_coordinates.max(axis=0)

padding = 0.68

x_limits = (
    minimum[0] - padding,
    maximum[0] + padding,
)

y_limits = (
    minimum[1] - padding,
    maximum[1] + padding,
)


def interpolate(
    before,
    after,
    fraction,
):
    result = before.copy()

    result.positions = (
        (1.0 - fraction) * before.positions
        + fraction * after.positions
    )

    return result


def draw_bonds(
    axis,
    atoms,
    color=BOND_COLOR,
    linewidth=2.7,
    alpha=1.0,
    linestyle="-",
    zorder=1,
):
    coordinates = atoms.positions[:, :2]

    for atom_a, atom_b in bond_pairs:
        axis.plot(
            [
                coordinates[atom_a, 0],
                coordinates[atom_b, 0],
            ],
            [
                coordinates[atom_a, 1],
                coordinates[atom_b, 1],
            ],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=zorder,
        )


def draw_atoms(
    axis,
    atoms,
    ghost=False,
    color=GHOST_COLOR,
    alpha=1.0,
    size_scale=1.0,
    zorder=4,
):
    coordinates = atoms.positions[:, :2]

    if ghost:
        axis.scatter(
            coordinates[carbon_indices, 0],
            coordinates[carbon_indices, 1],
            s=300 * size_scale,
            facecolors="none",
            edgecolors=color,
            linewidths=1.3,
            alpha=alpha,
            zorder=zorder,
        )

        axis.scatter(
            coordinates[hydrogen_indices, 0],
            coordinates[hydrogen_indices, 1],
            s=105 * size_scale,
            facecolors="none",
            edgecolors=color,
            linewidths=1.0,
            alpha=alpha,
            zorder=zorder,
        )

        return

    axis.scatter(
        coordinates[carbon_indices, 0],
        coordinates[carbon_indices, 1],
        s=300,
        facecolors=CARBON_FACE,
        edgecolors=CARBON_EDGE,
        linewidths=1.3,
        zorder=zorder,
    )

    axis.scatter(
        coordinates[hydrogen_indices, 0],
        coordinates[hydrogen_indices, 1],
        s=105,
        facecolors=HYDROGEN_FACE,
        edgecolors=HYDROGEN_EDGE,
        linewidths=1.1,
        zorder=zorder,
    )


def draw_structure(
    axis,
    atoms,
):
    draw_bonds(
        axis,
        atoms,
        zorder=4,
    )

    draw_atoms(
        axis,
        atoms,
        ghost=False,
        zorder=5,
    )


def draw_ghost_structure(
    axis,
    atoms,
    color=GHOST_COLOR,
    alpha=0.45,
    zorder=1,
):
    draw_bonds(
        axis,
        atoms,
        color=color,
        linewidth=1.3,
        alpha=alpha,
        linestyle="--",
        zorder=zorder,
    )

    draw_atoms(
        axis,
        atoms,
        ghost=True,
        color=color,
        alpha=alpha,
        size_scale=0.92,
        zorder=zorder + 0.1,
    )


def draw_relaxation_echoes(
    axis,
    before,
    after,
):
    # Original configuration.
    draw_ghost_structure(
        axis,
        before,
        color=GHOST_COLOR,
        alpha=0.40,
        zorder=1,
    )

    # Fading geometry-optimization iterations.
    echo_specs = [
        (0.25, 0.14),
        (0.50, 0.22),
        (0.75, 0.34),
    ]

    for echo_index, (
        fraction,
        alpha,
    ) in enumerate(echo_specs):
        intermediate = interpolate(
            before,
            after,
            fraction,
        )

        zorder = 2 + 0.2 * echo_index

        draw_bonds(
            axis,
            intermediate,
            color=RELAX_LIGHT,
            linewidth=1.4,
            alpha=alpha,
            linestyle="-",
            zorder=zorder,
        )

        draw_atoms(
            axis,
            intermediate,
            ghost=True,
            color=RELAX_COLOR,
            alpha=alpha,
            size_scale=0.88,
            zorder=zorder + 0.1,
        )

    # Final optimized structure.
    draw_structure(
        axis,
        after,
    )


def draw_perturbation_vectors(
    axis,
    before,
    after,
):
    before_xy = before.positions[:, :2]
    after_xy = after.positions[:, :2]

    starts = []
    vectors = []

    # Show vectors for all six core atoms.
    for atom_index in carbon_indices:
        start = before_xy[atom_index]
        end = after_xy[atom_index]

        displacement = end - start
        length = float(
            np.linalg.norm(displacement)
        )

        if length <= 1.0e-12:
            continue

        direction = displacement / length

        # Keep arrow tails and heads outside atomic circles.
        visible_start = (
            start + 0.16 * direction
        )

        visible_end = (
            end - 0.22 * direction
        )

        vector = (
            visible_end - visible_start
        )

        starts.append(visible_start)
        vectors.append(vector)

    starts = np.asarray(starts)
    vectors = np.asarray(vectors)

    # White underlay for contrast.
    axis.quiver(
        starts[:, 0],
        starts[:, 1],
        vectors[:, 0],
        vectors[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.018,
        headwidth=5.0,
        headlength=6.2,
        headaxislength=5.4,
        pivot="tail",
        color="white",
        zorder=7,
    )

    # FGSM displacement vectors.
    axis.quiver(
        starts[:, 0],
        starts[:, 1],
        vectors[:, 0],
        vectors[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.009,
        headwidth=4.5,
        headlength=5.8,
        headaxislength=5.0,
        pivot="tail",
        color=PERTURB_COLOR,
        zorder=8,
    )


def prepare_figure():
    figure, axis = plt.subplots(
        figsize=(5.0, 5.0),
        facecolor="white",
    )

    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal")
    axis.set_axis_off()

    figure.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.015,
        top=0.985,
    )

    return figure, axis


def save_figure(
    figure,
    filename,
):
    figure.savefig(
        output_dir / filename,
        dpi=400,
        facecolor="white",
        edgecolor="none",
    )

    plt.close(figure)


# 00: Initial distorted configuration.
figure, axis = prepare_figure()

draw_structure(
    axis,
    initial,
)

save_figure(
    figure,
    "00_initial_structure.png",
)

# 01: First relaxation with fading optimization echoes.
figure, axis = prepare_figure()

draw_relaxation_echoes(
    axis,
    initial,
    first_relaxed,
)

save_figure(
    figure,
    "01_first_relaxation.png",
)

# 02: FGSM-style adversarial perturbation.
figure, axis = prepare_figure()

draw_ghost_structure(
    axis,
    first_relaxed,
)

draw_structure(
    axis,
    perturbed,
)

draw_perturbation_vectors(
    axis,
    first_relaxed,
    perturbed,
)

save_figure(
    figure,
    "02_perturbation.png",
)

# 03: Second relaxation using the same echo treatment.
figure, axis = prepare_figure()

draw_relaxation_echoes(
    axis,
    perturbed,
    second_relaxed,
)

save_figure(
    figure,
    "03_second_relaxation.png",
)

print("Created:")
print("  visualize_workflow/00_initial_structure.png")
print("  visualize_workflow/01_first_relaxation.png")
print("  visualize_workflow/02_perturbation.png")
print("  visualize_workflow/03_second_relaxation.png")
PY
