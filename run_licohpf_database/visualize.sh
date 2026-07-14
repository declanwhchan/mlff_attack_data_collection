#!/usr/bin/env bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=visualize-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "$REPO_ROOT"

if [[ -f "$HOME/project/.venv-mace/bin/activate" ]]; then
    set +u
    source "$HOME/project/.venv-mace/bin/activate"
    set -u
fi

export MPLBACKEND=Agg

python - <<'PY'
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ase.io import read

root = Path.cwd()
input_file = root / "20_licohpf.xyz"
output_root = root / "outputs_visuals" / "licohpf"
individual_dir = output_root / "individual"

output_root.mkdir(parents=True, exist_ok=True)
individual_dir.mkdir(parents=True, exist_ok=True)

structures = read(input_file, index=":")

if len(structures) != 20:
    raise RuntimeError(
        f"Expected 20 structures in {input_file}, found {len(structures)}"
    )

element_colors = {
    "Li": "#5DADE2",
    "C": "#2C3E50",
    "O": "#E74C3C",
}

fallback_colors = [
    "#9B59B6",
    "#16A085",
    "#F39C12",
    "#E67E22",
    "#1ABC9C",
]

all_elements = sorted(
    {
        symbol
        for atoms in structures
        for symbol in atoms.get_chemical_symbols()
    }
)

for index, symbol in enumerate(all_elements):
    if symbol not in element_colors:
        element_colors[symbol] = fallback_colors[index % len(fallback_colors)]


def draw_structure(ax, atoms, title):
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    for symbol in all_elements:
        selected = [
            i for i, current_symbol in enumerate(symbols)
            if current_symbol == symbol
        ]

        if not selected:
            continue

        xyz = positions[selected]

        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=90 if symbol == "Li" else 150,
            c=element_colors[symbol],
            edgecolors="white",
            linewidths=0.7,
            alpha=0.95,
            depthshade=True,
            label=symbol,
        )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("x (Å)", fontsize=10)
    ax.set_ylabel("y (Å)", fontsize=10)
    ax.set_zlabel("z (Å)", fontsize=10)

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=38)

    ax.grid(True, alpha=0.2)
    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.04)

    limits = positions.min(axis=0), positions.max(axis=0)
    center = (limits[0] + limits[1]) / 2
    span = max((limits[1] - limits[0]).max() * 0.6, 1.0)

    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)


legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=symbol,
        markerfacecolor=element_colors[symbol],
        markeredgecolor="white",
        markersize=9,
    )
    for symbol in all_elements
]

fig = plt.figure(figsize=(24, 18), facecolor="white")
fig.suptitle(
    "LiCOHPF Structures",
    fontsize=30,
    fontweight="bold",
    y=0.985,
)

fig.text(
    0.5,
    0.955,
    "20 structures from 20_licohpf.xyz",
    ha="center",
    fontsize=16,
    color="#555555",
)

rows = 4
columns = 5

for index, atoms in enumerate(structures, start=1):
    ax = fig.add_subplot(rows, columns, index, projection="3d")
    draw_structure(ax, atoms, f"LiCOHPF {index:02d}")

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(legend_handles),
    frameon=False,
    fontsize=15,
    bbox_to_anchor=(0.5, 0.015),
)

fig.subplots_adjust(
    left=0.02,
    right=0.98,
    top=0.92,
    bottom=0.06,
    wspace=0.08,
    hspace=0.12,
)

fig.savefig(
    output_root / "all_20_licohpf.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)

for index, atoms in enumerate(structures, start=1):
    fig = plt.figure(figsize=(12, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    draw_structure(ax, atoms, f"LiCOHPF Structure {index:02d}")

    fig.text(
        0.5,
        0.035,
        "LiCOHPF structure from 20_licohpf.xyz",
        ha="center",
        fontsize=13,
        color="#555555",
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, 0.97),
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.94,
        top=0.88,
        bottom=0.08,
    )

    fig.savefig(
        individual_dir / f"licohpf_{index:02d}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

print(f"Created: {output_root / 'all_20_licohpf.png'}")
print(f"Created {len(structures)} individual slideshow PNGs")
print(f"Output directory: {output_root}")
PY
