#!/bin/bash
#SBATCH --account=rrg-j3goals
#SBATCH --time=00:03:00
#SBATCH --mem=4G

set -euo pipefail

module load gcc/12.3 python/3.11 arrow

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$HOME/project/.venv-mace/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found: $PYTHON"
    echo "Set PYTHON to an environment containing ASE, NumPy, and Matplotlib."
    exit 1
fi

if [ ! -f "20_licohpf.xyz" ]; then
    echo "ERROR: missing 20_licohpf.xyz"
    exit 1
fi

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

if [ -z "${MP_API_KEY:-}" ]; then
    echo "ERROR: MP_API_KEY is missing from .env"
    exit 1
fi

export MPLBACKEND=Agg

mkdir -p visualize_structures

"$PYTHON" - <<'PY'
from pathlib import Path
import os

from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ase.io import read
from ase.visualize.plot import plot_atoms


OUTPUT_DIR = Path("visualize_structures")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MP_IDENTIFIERS = [
    "mp-6922",
    "mp-48",
    "mp-5229",
]

TEXT_COLOR = "#202326"
SECONDARY_TEXT = "#555C63"
ACCENT = "#A62AA5"
ACCENT_LIGHT = "#F8EFF8"
CARD_EDGE = "#D8DCE0"

def render_structure(
    atoms,
    output_path,
    rotation,
    show_cell,
):
    figure, axis = plt.subplots(
        figsize=(3.2, 3.2),
        facecolor="white",
    )

    plot_atoms(
        atoms,
        axis,
        rotation=rotation,
        radii=0.72,
        show_unit_cell=show_cell,
    )

    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.margins(0.12)

    figure.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.99,
    )

    figure.savefig(
        output_path,
        dpi=350,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.04,
    )

    plt.close(figure)


# Read exactly the first three LiCOHPF structures.
licohpf_frames = read(
    "20_licohpf.xyz",
    index="0:3",
)

if len(licohpf_frames) != 3:
    raise SystemExit(
        "ERROR: 20_licohpf.xyz does not contain "
        "at least three structures"
    )

licohpf_records = []

for index, atoms in enumerate(
    licohpf_frames,
    start=1,
):
    identifier = f"licohpf_{index:03d}"

    output_path = (
        OUTPUT_DIR
        / f"{identifier}.png"
    )

    render_structure(
        atoms,
        output_path,
        rotation="18x,12y,8z",
        show_cell=1 if atoms.pbc.any() else 0,
    )

    licohpf_records.append({
        "identifier": identifier,
        "formula": atoms.get_chemical_formula(
            mode="hill",
        ),
        "image": output_path,
        "atoms": len(atoms),
    })


mp_records = []

with MPRester(
    os.environ["MP_API_KEY"]
) as mpr:
    documents = (
        mpr.materials.summary.search(
            material_ids=MP_IDENTIFIERS,
            fields=[
                "material_id",
                "formula_pretty",
                "structure",
            ],
        )
    )

documents_by_id = {
    str(document.material_id): document
    for document in documents
}

missing_identifiers = [
    identifier
    for identifier in MP_IDENTIFIERS
    if identifier not in documents_by_id
]

if missing_identifiers:
    raise SystemExit(
        "ERROR: Materials Project API did not return: "
        + ", ".join(missing_identifiers)
    )

adaptor = AseAtomsAdaptor()

for identifier in MP_IDENTIFIERS:
    document = documents_by_id[identifier]

    atoms = adaptor.get_atoms(
        document.structure
    )

    output_path = (
        OUTPUT_DIR
        / f"{identifier}.png"
    )

    render_structure(
        atoms,
        output_path,
        rotation="25x,0y,18z",
        show_cell=1,
    )

    mp_records.append({
        "identifier": identifier,
        "formula": (
            document.formula_pretty
            or atoms.get_chemical_formula(
                mode="hill",
            )
        ),
        "image": output_path,
        "atoms": len(atoms),
        "path": (
            "Materials Project API: "
            f"https://materialsproject.org/materials/{identifier}"
        ),
    })


# Create the complete 16:9 presentation slide.
figure = plt.figure(
    figsize=(16.0, 9.0),
    facecolor="white",
)

figure.text(
    0.055,
    0.925,
    "Benchmark Systems: 40 Structures Across Two Chemical Regimes",
    fontsize=30,
    color=TEXT_COLOR,
    ha="left",
    va="top",
)

figure.text(
    0.055,
    0.855,
    (
        "Complementary structure sets test whether recovery depends "
        "on chemistry and model training scope."
    ),
    fontsize=17,
    color=SECONDARY_TEXT,
    ha="left",
    va="top",
)

# Two dataset cards.
for x_position in (0.045, 0.515):
    card = plt.Rectangle(
        (x_position, 0.245),
        0.44,
        0.535,
        transform=figure.transFigure,
        facecolor="white",
        edgecolor=CARD_EDGE,
        linewidth=1.3,
    )

    figure.add_artist(card)


# Left: 2D Materials Project.
figure.text(
    0.070,
    0.745,
    "2D Materials Project",
    fontsize=23,
    weight="bold",
    color=TEXT_COLOR,
)

figure.text(
    0.445,
    0.745,
    "n = 20",
    fontsize=18,
    weight="bold",
    color=ACCENT,
    ha="right",
)

figure.text(
    0.070,
    0.690,
    (
        "Low-dimensional inorganic crystals\n"
        "Multiple chemistries and bonding environments\n"
        "Broader-chemistry benchmark"
    ),
    fontsize=14,
    color=SECONDARY_TEXT,
    linespacing=1.4,
    va="top",
)


# Right: LiCOHPF.
figure.text(
    0.540,
    0.745,
    "LiCOHPF",
    fontsize=23,
    weight="bold",
    color=TEXT_COLOR,
)

figure.text(
    0.915,
    0.745,
    "n = 20",
    fontsize=18,
    weight="bold",
    color=ACCENT,
    ha="right",
)

figure.text(
    0.540,
    0.690,
    (
        "Li–C–O configurations for SEI modelling\n"
        "Specialized chemical domain\n"
        "Training domain for custom MACE and MTP"
    ),
    fontsize=14,
    color=SECONDARY_TEXT,
    linespacing=1.4,
    va="top",
)


def add_structure_images(
    records,
    x_positions,
):
    image_y = 0.355
    image_width = 0.125
    image_height = 0.245

    for record, x_position in zip(
        records,
        x_positions,
    ):
        axis = figure.add_axes([
            x_position,
            image_y,
            image_width,
            image_height,
        ])

        axis.imshow(
            plt.imread(record["image"])
        )

        axis.set_axis_off()

        figure.text(
            x_position + image_width / 2,
            0.325,
            record["formula"],
            fontsize=12.5,
            weight="bold",
            color=TEXT_COLOR,
            ha="center",
        )

        figure.text(
            x_position + image_width / 2,
            0.295,
            record["identifier"],
            fontsize=11,
            color=SECONDARY_TEXT,
            ha="center",
        )


add_structure_images(
    mp_records,
    [
        0.065,
        0.207,
        0.349,
    ],
)

add_structure_images(
    licohpf_records,
    [
        0.535,
        0.677,
        0.819,
    ],
)


# Bottom conclusion.
highlight = plt.Rectangle(
    (0.055, 0.115),
    0.89,
    0.085,
    transform=figure.transFigure,
    facecolor=ACCENT_LIGHT,
    edgecolor="none",
)

figure.add_artist(highlight)

figure.text(
    0.075,
    0.158,
    "Why both?",
    fontsize=17,
    weight="bold",
    color=ACCENT,
    va="center",
)

figure.text(
    0.175,
    0.158,
    (
        "Distinguish broadly shared MLFF failure modes from behaviour "
        "associated with chemistry and training scope."
    ),
    fontsize=15,
    color=TEXT_COLOR,
    va="center",
)

figure.text(
    0.055,
    0.055,
    (
        "Sources: Materials Project and LiCOHPF database. "
        "Rendered from the experimental input structures."
    ),
    fontsize=10.5,
    color="#777D83",
)

plt.close(figure)


# Save a record of exactly what was used.
with (
    OUTPUT_DIR
    / "selected_structures.tsv"
).open(
    "w",
    encoding="utf-8",
) as handle:
    handle.write(
        "dataset\tidentifier\tformula\tatoms\tpath\n"
    )

    for record in mp_records:
        handle.write(
            "2D Materials Project\t"
            f"{record['identifier']}\t"
            f"{record['formula']}\t"
            f"{record['atoms']}\t"
            f"{record['path']}\n"
        )

    for record in licohpf_records:
        handle.write(
            "LiCOHPF\t"
            f"{record['identifier']}\t"
            f"{record['formula']}\t"
            f"{record['atoms']}\t"
            "20_licohpf.xyz\n"
        )


print("Created:")


for record in mp_records:
    print(
        f"  {record['identifier']}: "
        f"{record['path']}"
    )
PY
