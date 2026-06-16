#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import re
import string

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from ase.io import read
from ase.visualize.plot import plot_atoms


BASE_DIR = Path(__file__).resolve().parent.parent


def slug(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def read_materials(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 20:
        raise SystemExit(f"ERROR: expected 20 materials, found {len(rows)} in {path}")

    return rows


def structure_path(row, structures_dir):
    mpid = row["mpid"].strip()
    label = slug(row["material_label"])
    return Path(structures_dir) / f"{mpid}_{label}.cif"


def element_color(symbol):
    return jmol_colors[atomic_numbers[symbol]]


def apply_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 13,
        "legend.title_fontsize": 14,
    })


def load_structures(materials, structures_dir):
    loaded = []
    missing = []

    for row in materials:
        path = structure_path(row, structures_dir)
        if not path.exists():
            missing.append(path)
            continue

        atoms = read(path)
        loaded.append((row, atoms, path))

    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "ERROR: missing MP structure CIF files:\n"
            f"{lines}\n\n"
            "Download them first with:\n"
            "  python scripts_python/run_material_mpids.py --download-only"
        )

    return loaded


def make_element_legend(fig, all_symbols):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=element_color(symbol),
            markeredgecolor="#222222",
            markeredgewidth=0.5,
            markersize=10,
            label=symbol,
        )
        for symbol in sorted(all_symbols, key=lambda symbol: atomic_numbers[symbol])
    ]

    fig.legend(
        handles=handles,
        title="Elements",
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        frameon=True,
        borderpad=0.8,
        labelspacing=0.6,
        handletextpad=0.7,
        columnspacing=1.0,
        ncol=2,
    )


def draw_structure(ax, row, atoms, rotation, scale, radii_scale):
    symbols = atoms.get_chemical_symbols()
    colors = [element_color(symbol) for symbol in symbols]

    plot_atoms(
        atoms,
        ax=ax,
        rotation=rotation,
        show_unit_cell=2,
        colors=colors,
        radii=radii_scale,
        scale=scale,
    )

    formula = row.get("formula", "").strip()
    mpid = row.get("mpid", "").strip()
    ax.set_title(f"{formula}\n{mpid}", pad=4, fontsize=13)


def plot_5x4_structures(loaded, output_path, dpi, rotation, scale, radii_scale):
    apply_plot_style()

    fig, axes = plt.subplots(4, 5, figsize=(16, 11))
    axes = axes.ravel()

    all_symbols = set()

    for index, (row, atoms, _) in enumerate(loaded):
        ax = axes[index]
        all_symbols.update(atoms.get_chemical_symbols())

        draw_structure(
            ax=ax,
            row=row,
            atoms=atoms,
            rotation=rotation,
            scale=scale,
            radii_scale=radii_scale,
        )

        panel = string.ascii_uppercase[index]
        ax.text(
            0.0,
            1.13,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=18,
            fontweight="bold",
        )

    for ax in axes[len(loaded):]:
        ax.set_axis_off()

    make_element_legend(fig, all_symbols)

    fig.suptitle(
        "Initial MP Structures Before Attack and Before Relaxation",
        y=0.985,
        fontsize=22,
        fontweight="bold",
    )

    fig.subplots_adjust(
        left=0.025,
        right=0.84,
        top=0.88,
        bottom=0.035,
        wspace=0.04,
        hspace=0.42,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_path}")


def plot_single_structure(row, atoms, output_path, dpi, rotation, scale, radii_scale):
    apply_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))

    draw_structure(
        ax=ax,
        row=row,
        atoms=atoms,
        rotation=rotation,
        scale=scale,
        radii_scale=radii_scale,
    )

    fig.suptitle(
        "Before Attack and Before Relaxation",
        y=0.98,
        fontsize=16,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot only the 20 initial MP structures before attack and before relaxation."
    )
    parser.add_argument("--materials", default="tests_materials.csv")
    parser.add_argument("--structures-dir", default="mp_structures")
    parser.add_argument("--output-dir", default="outputs_visuals")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--rotation", default="10x,-20y,0z")
    parser.add_argument("--scale", type=float, default=0.85)
    parser.add_argument("--radii-scale", type=float, default=0.7)
    args = parser.parse_args()

    materials_path = BASE_DIR / args.materials
    structures_dir = BASE_DIR / args.structures_dir
    output_dir = BASE_DIR / args.output_dir

    materials = read_materials(materials_path)
    loaded = load_structures(materials, structures_dir)

    plot_5x4_structures(
        loaded=loaded,
        output_path=output_dir / "initial_structures_5x4.png",
        dpi=args.dpi,
        rotation=args.rotation,
        scale=args.scale,
        radii_scale=args.radii_scale,
    )

    for row, atoms, _ in loaded:
        filename = f"{slug(row.get('material_label'))}_{slug(row.get('mpid'))}.png"
        plot_single_structure(
            row=row,
            atoms=atoms,
            output_path=output_dir / "materials" / filename,
            dpi=args.dpi,
            rotation=args.rotation,
            scale=args.scale,
            radii_scale=args.radii_scale,
        )


if __name__ == "__main__":
    main()