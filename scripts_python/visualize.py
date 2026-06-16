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


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_materials(path):
    rows = read_csv_rows(path)
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
        "font.size": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 13,
    })


def load_initial_structures(materials, structures_dir):
    loaded = []
    missing = []

    for row in materials:
        path = structure_path(row, structures_dir)
        if not path.exists():
            missing.append(path)
            continue
        loaded.append((row, read(path), path))

    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"ERROR: missing initial CIF files:\n{lines}")

    return loaded


def resolve_run_dir(base_dir, row):
    run_id = str(row.get("run_id", "")).strip()
    if run_id:
        candidate = Path(base_dir) / run_id
        if candidate.exists():
            return candidate

    for key in ["final_relaxed_cif", "output_cif", "before_force_csv", "after_force_csv"]:
        value = str(row.get(key, "")).strip()
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = BASE_DIR / path
            if path.exists():
                return path.parent

    return None


def run_label(row, calculator):
    attack = str(row.get("attack_type") or row.get("attack_label") or "attack").lower()
    epsilon = str(row.get("epsilon", "na")).replace(".", "p")
    n_steps = str(row.get("n_steps", "na")).replace(".", "p")
    run_id = slug(row.get("run_id", "run"))
    return slug(f"{calculator}_{attack}_eps-{epsilon}_steps-{n_steps}_{run_id}")


def collect_final_runs(materials, mace_dir, uma_dir):
    material_by_key = {}
    for row in materials:
        keys = {
            slug(row.get("material_label", "")),
            slug(row.get("formula", "")),
            slug(row.get("mpid", "")),
        }
        for key in keys:
            if key:
                material_by_key[key] = row

    final_runs = []
    for calculator, base_dir in [("mace", mace_dir), ("uma", uma_dir)]:
        summary_path = Path(base_dir) / "summary.csv"
        for summary_row in read_csv_rows(summary_path):
            if str(summary_row.get("success", "true")).lower() in {"false", "0", "no"}:
                continue

            material_key = slug(
                summary_row.get("material_label")
                or summary_row.get("formula")
                or summary_row.get("mpid")
                or ""
            )
            material_row = material_by_key.get(material_key)
            if material_row is None:
                mpid = slug(summary_row.get("mpid", ""))
                material_row = material_by_key.get(mpid)
            if material_row is None:
                continue

            run_dir = resolve_run_dir(base_dir, summary_row)
            if run_dir is None:
                continue

            final_cif = run_dir / "final_relaxed.cif"
            if not final_cif.exists():
                value = str(summary_row.get("final_relaxed_cif", "")).strip()
                if value:
                    final_cif = Path(value)
                    if not final_cif.is_absolute():
                        final_cif = BASE_DIR / final_cif

            if final_cif.exists():
                final_runs.append({
                    "material": material_row,
                    "atoms": read(final_cif),
                    "path": final_cif,
                    "calculator": calculator,
                    "label": run_label(summary_row, calculator),
                })

    return final_runs


def make_element_legend(fig, all_symbols):
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=element_color(symbol),
            markeredgecolor="#222222",
            markeredgewidth=0.5,
            markersize=9,
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
        borderpad=0.7,
        labelspacing=0.55,
        handletextpad=0.6,
        columnspacing=1.0,
        ncol=2,
    )


def draw_atoms(ax, atoms, row, rotation, scale, radii_scale, panel=None):
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

    if panel:
        ax.text(
            -0.02,
            1.04,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    formula = row.get("formula", "").strip()
    mpid = row.get("mpid", "").strip()
    ax.set_title(f"{formula}\n{mpid}", pad=4, fontsize=12)


def save_single_structure(row, atoms, output_path, dpi, rotation, scale, radii_scale, title):
    apply_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    draw_atoms(ax, atoms, row, rotation, scale, radii_scale)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def save_grid(loaded, output_path, dpi, rotation, scale, radii_scale, title):
    apply_plot_style()

    fig, axes = plt.subplots(4, 5, figsize=(16, 11))
    axes = axes.ravel()
    all_symbols = set()

    for index, item in enumerate(loaded[:20]):
        row, atoms, _ = item
        all_symbols.update(atoms.get_chemical_symbols())
        draw_atoms(
            axes[index],
            atoms,
            row,
            rotation,
            scale,
            radii_scale,
            panel=string.ascii_uppercase[index],
        )

    for ax in axes[len(loaded[:20]):]:
        ax.set_axis_off()

    make_element_legend(fig, all_symbols)

    fig.suptitle(title, y=0.985, fontsize=18, fontweight="bold")
    fig.subplots_adjust(
        left=0.025,
        right=0.84,
        top=0.90,
        bottom=0.035,
        wspace=0.04,
        hspace=0.28,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot initial and final MLFF attack structures."
    )
    parser.add_argument("--materials", default="tests_materials.csv")
    parser.add_argument("--structures-dir", default="mp_structures")
    parser.add_argument("--mace-dir", default="outputs_mace")
    parser.add_argument("--uma-dir", default="outputs_uma")
    parser.add_argument("--output-dir", default="outputs_visuals")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--rotation", default="10x,-20y,0z")
    parser.add_argument("--scale", type=float, default=0.85)
    parser.add_argument("--radii-scale", type=float, default=0.7)
    args = parser.parse_args()

    materials = read_materials(BASE_DIR / args.materials)
    structures_dir = BASE_DIR / args.structures_dir
    output_dir = BASE_DIR / args.output_dir

    initial_loaded = load_initial_structures(materials, structures_dir)
    save_grid(
        loaded=initial_loaded,
        output_path=output_dir / "initial" / "initial_structures_5x4.png",
        dpi=args.dpi,
        rotation=args.rotation,
        scale=args.scale,
        radii_scale=args.radii_scale,
        title="Initial MP Structures Before Attack and Before Relaxation",
    )

    for row, atoms, _ in initial_loaded:
        name = f"{slug(row.get('material_label'))}_{slug(row.get('mpid'))}.png"
        save_single_structure(
            row=row,
            atoms=atoms,
            output_path=output_dir / "initial" / "materials" / name,
            dpi=args.dpi,
            rotation=args.rotation,
            scale=args.scale,
            radii_scale=args.radii_scale,
            title="Initial Structure",
        )

    final_runs = collect_final_runs(
        materials=materials,
        mace_dir=BASE_DIR / args.mace_dir,
        uma_dir=BASE_DIR / args.uma_dir,
    )

    first_final_by_material = {}
    for run in final_runs:
        key = slug(run["material"].get("mpid"))
        if key not in first_final_by_material:
            first_final_by_material[key] = run

    final_grid = []
    for row in materials:
        run = first_final_by_material.get(slug(row.get("mpid")))
        if run is not None:
            final_grid.append((run["material"], run["atoms"], run["path"]))

    if final_grid:
        save_grid(
            loaded=final_grid,
            output_path=output_dir / "final" / "final_structures_5x4.png",
            dpi=args.dpi,
            rotation=args.rotation,
            scale=args.scale,
            radii_scale=args.radii_scale,
            title="Final Structures After Attack and After Relaxation",
        )
    else:
        print("WARNING: no final_relaxed.cif files found for final 5x4 grid.")

    for run in final_runs:
        row = run["material"]
        material_dir = output_dir / "final" / "materials" / f"{slug(row.get('material_label'))}_{slug(row.get('mpid'))}"
        save_single_structure(
            row=row,
            atoms=run["atoms"],
            output_path=material_dir / f"{run['label']}.png",
            dpi=args.dpi,
            rotation=args.rotation,
            scale=args.scale,
            radii_scale=args.radii_scale,
            title="Final Structure After Attack and After Relaxation",
        )


if __name__ == "__main__":
    main()