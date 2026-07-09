#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import math
import re
import string

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ase.data import atomic_numbers, covalent_radii
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
            "  python pipeline/run_material_mpids.py --download-only"
        )

    return loaded


def optional_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_calc_context(base_dir):
    rows_by_material = {}

    for calculator, summary_path in [
        ("mace", base_dir / "outputs_mace" / "summary.csv"),
        ("uma", base_dir / "outputs_uma" / "summary.csv"),
        ("chgnet", base_dir / "outputs_chgnet" / "summary.csv"),
    ]:
        if not summary_path.exists():
            continue

        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                material_slug = row.get("material_slug")
                if not material_slug:
                    continue

                item = dict(row)
                item["calculator"] = calculator
                rows_by_material.setdefault(material_slug, []).append(item)

    return rows_by_material


def summarize_calc_context(material_slug, calc_rows_by_material):
    rows = calc_rows_by_material.get(material_slug, [])
    if not rows:
        return "not_available"

    statuses = {}
    max_displacements = []
    mean_displacements = []
    final_energies = []

    for row in rows:
        status = str(row.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1

        max_displacement = optional_float(row.get("max_displacement"))
        mean_displacement = optional_float(row.get("mean_displacement"))
        final_energy = optional_float(row.get("final_energy"))

        if max_displacement is not None:
            max_displacements.append(max_displacement)
        if mean_displacement is not None:
            mean_displacements.append(mean_displacement)
        if final_energy is not None:
            final_energies.append(final_energy)

    parts = [f"{len(rows)} calc runs"]
    parts.extend(f"{status}={count}" for status, count in sorted(statuses.items()))

    if max_displacements:
        parts.append(f"max displacement={max(max_displacements):.3f} A")
    if mean_displacements:
        parts.append(f"mean displacement={np.mean(mean_displacements):.3f} A")
    if final_energies:
        parts.append(
            f"final energy range={min(final_energies):.3f} to {max(final_energies):.3f} eV"
        )

    return "; ".join(parts)


def symmetry_summary(atoms):
    try:
        import spglib
    except ImportError:
        return "not_available", "spglib not installed"

    try:
        cell = (
            atoms.cell.array,
            atoms.get_scaled_positions(wrap=True),
            atoms.get_atomic_numbers(),
        )
        dataset = spglib.get_symmetry_dataset(cell, symprec=1e-2)
    except Exception as exc:
        return "failed", f"symmetry check failed: {exc}"

    if dataset is None:
        return "undetected", "no symmetry dataset found"

    if isinstance(dataset, dict):
        number = dataset.get("number")
        international = dataset.get("international")
    else:
        number = getattr(dataset, "number", None)
        international = getattr(dataset, "international", None)

    if number is None and international is None:
        return "unknown", "symmetry dataset missing spacegroup fields"

    return "available", f"{international} ({number})"


def closest_contact_summary(atoms):
    n_atoms = len(atoms)
    if n_atoms < 2:
        return {
            "closest_pair": "",
            "closest_distance_a": "",
            "closest_covalent_ratio": "",
            "overlap_flag": False,
            "close_contact_flag": False,
        }

    distances = atoms.get_all_distances(mic=True)
    symbols = atoms.get_chemical_symbols()

    best = None

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = float(distances[i, j])
            if distance <= 0:
                continue

            zi = atomic_numbers[symbols[i]]
            zj = atomic_numbers[symbols[j]]
            covalent_sum = float(covalent_radii[zi] + covalent_radii[zj])

            if covalent_sum <= 0 or not np.isfinite(covalent_sum):
                ratio = math.inf
            else:
                ratio = distance / covalent_sum

            if best is None or ratio < best["ratio"]:
                best = {
                    "pair": f"{symbols[i]}-{symbols[j]}",
                    "distance": distance,
                    "ratio": ratio,
                }

    if best is None:
        return {
            "closest_pair": "",
            "closest_distance_a": "",
            "closest_covalent_ratio": "",
            "overlap_flag": False,
            "close_contact_flag": False,
        }

    return {
        "closest_pair": best["pair"],
        "closest_distance_a": best["distance"],
        "closest_covalent_ratio": best["ratio"],
        "overlap_flag": best["ratio"] < 0.65,
        "close_contact_flag": best["ratio"] < 0.85,
    }


def diagnose_structure(row, atoms, calc_rows_by_material, boundary_tol=0.03):
    material_label = row.get("material_label", "").strip()
    material_slug = slug(material_label)
    formula = row.get("formula", "").strip()
    mpid = row.get("mpid", "").strip()
    category = row.get("category", "").strip()

    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    min_length = float(np.min(lengths))
    max_length = float(np.max(lengths))
    anisotropy = max_length / min_length if min_length > 0 else math.inf

    scaled = atoms.get_scaled_positions(wrap=True)
    near_boundary = np.any(
        (scaled < boundary_tol) | (scaled > 1.0 - boundary_tol),
        axis=1,
    )
    boundary_atom_count = int(np.count_nonzero(near_boundary))

    expected_atoms = optional_float(row.get("prompt_atoms_cell"))
    expected_atoms = int(expected_atoms) if expected_atoms is not None else None
    n_atoms = len(atoms)
    atom_count_match = expected_atoms is None or n_atoms == expected_atoms

    contact = closest_contact_summary(atoms)
    symmetry_status, spacegroup = symmetry_summary(atoms)
    calc_context = summarize_calc_context(material_slug, calc_rows_by_material)

    issues = []

    if contact["overlap_flag"]:
        issues.append(
            f"likely atom overlap: closest {contact['closest_pair']} contact is "
            f"{contact['closest_covalent_ratio']:.2f}x covalent-radius sum"
        )
    elif contact["close_contact_flag"]:
        issues.append(
            f"close contact: closest {contact['closest_pair']} contact is "
            f"{contact['closest_covalent_ratio']:.2f}x covalent-radius sum"
        )
    else:
        issues.append("no severe atom-overlap warning from nearest-neighbor geometry")

    if anisotropy > 3.0:
        issues.append(
            f"highly anisotropic cell ({anisotropy:.2f}x); 2D projection may make layers/channels look misleading"
        )

    skewed_angles = [
        angle for angle in angles
        if abs(float(angle) - 90.0) > 12.0 and abs(float(angle) - 120.0) > 12.0
    ]
    if skewed_angles:
        issues.append("skewed cell angles may make the rendered unit cell look distorted")

    if boundary_atom_count:
        issues.append(
            f"{boundary_atom_count} atoms lie near periodic cell boundaries; apparent edge overlaps may be periodic images"
        )

    if not atom_count_match:
        issues.append(
            f"atom count differs from prompt expectation: rendered {n_atoms}, expected {expected_atoms}"
        )

    if symmetry_status == "not_available":
        issues.append("symmetry not checked because spglib is not installed")
    elif symmetry_status in {"failed", "undetected", "unknown"}:
        issues.append(spacegroup)
    else:
        issues.append(f"symmetry detected as {spacegroup}")

    if calc_context == "not_available":
        issues.append("calculation-output context not available")
    else:
        issues.append(f"calculation context: {calc_context}")

    severe = contact["overlap_flag"] or not atom_count_match
    caution = contact["close_contact_flag"] or anisotropy > 3.0 or boundary_atom_count > 0

    if severe:
        verdict = "inspect carefully; visual may be inaccurate or structurally problematic"
    elif caution:
        verdict = "mostly usable, but projection/periodic-boundary effects may be misleading"
    else:
        verdict = "visual is likely a reasonable representation of the initial structure"

    summary = f"{verdict}. Closest contact: {contact['closest_pair']} at {contact['closest_distance_a']:.3f} A."

    return {
        "material_label": material_label,
        "formula": formula,
        "mpid": mpid,
        "category": category,
        "n_atoms": n_atoms,
        "expected_atoms": expected_atoms if expected_atoms is not None else "",
        "atom_count_match": atom_count_match,
        "cell_a": float(lengths[0]),
        "cell_b": float(lengths[1]),
        "cell_c": float(lengths[2]),
        "cell_alpha": float(angles[0]),
        "cell_beta": float(angles[1]),
        "cell_gamma": float(angles[2]),
        "cell_anisotropy": anisotropy,
        "closest_pair": contact["closest_pair"],
        "closest_distance_a": contact["closest_distance_a"],
        "closest_covalent_ratio": contact["closest_covalent_ratio"],
        "overlap_flag": contact["overlap_flag"],
        "close_contact_flag": contact["close_contact_flag"],
        "boundary_atom_count": boundary_atom_count,
        "symmetry_status": symmetry_status,
        "spacegroup": spacegroup,
        "calc_context": calc_context,
        "issues": "; ".join(issues),
        "summary": summary,
    }


def write_structure_diagnostics(loaded, output_dir):
    calc_rows_by_material = load_calc_context(BASE_DIR)
    diagnostics = [
        diagnose_structure(row, atoms, calc_rows_by_material)
        for row, atoms, _ in loaded
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "structure_diagnostics.csv"
    fieldnames = [
        "material_label",
        "formula",
        "mpid",
        "category",
        "n_atoms",
        "expected_atoms",
        "atom_count_match",
        "cell_a",
        "cell_b",
        "cell_c",
        "cell_alpha",
        "cell_beta",
        "cell_gamma",
        "cell_anisotropy",
        "closest_pair",
        "closest_distance_a",
        "closest_covalent_ratio",
        "overlap_flag",
        "close_contact_flag",
        "boundary_atom_count",
        "symmetry_status",
        "spacegroup",
        "calc_context",
        "issues",
        "summary",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)

    md_path = output_dir / "structure_diagnostics.md"
    with md_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# Initial Structure Visual Diagnostics\n\n")
        handle.write(
            "These diagnostics are automated checks to help judge whether each rendered visual "
            "is likely representative or potentially misleading. They do not replace crystallographic review.\n\n"
        )

        for item in diagnostics:
            handle.write(f"## {item['material_label']} ({item['formula']}, {item['mpid']})\n\n")
            handle.write(f"**Verdict:** {item['summary']}\n\n")
            handle.write(
                f"- Category: {item['category']}\n"
                f"- Atom count: {item['n_atoms']} rendered; expected {item['expected_atoms']}\n"
                f"- Closest contact: {item['closest_pair']} at {item['closest_distance_a']:.3f} A "
                f"({item['closest_covalent_ratio']:.2f}x covalent-radius sum)\n"
                f"- Cell lengths: a={item['cell_a']:.3f} A, b={item['cell_b']:.3f} A, c={item['cell_c']:.3f} A\n"
                f"- Cell angles: alpha={item['cell_alpha']:.2f}, beta={item['cell_beta']:.2f}, gamma={item['cell_gamma']:.2f}\n"
                f"- Symmetry: {item['spacegroup']}\n"
                f"- Calc context: {item['calc_context']}\n"
            )

            handle.write("- Notes:\n")
            for issue in item["issues"].split("; "):
                handle.write(f"  - {issue}\n")
            handle.write("\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


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

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.5))

    draw_structure(
        ax=ax,
        row=row,
        atoms=atoms,
        rotation=rotation,
        scale=scale,
        radii_scale=radii_scale,
    )

    make_element_legend(fig, set(atoms.get_chemical_symbols()))

    fig.subplots_adjust(
        left=0.03,
        right=0.72,
        top=0.88,
        bottom=0.04,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot only the 20 initial MP structures before attack and before relaxation."
    )
    parser.add_argument("--materials", default="datasets/tests_materials.csv")
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

    write_structure_diagnostics(loaded, output_dir)

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