#!/usr/bin/env python3

from pathlib import Path
import argparse
import gc
import threading
import time
import shutil

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "mace": "#0072B2",
    "uma": "#D55E00",
}

LINESTYLES = {
    "FGSM": "-",
    "I-FGSM": "--",
    "PGD": ":",
}

ATTACKS = ["FGSM", "I-FGSM", "PGD"]

METADATA_COLUMNS = [
    "base_material_slug",
    "base_material_label",
    "base_input_path",
    "supercell_repeat_x",
    "supercell_repeat_y",
    "supercell_repeat_z",
    "supercell_repeat_tuple",
    "unit_cell_atoms",
    "supercell_atoms",
]


def as_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    return value if np.isfinite(value) else np.nan


def as_int(value):
    value = as_float(value)
    return int(value) if np.isfinite(value) else None


def current_rss_mib():
    """Return this process's current resident memory on Linux."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss_kib = float(line.split()[1])
                    return rss_kib / 1024.0
    except (OSError, ValueError, IndexError):
        pass

    return np.nan


class PeakRSSSampler:
    """Sample peak resident CPU RAM during one test row."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak_mib = np.nan
        self._stop_event = threading.Event()
        self._thread = None

    def _sample_once(self):
        rss_mib = current_rss_mib()

        if np.isfinite(rss_mib):
            if not np.isfinite(self.peak_mib):
                self.peak_mib = rss_mib
            else:
                self.peak_mib = max(self.peak_mib, rss_mib)

    def _sample_loop(self):
        while not self._stop_event.wait(self.interval):
            self._sample_once()

    def start(self):
        gc.collect()
        self._sample_once()

        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._sample_once()
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=1.0)

        self._sample_once()
        return self.peak_mib


def metadata_from_row(row):
    return {
        column: row.get(column, "")
        for column in METADATA_COLUMNS
    }


def attack_label(row):
    run_id = str(row.get("run_id", "")).lower()
    attack_type = str(row.get("attack_type", "")).lower()
    n_steps = as_int(row.get("n_steps")) or 1

    if "_ifgsm_" in run_id:
        return "I-FGSM"
    if attack_type == "fgsm" and n_steps > 1:
        return "I-FGSM"
    if attack_type == "fgsm":
        return "FGSM"
    if attack_type == "pgd":
        return "PGD"

    return attack_type.upper()


def write_summary(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_command(args):
    # Lazy import keeps plot mode independent of mlff_attack.
    import run_tests

    experiments = pd.read_csv(args.tests, keep_default_na=False)
    summaries = []
    active_environment = run_tests.active_environment()

    print(f"Reading supercell tests from {args.tests}")
    print(f"Active environment: {active_environment}")
    print("Supercell runtime device: CPU")

    for index, original_row in experiments.iterrows():
        row = original_row.copy()
        row["device"] = "cpu"

        run_id = str(row["run_id"])
        calculator = run_tests.infer_calculator(row["model_path"])
        atom_count = as_int(row.get("supercell_atoms"))
        metadata = metadata_from_row(row)

        print(
            f"Running row {index + 1}/{len(experiments)}: {run_id}"
        )

        if (
            active_environment in {"mace", "uma"}
            and calculator != active_environment
        ):
            summary = {
                "run_id": run_id,
                "status": "skipped",
                "reason": (
                    f"Active environment is {active_environment}, "
                    f"row calculator is {calculator}"
                ),
            }
            summary.update(metadata)
            summaries.append(summary)
            write_summary(summaries, args.summary_file)
            continue

        sampler = PeakRSSSampler()
        sampler.start()
        start_time = time.perf_counter()

        try:
            summary = run_tests.run_one(row)
        except Exception as error:
            summary = {
                "run_id": run_id,
                "status": "failed",
                "error": str(error),
                "calculator": calculator,
                "device": "cpu",
            }
            print(f"Failed {run_id}: {error}")

        runtime_seconds = time.perf_counter() - start_time
        peak_rss_mib = sampler.stop()

        summary.update(metadata)
        summary["runtime_seconds"] = runtime_seconds
        summary["runtime_seconds_per_atom"] = (
            runtime_seconds / atom_count
            if atom_count is not None and atom_count > 0
            else np.nan
        )
        summary["cpu_peak_rss_mib"] = peak_rss_mib
        summary["cpu_peak_rss_mib_per_atom"] = (
            peak_rss_mib / atom_count
            if atom_count is not None
            and atom_count > 0
            and np.isfinite(peak_rss_mib)
            else np.nan
        )

        summaries.append(summary)
        write_summary(summaries, args.summary_file)

        print(
            f"Finished {run_id}: "
            f"{runtime_seconds:.3f} seconds, "
            f"{peak_rss_mib:.1f} MiB peak CPU RAM"
        )

    print(f"Summary saved to {args.summary_file}")


def clean_path(value):
    if value is None or pd.isna(value):
        return None

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return None

    return Path(value)


def force_change_rms(before_path, after_path):
    before_path = clean_path(before_path)
    after_path = clean_path(after_path)

    if (
        before_path is None
        or after_path is None
        or not before_path.exists()
        or not after_path.exists()
    ):
        return np.nan

    try:
        before = pd.read_csv(before_path)
        after = pd.read_csv(after_path)
    except Exception:
        return np.nan

    required = ["atom_index", "fx", "fy", "fz"]

    if not set(required).issubset(before.columns):
        return np.nan
    if not set(required).issubset(after.columns):
        return np.nan

    merged = before[required].merge(
        after[required],
        on="atom_index",
        suffixes=("_before", "_after"),
    )

    if merged.empty:
        return np.nan

    delta = np.column_stack([
        merged["fx_after"] - merged["fx_before"],
        merged["fy_after"] - merged["fy_before"],
        merged["fz_after"] - merged["fz_before"],
    ])

    magnitudes = np.linalg.norm(delta, axis=1)
    return float(np.sqrt(np.mean(magnitudes ** 2)))


def relaxation_converged(row):
    after_force_path = clean_path(row.get("after_force_csv"))
    relax_fmax = as_float(row.get("relax_fmax"))

    if after_force_path is None or not np.isfinite(relax_fmax):
        return np.nan

    data_path = (
        after_force_path.parent
        / "after_attack_relaxation_data.csv"
    )

    if not data_path.exists():
        return np.nan

    try:
        data = pd.read_csv(data_path)
    except Exception:
        return np.nan

    force_columns = [
        column
        for column in data.columns
        if str(column).startswith("Max Force")
    ]

    if data.empty or not force_columns:
        return np.nan

    final_force = as_float(data[force_columns[0]].iloc[-1])

    if not np.isfinite(final_force):
        return np.nan

    return float(final_force <= relax_fmax)


def load_summaries(mace_summary, uma_summary):
    frames = []

    for calculator, path in [
        ("mace", Path(mace_summary)),
        ("uma", Path(uma_summary)),
    ]:
        if not path.exists():
            print(f"Warning: missing {path}")
            continue

        frame = pd.read_csv(path)
        frame["calculator"] = calculator
        frames.append(frame)

    if not frames:
        raise SystemExit("No MACE or UMA summaries were found.")

    return pd.concat(frames, ignore_index=True, sort=False)


def prepare_metrics(records):
    data = records.copy()

    numeric_columns = [
        "epsilon",
        "n_steps",
        "unit_cell_atoms",
        "supercell_atoms",
        "runtime_seconds",
        "runtime_seconds_per_atom",
        "cpu_peak_rss_mib",
        "cpu_peak_rss_mib_per_atom",
        "before_relax_steps",
        "after_relax_steps",
        "mean_displacement",
        "max_displacement",
        "final_energy",
        "neighbor_jaccard_distance",
        "rdf_l1_distance",
        "coordination_change_mean",
        "coordination_change_max",
    ]

    for column in numeric_columns:
        if column not in data.columns:
            data[column] = np.nan

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if "base_material_slug" not in data.columns:
        data["base_material_slug"] = "unknown"

    data["attack_label"] = data.apply(attack_label, axis=1)
    data["is_step_sweep"] = (
        data["run_id"]
        .astype(str)
        .str.contains("_steps", regex=False)
    )

    data["pre_relax_force_change_rms"] = data.apply(
        lambda row: force_change_rms(
            row.get("before_force_csv"),
            row.get("perturbed_force_csv"),
        ),
        axis=1,
    )

    data["post_relax_force_change_rms"] = data.apply(
        lambda row: force_change_rms(
            row.get("before_force_csv"),
            row.get("after_force_csv"),
        ),
        axis=1,
    )

    data["relaxation_converged"] = data.apply(
        relaxation_converged,
        axis=1,
    )

    data["final_energy_per_atom"] = np.where(
        data["supercell_atoms"] > 0,
        data["final_energy"] / data["supercell_atoms"],
        np.nan,
    )

    return data


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
        "legend.frameon": False,
    })


def plot_attack_panels(data, metric, ylabel, output_path):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 3.7),
    )

    for axis, attack in zip(axes, ATTACKS):
        attack_data = data[data["attack_label"] == attack]

        for calculator in ["mace", "uma"]:
            selected = attack_data[
                attack_data["calculator"] == calculator
            ].dropna(subset=["supercell_atoms", metric])

            if selected.empty:
                continue

            grouped = selected.groupby("supercell_atoms")[metric]
            median = grouped.median()
            q1 = grouped.quantile(0.25)
            q3 = grouped.quantile(0.75)
            x = median.index.to_numpy(dtype=float)

            axis.scatter(
                selected["supercell_atoms"],
                selected[metric],
                s=13,
                alpha=0.25,
                color=COLORS[calculator],
            )
            axis.plot(
                x,
                median.to_numpy(dtype=float),
                marker="o",
                linewidth=1.8,
                color=COLORS[calculator],
                label=calculator.upper(),
            )
            axis.fill_between(
                x,
                q1.to_numpy(dtype=float),
                q3.to_numpy(dtype=float),
                alpha=0.15,
                color=COLORS[calculator],
            )

        axis.set_title(attack)
        axis.set_xlabel("Supercell atoms")
        axis.set_ylabel(ylabel)

        if axis.lines:
            axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_vs_memory(data, output_path):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 3.7),
    )

    color_mappable = None

    for axis, attack in zip(axes, ATTACKS):
        attack_data = data[data["attack_label"] == attack]

        for calculator in ["mace", "uma"]:
            selected = attack_data[
                attack_data["calculator"] == calculator
            ].dropna(subset=[
                "runtime_seconds_per_atom",
                "cpu_peak_rss_mib_per_atom",
                "supercell_atoms",
            ])

            if selected.empty:
                continue

            color_mappable = axis.scatter(
                selected["cpu_peak_rss_mib_per_atom"],
                selected["runtime_seconds_per_atom"],
                c=selected["supercell_atoms"],
                cmap="viridis",
                marker="o" if calculator == "mace" else "^",
                edgecolor=COLORS[calculator],
                linewidth=0.8,
                alpha=0.8,
                label=calculator.upper(),
            )

        axis.set_title(attack)
        axis.set_xlabel("Peak CPU RAM (MiB/atom)")
        axis.set_ylabel("Runtime (s/atom)")

        if axis.collections:
            axis.legend()

    if color_mappable is not None:
        colorbar = fig.colorbar(
            color_mappable,
            ax=axes,
            shrink=0.85,
            pad=0.02,
        )
        colorbar.set_label("Supercell atoms")

    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(data, output_path):
    fig, axis = plt.subplots(figsize=(7.5, 4.5))

    for calculator in ["mace", "uma"]:
        for attack in ATTACKS:
            selected = data[
                (data["calculator"] == calculator)
                & (data["attack_label"] == attack)
            ].dropna(subset=[
                "supercell_atoms",
                "relaxation_converged",
            ])

            if selected.empty:
                continue

            grouped = (
                selected.groupby("supercell_atoms")[
                    "relaxation_converged"
                ]
                .mean()
                * 100.0
            )

            axis.plot(
                grouped.index,
                grouped.values,
                color=COLORS[calculator],
                linestyle=LINESTYLES[attack],
                marker="o",
                label=f"{calculator.upper()} {attack}",
            )

    axis.set_xlabel("Supercell atoms")
    axis.set_ylabel("Relaxation convergence (%)")
    axis.set_ylim(-2, 102)

    if axis.lines:
        axis.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_size_consistency(data, output_path):
    fig, axis = plt.subplots(figsize=(8, 4.8))

    for calculator in ["mace", "uma"]:
        calculator_data = data[
            data["calculator"] == calculator
        ]

        for material, selected in calculator_data.groupby(
            "base_material_slug"
        ):
            selected = selected.dropna(subset=[
                "supercell_atoms",
                "final_energy_per_atom",
            ])

            if selected.empty:
                continue

            grouped = (
                selected.groupby("supercell_atoms")[
                    "final_energy_per_atom"
                ]
                .median()
            )

            axis.plot(
                grouped.index,
                grouped.values,
                marker="o",
                linewidth=1.3,
                color=COLORS[calculator],
                alpha=0.8,
                label=f"{calculator.upper()} {material}",
            )

    axis.set_xlabel("Supercell atoms")
    axis.set_ylabel("Final energy per atom (eV/atom)")

    if axis.lines:
        axis.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def force_angle_median(before_path, after_path):
    before_path = clean_path(before_path)
    after_path = clean_path(after_path)

    if (
        before_path is None
        or after_path is None
        or not before_path.exists()
        or not after_path.exists()
    ):
        return np.nan

    try:
        before = pd.read_csv(before_path)
        after = pd.read_csv(after_path)
    except Exception:
        return np.nan

    columns = ["atom_index", "fx", "fy", "fz"]
    if not set(columns).issubset(before.columns):
        return np.nan
    if not set(columns).issubset(after.columns):
        return np.nan

    merged = before[columns].merge(
        after[columns],
        on="atom_index",
        suffixes=("_before", "_after"),
    )

    first = merged[
        ["fx_before", "fy_before", "fz_before"]
    ].to_numpy(dtype=float)
    second = merged[
        ["fx_after", "fy_after", "fz_after"]
    ].to_numpy(dtype=float)

    denominator = (
        np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
    )
    valid = denominator > 1e-12

    if not np.any(valid):
        return np.nan

    cosine = np.sum(first[valid] * second[valid], axis=1)
    cosine = np.clip(cosine / denominator[valid], -1.0, 1.0)
    return float(np.median(np.degrees(np.arccos(cosine))))


def plot_metric_by_atoms(data, metric, ylabel, title, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    for axis, attack in zip(axes, ATTACKS):
        attack_data = data[data["attack_label"] == attack]

        for calculator in ["mace", "uma"]:
            selected = attack_data[
                attack_data["calculator"] == calculator
            ].dropna(subset=["supercell_atoms", metric])

            if selected.empty:
                continue

            grouped = selected.groupby("supercell_atoms")[metric]
            median = grouped.median()
            q1 = grouped.quantile(0.25)
            q3 = grouped.quantile(0.75)
            atoms = median.index.to_numpy(dtype=float)

            axis.scatter(
                selected["supercell_atoms"],
                selected[metric],
                color=COLORS[calculator],
                alpha=0.18,
                s=14,
            )
            axis.plot(
                atoms,
                median.to_numpy(dtype=float),
                color=COLORS[calculator],
                marker="o",
                linewidth=1.8,
                label=calculator.upper(),
            )
            axis.fill_between(
                atoms,
                q1.to_numpy(dtype=float),
                q3.to_numpy(dtype=float),
                color=COLORS[calculator],
                alpha=0.14,
            )

        axis.set_title(attack)
        axis.set_xlabel("Number of atoms")
        axis.set_ylabel(ylabel)

        if axis.lines:
            axis.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_relation_by_atoms(
    data,
    x_metric,
    y_metric,
    xlabel,
    ylabel,
    title,
    output_path,
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    maximum_atoms = data["supercell_atoms"].max()

    for axis, attack in zip(axes, ATTACKS):
        attack_data = data[data["attack_label"] == attack]

        for calculator in ["mace", "uma"]:
            selected = attack_data[
                attack_data["calculator"] == calculator
            ].dropna(
                subset=["supercell_atoms", x_metric, y_metric]
            )

            if selected.empty:
                continue

            grouped = (
                selected.groupby("supercell_atoms")[
                    [x_metric, y_metric]
                ]
                .median()
                .reset_index()
            )

            sizes = 35 + 145 * (
                grouped["supercell_atoms"] / maximum_atoms
            )

            axis.scatter(
                grouped[x_metric],
                grouped[y_metric],
                s=sizes,
                color=COLORS[calculator],
                alpha=0.7,
                edgecolor="white",
                linewidth=0.7,
                label=calculator.upper(),
            )

        axis.set_title(attack)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

        if axis.collections:
            axis.legend()

    fig.suptitle(title + "\nBubble size represents atom count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def force_component_rms(before_path, after_path, component):
    before_path = clean_path(before_path)
    after_path = clean_path(after_path)

    if (
        before_path is None
        or after_path is None
        or not before_path.exists()
        or not after_path.exists()
    ):
        return np.nan

    try:
        before = pd.read_csv(before_path)
        after = pd.read_csv(after_path)
    except Exception:
        return np.nan

    column = {"x": "fx", "y": "fy", "z": "fz"}[component]
    required = ["atom_index", column]

    if not set(required).issubset(before.columns):
        return np.nan
    if not set(required).issubset(after.columns):
        return np.nan

    merged = before[required].merge(
        after[required],
        on="atom_index",
        suffixes=("_before", "_after"),
    )

    if merged.empty:
        return np.nan

    difference = (
        merged[f"{column}_after"]
        - merged[f"{column}_before"]
    ).to_numpy(dtype=float)

    return float(np.sqrt(np.mean(difference ** 2)))


def displacement_component(row, component):
    force_path = clean_path(row.get("after_force_csv"))

    if force_path is None:
        return np.nan

    run_dir = force_path.parent
    before_path = run_dir / "before_attack_relaxation.traj"
    after_path = run_dir / "final_relaxed.cif"

    if not before_path.exists() or not after_path.exists():
        return np.nan

    try:
        from ase.geometry import find_mic
        from ase.io import read

        before = read(before_path, index=-1)
        after = read(after_path)

        difference = after.positions - before.positions
        difference, _ = find_mic(
            difference,
            before.cell,
            pbc=before.pbc,
        )

        axis = {"x": 0, "y": 1, "z": 2}[component]
        return float(np.median(np.abs(difference[:, axis])))
    except Exception:
        return np.nan


def make_component_figures(data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for component in ["x", "y", "z"]:
        plot_metric_by_atoms(
            data,
            f"delta_force_{component}",
            rf"{component.upper()} force-change RMS (eV/$\AA$)",
            f"{component.upper()} force change vs atoms",
            output_dir
            / f"delta_force_{component}_by_atoms.png",
        )

        plot_metric_by_atoms(
            data,
            f"displacement_{component}",
            rf"Median absolute {component.upper()} displacement ($\AA$)",
            f"{component.upper()} displacement vs atoms",
            output_dir
            / f"displacement_{component}_by_atoms.png",
        )


def make_figures_1_to_9(data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_figures = [
        (
            1,
            "after_relax_steps",
            "Relaxation steps",
            "Relaxation steps vs atoms",
            "relaxation_steps",
        ),
        (
            2,
            "post_relax_force_change_rms",
            r"Force-change RMS (eV/$\AA$)",
            "Delta force vs atoms",
            "delta_force",
        ),
        (
            3,
            "mean_displacement",
            r"Mean displacement ($\AA$)",
            "Displacement vs atoms",
            "displacement",
        ),
        (
            4,
            "post_relax_force_angle",
            "Median force-vector angle (degrees)",
            "Delta-force angle vs atoms",
            "delta_force_angle",
        ),
        (
            5,
            "neighbor_jaccard_distance",
            "Neighbor Jaccard distance",
            "Neighbor Jaccard distance vs atoms",
            "jaccard_distance",
        ),
        (
            6,
            "rdf_l1_distance",
            "RDF L1 distance",
            "RDF L1 distance vs atoms",
            "rdf_l1_distance",
        ),
    ]

    for number, metric, ylabel, title, filename in metric_figures:
        plot_metric_by_atoms(
            data,
            metric,
            ylabel,
            title,
            output_dir
            / f"figure_{number}_{filename}_by_atoms.png",
        )

    relations = [
        (
            7,
            "mean_displacement",
            "after_relax_steps",
            r"Mean displacement ($\AA$)",
            "Relaxation steps",
            "Relaxation steps vs displacement by atoms",
            "convergence_vs_displacement",
        ),
        (
            8,
            "post_relax_force_change_rms",
            "after_relax_steps",
            r"Force-change RMS (eV/$\AA$)",
            "Relaxation steps",
            "Relaxation steps vs delta force by atoms",
            "convergence_vs_delta_force",
        ),
        (
            9,
            "mean_displacement",
            "post_relax_force_change_rms",
            r"Mean displacement ($\AA$)",
            r"Force-change RMS (eV/$\AA$)",
            "Delta force vs displacement by atoms",
            "delta_force_vs_displacement",
        ),
    ]

    for (
        number,
        x_metric,
        y_metric,
        xlabel,
        ylabel,
        title,
        filename,
    ) in relations:
        plot_relation_by_atoms(
            data,
            x_metric,
            y_metric,
            xlabel,
            ylabel,
            title,
            output_dir
            / f"figure_{number}_{filename}_by_atoms.png",
        )


def make_topology_figures(data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topology_metrics = [
        (
            "neighbor_jaccard_distance",
            "Neighbor Jaccard distance",
            "jaccard_distance",
        ),
        (
            "rdf_l1_distance",
            "RDF L1 distance",
            "rdf_l1_distance",
        ),
        (
            "coordination_change_max",
            "Maximum coordination-number change",
            "coordination_change",
        ),
    ]

    for metric, label, filename in topology_metrics:
        plot_metric_by_atoms(
            data,
            metric,
            label,
            f"{label} vs atoms",
            output_dir / f"{filename}_by_atoms.png",
        )

        plot_relation_by_atoms(
            data,
            "mean_displacement",
            metric,
            r"Mean displacement ($\AA$)",
            label,
            f"{label} vs displacement by atoms",
            output_dir
            / f"{filename}_vs_displacement_by_atoms.png",
        )

        plot_relation_by_atoms(
            data,
            metric,
            "after_relax_steps",
            label,
            "Relaxation steps",
            f"Relaxation steps vs {label.lower()} by atoms",
            output_dir
            / f"convergence_vs_{filename}_by_atoms.png",
        )

        plot_relation_by_atoms(
            data,
            "post_relax_force_change_rms",
            metric,
            r"Force-change RMS (eV/$\AA$)",
            label,
            f"{label} vs delta force by atoms",
            output_dir
            / f"{filename}_vs_delta_force_by_atoms.png",
        )


def make_material_rankings(data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rankings = [
        ("after_relax_steps", "Relaxation steps"),
        (
            "post_relax_force_change_rms",
            r"Force-change RMS (eV/$\AA$)",
        ),
        (
            "post_relax_force_angle",
            "Force-vector angle (degrees)",
        ),
        ("mean_displacement", r"Mean displacement ($\AA$)"),
        ("neighbor_jaccard_distance", "Jaccard distance"),
        ("rdf_l1_distance", "RDF L1 distance"),
        (
            "coordination_change_max",
            "Maximum coordination-number change",
        ),
    ]

    for metric, xlabel in rankings:
        clean = data.dropna(
            subset=["base_material_slug", "calculator", metric]
        )

        if clean.empty:
            continue

        ranking = (
            clean.groupby(
                ["base_material_slug", "calculator"]
            )[metric]
            .median()
            .unstack("calculator")
        )
        ranking["sort_value"] = ranking.median(axis=1)
        ranking = ranking.sort_values("sort_value")
        ranking = ranking.drop(columns="sort_value")

        fig, axis = plt.subplots(
            figsize=(8, max(5, len(ranking) * 0.32))
        )
        ranking.plot.barh(
            ax=axis,
            color=[
                COLORS.get(column, "#777777")
                for column in ranking.columns
            ],
        )
        axis.set_xlabel(f"Median {xlabel}")
        axis.set_ylabel("Material")
        axis.legend(
            [str(column).upper() for column in ranking.columns]
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"material_{metric}.png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_command(args):
    apply_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove obsolete generic comprehensive plots and directories.
    for path in output_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        elif path.suffix.lower() == ".png":
            path.unlink()

    records = load_summaries(
        args.mace_summary,
        args.uma_summary,
    )
    metrics = prepare_metrics(records)

    metrics["post_relax_force_angle"] = metrics.apply(
        lambda row: force_angle_median(
            row.get("before_force_csv"),
            row.get("after_force_csv"),
        ),
        axis=1,
    )

    for component in ["x", "y", "z"]:
        metrics[f"delta_force_{component}"] = metrics.apply(
            lambda row, axis=component: force_component_rms(
                row.get("before_force_csv"),
                row.get("after_force_csv"),
                axis,
            ),
            axis=1,
        )

        metrics[f"displacement_{component}"] = metrics.apply(
            lambda row, axis=component: displacement_component(
                row,
                axis,
            ),
            axis=1,
        )

    metrics.to_csv(
        output_dir / "supercell_runtime_metrics.csv",
        index=False,
    )

    primary = metrics[
        (
            metrics["status"]
            .astype(str)
            .str.lower()
            == "success"
        )
        & (~metrics["is_step_sweep"])
        & np.isclose(metrics["epsilon"], args.epsilon)
    ].copy()

    if primary.empty:
        raise SystemExit(
            f"No successful supercell rows at epsilon={args.epsilon}"
        )

    # Combined figures 1-9.
    make_figures_1_to_9(primary, output_dir)

    make_component_figures(
        primary,
        output_dir / "components",
    )

    make_topology_figures(
        primary,
        output_dir / "topology",
    )

    # One folder containing figures 1-9 for each base material.
    for material, material_data in primary.groupby(
        "base_material_slug"
    ):
        safe_name = (
            str(material)
            .replace("/", "_")
            .replace("\\", "_")
        )

        make_figures_1_to_9(
            material_data,
            output_dir / safe_name,
        )

        make_component_figures(
            material_data,
            output_dir / "components" / safe_name,
        )

        make_topology_figures(
            material_data,
            output_dir / "topology" / safe_name,
        )

    # Raw computational-scaling plots.
    plot_attack_panels(
        primary,
        "runtime_seconds",
        "Total runtime (seconds)",
        output_dir / "runtime_vs_atoms.png",
    )
    plot_attack_panels(
        primary,
        "runtime_seconds_per_atom",
        "Runtime (seconds/atom)",
        output_dir / "runtime_per_atom_vs_atoms.png",
    )
    plot_attack_panels(
        primary,
        "cpu_peak_rss_mib",
        "Peak CPU RAM (MiB)",
        output_dir / "cpu_ram_vs_atoms.png",
    )
    plot_attack_panels(
        primary,
        "cpu_peak_rss_mib_per_atom",
        "Peak CPU RAM (MiB/atom)",
        output_dir / "cpu_ram_per_atom_vs_atoms.png",
    )
    plot_runtime_vs_memory(
        primary,
        output_dir / "runtime_vs_cpu_ram_per_atom.png",
    )
    plot_convergence(
        primary,
        output_dir / "relaxation_convergence_vs_atoms.png",
    )

    make_material_rankings(
        primary,
        output_dir / "materials_ranking",
    )

    print(f"Supercell plots written to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--tests",
        required=True,
        type=Path,
    )
    run_parser.add_argument(
        "--summary-file",
        required=True,
        type=Path,
    )

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument(
        "--mace-summary",
        required=True,
        type=Path,
    )
    plot_parser.add_argument(
        "--uma-summary",
        required=True,
        type=Path,
    )
    plot_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    plot_parser.add_argument(
        "--epsilon",
        default=0.1,
        type=float,
    )

    args = parser.parse_args()

    if args.command == "run":
        run_command(args)
    else:
        plot_command(args)


if __name__ == "__main__":
    main()
