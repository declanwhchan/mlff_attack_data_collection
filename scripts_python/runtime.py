#!/usr/bin/env python3

from pathlib import Path
import argparse
import gc
import threading
import time

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
        "mean_displacement",
        "max_displacement",
        "final_energy",
        "neighbor_jaccard_distance",
        "neighbor_edge_change_count",
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


def plot_robustness(data, output_path, title):
    metrics = [
        (
            "pre_relax_force_change_rms",
            r"Pre-relax force change RMS (eV/$\AA$)",
        ),
        (
            "post_relax_force_change_rms",
            r"Post-relax force change RMS (eV/$\AA$)",
        ),
        (
            "max_displacement",
            r"Maximum displacement ($\AA$)",
        ),
        (
            "neighbor_jaccard_distance",
            "Neighbor Jaccard distance",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for axis, (metric, ylabel) in zip(axes.flat, metrics):
        for calculator in ["mace", "uma"]:
            for attack in ATTACKS:
                selected = data[
                    (data["calculator"] == calculator)
                    & (data["attack_label"] == attack)
                ].dropna(subset=["supercell_atoms", metric])

                if selected.empty:
                    continue

                grouped = (
                    selected.groupby("supercell_atoms")[metric]
                    .median()
                )

                axis.plot(
                    grouped.index,
                    grouped.values,
                    color=COLORS[calculator],
                    linestyle=LINESTYLES[attack],
                    marker="o",
                    markersize=3,
                    linewidth=1.4,
                    label=f"{calculator.upper()} {attack}",
                )

        axis.set_xlabel("Supercell atoms")
        axis.set_ylabel(ylabel)

    handles, labels = axes.flat[0].get_legend_handles_labels()

    if handles:
        unique = dict(zip(labels, handles))
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.99),
        )

    fig.suptitle(title, y=1.04)
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


def plot_command(args):
    apply_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_summaries(
        args.mace_summary,
        args.uma_summary,
    )
    metrics = prepare_metrics(records)

    metrics.to_csv(
        output_dir / "supercell_runtime_metrics.csv",
        index=False,
    )

    primary = metrics[
        (metrics["status"].astype(str).str.lower() == "success")
        & (~metrics["is_step_sweep"])
        & np.isclose(metrics["epsilon"], args.epsilon)
    ].copy()

    if primary.empty:
        raise SystemExit(
            f"No successful primary rows at epsilon={args.epsilon}"
        )

    plot_attack_panels(
        primary,
        "runtime_seconds_per_atom",
        "Runtime (s/atom)",
        output_dir / "runtime_per_atom_vs_supercell_atoms.png",
    )

    plot_attack_panels(
        primary,
        "cpu_peak_rss_mib_per_atom",
        "Peak CPU RAM (MiB/atom)",
        output_dir / "cpu_memory_per_atom_vs_supercell_atoms.png",
    )

    plot_runtime_vs_memory(
        primary,
        output_dir / "runtime_vs_cpu_memory_per_atom.png",
    )

    plot_robustness(
        primary,
        output_dir / "supercell_robustness_dashboard.png",
        "Supercell robustness",
    )

    plot_convergence(
        primary,
        output_dir / "supercell_relaxation_convergence.png",
    )

    plot_size_consistency(
        primary,
        output_dir / "supercell_size_consistency.png",
    )

    for material, selected in primary.groupby(
        "base_material_slug"
    ):
        safe_material = (
            str(material)
            .replace("/", "_")
            .replace("\\", "_")
        )

        plot_robustness(
            selected,
            output_dir
            / f"supercell_{safe_material}_robustness_dashboard.png",
            f"Supercell robustness: {material}",
        )

    print(f"CPU runtime plots written to {output_dir}")


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
