from pathlib import Path
import csv
import sys

import numpy as np
from ase import Atoms


scripts_dir = Path(__file__).resolve().parents[1] / "scripts_python"
sys.path.insert(0, str(scripts_dir))

import supercell


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_repeat_tuples_contains_every_1_to_3_combination():
    repeats = supercell.repeat_tuples()

    assert len(repeats) == 27
    assert len(set(repeats)) == 27
    assert (1, 1, 1) in repeats
    assert (3, 3, 3) in repeats


def test_ase_repeat_scales_atom_count_and_cell():
    unit_cell = Atoms(
        "NaCl",
        positions=[[0, 0, 0], [1, 1, 1]],
        cell=[2, 3, 4],
        pbc=True,
    )

    repeated = unit_cell.repeat((2, 3, 1))

    assert len(repeated) == len(unit_cell) * 2 * 3 * 1
    assert np.allclose(repeated.cell.lengths(), [4, 9, 4])


def test_sweep_row_has_clear_folder_name_and_metadata():
    context = {
        "device": "cpu",
        "relax_fmax": 0.01,
        "relax_max_steps": 300,
        "relax_optimizer": "LBFGS",
        "base_material_slug": "graphite",
        "base_material_label": "Graphite",
        "base_input_path": "graphite.cif",
    }
    repeated = {
        "material_slug": "graphite_r1x2x3",
        "material_label": "Graphite_r1x2x3",
        "input_path": "graphite_r1x2x3.cif",
        "repeat_x": 1,
        "repeat_y": 2,
        "repeat_z": 3,
        "repeat_tuple": "1x2x3",
        "unit_cell_atoms": 4,
        "supercell_atoms": 24,
    }
    model = {
        "calculator": "mace",
        "model_path": "mace.model",
        "mace_head": "omat_pbe",
    }
    attack = {
        "name": "pgd",
        "attack_type": "pgd",
        "alpha_ratio": 0.1,
    }

    row = supercell.make_run_row(
        context,
        repeated,
        model,
        attack,
        epsilon=0.1,
        n_steps=20,
        sweep=True,
    )

    assert row["run_folder"] == "pgd_eps01_steps020"
    assert row["run_id"] == "graphite_r1x2x3_mace_pgd_eps01_steps020"
    assert row["supercell_repeat_tuple"] == "1x2x3"
    assert row["supercell_atoms"] == 24
    assert row["alpha"] == "0.01"
