from pathlib import Path
import sys

import numpy as np
from ase import Atoms


scripts_dir = Path(__file__).resolve().parents[1] / "scripts_python"
sys.path.insert(0, str(scripts_dir))


def test_ase_repeat_scales_atoms_and_cell():
    unit_cell = Atoms(
        "NaCl",
        positions=[
            [0, 0, 0],
            [1, 1, 1],
        ],
        cell=[2, 3, 4],
        pbc=True,
    )

    repeated = unit_cell.repeat((2, 1, 2))

    assert len(repeated) == len(unit_cell) * 2 * 1 * 2
    assert np.allclose(
        repeated.cell.lengths(),
        [4, 3, 8],
    )
