from pathlib import Path
import sys

import numpy as np
import pandas as pd
from ase import Atoms


scripts_dir = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(scripts_dir))

import contour

def test_parse_betas_uses_configured_defaults():
    assert contour.parse_betas(None, [0.1, 0.05, 0.0]) == [0.1, 0.05, 0.0]


def test_parse_betas_accepts_comma_separated_values():
    assert contour.parse_betas("0.2, 0.1, 0", []) == [0.2, 0.1, 0.0]


def test_beta_tag_is_stable():
    assert contour.beta_tag(0.1) == "beta_100"
    assert contour.beta_tag(0.05) == "beta_050"
    assert contour.beta_tag(0.0) == "beta_000"


def test_initial_velocities_are_repeatable_and_centered():
    first = Atoms("H4", positions=np.zeros((4, 3)))
    second = first.copy()

    contour.initial_velocities(first, seed=42)
    contour.initial_velocities(second, seed=42)

    assert np.allclose(first.get_velocities(), second.get_velocities())
    assert np.allclose(first.get_velocities().mean(axis=0), 0.0)


def test_out_of_plane_angle_for_right_angle_is_90_degrees():
    first = np.array([[0.0, 0.0, 0.0]])
    second = np.array([[1.0, 0.0, 0.0]])
    third = np.array([[1.0, 1.0, 0.0]])

    angle = contour.out_of_plane_angle(first, second, third)

    assert np.isclose(angle, 90.0)
