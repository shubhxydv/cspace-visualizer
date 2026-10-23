from __future__ import annotations

import numpy as np

from config import BASE, L1, L2, GRID_N, normalize_angle_deg
from core.arm import Arm
from core.collision import arm_collides
from core.cspace import CSpace


def test_cspace_shape_and_dtype():
    cspace = CSpace(N=72)
    assert cspace.occ.shape == (72, 72)
    assert cspace.occ.dtype == np.bool_


def test_angle_index_round_trip():
    cspace = CSpace(N=72)
    for value in [0.0, 45.0, 359.0, 720.0, -10.0, 365.0]:
        idx = cspace.angle_to_index(value)
        assert 0 <= idx < cspace.N
        angle = cspace.index_to_angle(idx)
        assert 0.0 <= angle < 360.0


def test_wraparound_neighbours():
    cspace = CSpace(N=72)
    assert (71, 0) in cspace.neighbours(0, 0)
    assert (0, 71) in cspace.neighbours(0, 0)
    assert (71, 71) in cspace.neighbours(71, 71)
    assert (0, 0) in cspace.neighbours(71, 71)


def test_grid_matches_scalar_collision_on_sample_angles():
    arm = Arm(BASE, L1, L2)
    cspace = CSpace(N=72, arm=arm)
    angles = np.linspace(0.0, 360.0, 200, endpoint=False)
    for q1 in angles:
        for q2 in angles:
            a1 = normalize_angle_deg(float(q1))
            a2 = normalize_angle_deg(float(q2))
            idx1 = cspace.angle_to_index(a1)
            idx2 = cspace.angle_to_index(a2)
            scalar = arm_collides(a1, a2, arm)
            grid_value = bool(cspace.occ[idx1, idx2])
            assert scalar == grid_value
