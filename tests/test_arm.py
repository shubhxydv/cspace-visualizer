from __future__ import annotations

import numpy as np


def test_normalize_angles():
    from config import normalize_angle_deg
    assert normalize_angle_deg(0.0) == 0.0
    assert normalize_angle_deg(360.0) == 0.0
    assert normalize_angle_deg(-1.0) == 359.0
    assert normalize_angle_deg(359.0) == 359.0


def test_fk_scalar_cases():
    from config import L1, L2
    from core.arm import Arm
    arm = Arm()
    elbow, tip = arm.fk_scalar(0, 0)
    assert np.allclose(elbow, (L1, 0.0), atol=1e-9)
    assert np.allclose(tip, (L1 + L2, 0.0), atol=1e-9)

    elbow, tip = arm.fk_scalar(90, 0)
    assert np.allclose(elbow, (0.0, L1), atol=1e-9)
    assert np.allclose(tip, (0.0, L1 + L2), atol=1e-9)

    elbow, tip = arm.fk_scalar(0, 90)
    assert np.allclose(elbow, (L1, 0.0), atol=1e-9)
    assert np.allclose(tip, (L1, L2), atol=1e-9)

    elbow, tip = arm.fk_scalar(0, 180)
    assert np.allclose(tip, (L1 - L2, 0.0), atol=1e-9)


def test_fk_vectorized_matches_scalar():
    from core.arm import Arm
    rng = np.random.default_rng(0)
    arm = Arm()
    theta1 = rng.uniform(0.0, 360.0, size=1000)
    theta2 = rng.uniform(0.0, 360.0, size=1000)
    elbow_vec, tip_vec = arm.fk_vectorized(theta1, theta2)
    for i in range(1000):
        elbow_s, tip_s = arm.fk_scalar(theta1[i], theta2[i])
        assert np.allclose(elbow_vec[i], elbow_s, atol=1e-9)
        assert np.allclose(tip_vec[i], tip_s, atol=1e-9)


def test_angle_periodicity():
    from core.arm import Arm
    arm = Arm()
    s1, s2 = arm.fk_scalar(30, 45)
    p1, p2 = arm.fk_scalar(390, 45)
    q1, q2 = arm.fk_scalar(30, -315)
    assert np.allclose(s1, p1, atol=1e-9)
    assert np.allclose(s2, p2, atol=1e-9)
    assert np.allclose(s1, q1, atol=1e-9)
    assert np.allclose(s2, q2, atol=1e-9)
