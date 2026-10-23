from __future__ import annotations

import numpy as np

from core.obstacles import Circle, Rectangle, segment_circle_intersects, segment_rect_intersects


def test_segment_circle_cases():
    assert not segment_circle_intersects((0.0, 5.0), (1.0, 5.0), (0.0, 0.0), 0.5)
    assert segment_circle_intersects((0.0, 2.0), (10.0, 2.0), (5.0, 0.0), 2.0)
    assert segment_circle_intersects((0.0, 2.0), (10.0, 2.0), (5.0, 0.0), 2.0)
    assert segment_circle_intersects((0.0, 1.0), (2.0, 1.0), (1.0, 0.0), 1.0)
    assert segment_circle_intersects((1.0, 0.0), (1.0, 0.0), (0.0, 0.0), 1.0)


def test_segment_rect_cases():
    rect = Rectangle(0.0, 0.0, 2.0, 2.0)
    assert segment_rect_intersects((0.5, 0.5), (1.0, 1.0), rect)
    assert segment_rect_intersects((-1.0, 1.0), (3.0, 1.0), rect)
    assert not segment_rect_intersects((-1.0, -1.0), (-0.5, -0.5), rect)
    assert segment_rect_intersects((0.0, 0.0), (0.0, 0.0), rect)
    assert segment_rect_intersects((1.0, 3.0), (2.0, 3.0), rect) is False


def test_zero_length_segment_behaviour():
    rect = Rectangle(0.0, 0.0, 2.0, 2.0)
    assert segment_rect_intersects((1.0, 1.0), (1.0, 1.0), rect)
    assert segment_circle_intersects((1.0, 1.0), (1.0, 1.0), (0.0, 0.0), 0.5)
    assert not segment_circle_intersects((5.0, 5.0), (5.0, 5.0), (0.0, 0.0), 0.5)


def test_collision_vectorized_matches_scalar():
    from core.collision import Arm, arm_collides, collides_vectorized
    arm = Arm()
    obstacles = [Circle(0.0, 0.0, 0.25), Rectangle(-1.0, -0.5, 0.5, 0.5)]
    theta1 = np.linspace(0.0, 180.0, 200)
    theta2 = np.linspace(0.0, 180.0, 200)
    vec = collides_vectorized(theta1, theta2, arm, obstacles)
    for i in range(theta1.size):
        assert vec.flat[i] == arm_collides(theta1.flat[i], theta2.flat[i], arm, obstacles)
