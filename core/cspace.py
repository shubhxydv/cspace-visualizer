from __future__ import annotations

import numpy as np

from config import BASE, EPS, L1, L2
from core.arm import Circle, Rectangle, arm_collides
from core.obstacles import segment_circle_intersects, segment_rect_intersects


def collides_vectorized(theta1_deg, theta2_deg, arm=None, obstacles=None):
    """Return a bool array describing which (θ1, θ2) pairs are in collision."""
    if obstacles is None:
        obstacles = []
    if arm is None:
        base = BASE
        L1v = L1
        L2v = L2
    else:
        base = arm.base
        L1v = arm.L1
        L2v = arm.L2

    theta1 = np.asarray(theta1_deg, dtype=float)
    theta2 = np.asarray(theta2_deg, dtype=float)
    out = np.zeros_like(theta1, dtype=bool)
    for idx in range(theta1.size):
        t1 = float(theta1.flat[idx])
        t2 = float(theta2.flat[idx])
        t1r = np.deg2rad(t1)
        p1 = (base[0] + L1v * np.cos(t1r), base[1] + L1v * np.sin(t1r))
        p2 = (
            p1[0] + L2v * np.cos(t1r + np.deg2rad(t2)),
            p1[1] + L2v * np.sin(t1r + np.deg2rad(t2)),
        )
        for obs in obstacles:
            if isinstance(obs, Circle):
                if segment_circle_intersects((base[0], base[1]), p1, (obs.cx, obs.cy), obs.r):
                    out.flat[idx] = True
                    break
                if segment_circle_intersects(p1, p2, (obs.cx, obs.cy), obs.r):
                    out.flat[idx] = True
                    break
            elif isinstance(obs, Rectangle):
                if segment_rect_intersects((base[0], base[1]), p1, obs):
                    out.flat[idx] = True
                    break
                if segment_rect_intersects(p1, p2, obs):
                    out.flat[idx] = True
                    break
    return out


__all__ = ["collides_vectorized", "arm_collides", "segment_circle_intersects", "segment_rect_intersects"]
