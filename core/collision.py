from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import BASE, L1, L2
from core.obstacles import Circle, Rectangle, segment_circle_intersects, segment_rect_intersects


@dataclass(frozen=True)
class Arm:
    base: tuple[float, float] = BASE
    L1: float = L1
    L2: float = L2

    def fk_scalar(self, theta1_deg: float, theta2_deg: float):
        t1 = np.deg2rad(theta1_deg)
        t2 = np.deg2rad(theta2_deg)
        elbow = (self.base[0] + self.L1 * np.cos(t1), self.base[1] + self.L1 * np.sin(t1))
        tip = (
            elbow[0] + self.L2 * np.cos(t1 + t2),
            elbow[1] + self.L2 * np.sin(t1 + t2),
        )
        return elbow, tip

    def fk_vectorized(self, theta1_deg, theta2_deg):
        theta1 = np.asarray(theta1_deg, dtype=float)
        theta2 = np.asarray(theta2_deg, dtype=float)
        t1 = np.deg2rad(theta1)
        t2 = np.deg2rad(theta2)
        elbow_x = self.base[0] + self.L1 * np.cos(t1)
        elbow_y = self.base[1] + self.L1 * np.sin(t1)
        tip_x = elbow_x + self.L2 * np.cos(t1 + t2)
        tip_y = elbow_y + self.L2 * np.sin(t1 + t2)
        elbow = np.stack([elbow_x, elbow_y], axis=-1)
        tip = np.stack([tip_x, tip_y], axis=-1)
        return elbow, tip


def arm_collides(theta1_deg, theta2_deg, arm=None, obstacles=None):
    if obstacles is None:
        obstacles = []
    if arm is None:
        arm = Arm()
    t1 = np.deg2rad(theta1_deg)
    elbow = (arm.base[0] + arm.L1 * np.cos(t1), arm.base[1] + arm.L1 * np.sin(t1))
    tip = (
        elbow[0] + arm.L2 * np.cos(t1 + np.deg2rad(theta2_deg)),
        elbow[1] + arm.L2 * np.sin(t1 + np.deg2rad(theta2_deg)),
    )
    for obs in obstacles:
        for seg_a, seg_b in [((arm.base[0], arm.base[1]), elbow), (elbow, tip)]:
            if isinstance(obs, Circle):
                if segment_circle_intersects(seg_a, seg_b, (obs.cx, obs.cy), obs.r):
                    return True
            elif isinstance(obs, Rectangle):
                if segment_rect_intersects(seg_a, seg_b, obs):
                    return True
    return False


def collides_vectorized(theta1_deg, theta2_deg, arm=None, obstacles=None):
    if obstacles is None:
        obstacles = []
    if arm is None:
        arm = Arm()
    theta1 = np.asarray(theta1_deg, dtype=float)
    theta2 = np.asarray(theta2_deg, dtype=float)
    out = np.zeros_like(theta1, dtype=bool)
    for i in range(theta1.size):
        out.flat[i] = arm_collides(theta1.flat[i], theta2.flat[i], arm, obstacles)
    return out
