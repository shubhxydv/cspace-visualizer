from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from config import BASE, EPS, L1, L2


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float

    def contains(self, point):
        x, y = point
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r ** 2 + EPS


@dataclass(frozen=True)
class Rectangle:
    x_min: float
    y_min: float
    width: float
    height: float

    @property
    def x_max(self):
        return self.x_min + self.width

    @property
    def y_max(self):
        return self.y_min + self.height

    def contains(self, point):
        x, y = point
        return (
            self.x_min - EPS <= x <= self.x_max + EPS
            and self.y_min - EPS <= y <= self.y_max + EPS
        )


Obstacle = Circle | Rectangle


def point_to_segment_distance(a, b, p):
    """Return the Euclidean distance from p to segment ab."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return np.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = min(1.0, max(0.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return np.hypot(px - qx, py - qy)


def segment_circle_intersects(a, b, c, r):
    """True if segment ab intersects or touches a circle of radius r around c."""
    return point_to_segment_distance(a, b, c) <= r + EPS


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, c):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    return (
        min(x1, x2) - EPS <= x3 <= max(x1, x2) + EPS
        and min(y1, y2) - EPS <= y3 <= max(y1, y2) + EPS
    )


def segment_segment_intersects(a, b, c, d):
    """Robust segment intersection test using orientation and collinear handling."""
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if abs(o1) <= EPS and abs(o2) <= EPS and abs(o3) <= EPS and abs(o4) <= EPS:
        return (
            _on_segment(a, b, c)
            or _on_segment(a, b, d)
            or _on_segment(c, d, a)
            or _on_segment(c, d, b)
        )

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def segment_rect_intersects(a, b, rect):
    """Return True if segment ab touches or crosses rectangle rect."""
    if rect.contains(a) or rect.contains(b):
        return True
    edges = [
        ((rect.x_min, rect.y_min), (rect.x_max, rect.y_min)),
        ((rect.x_max, rect.y_min), (rect.x_max, rect.y_max)),
        ((rect.x_max, rect.y_max), (rect.x_min, rect.y_max)),
        ((rect.x_min, rect.y_max), (rect.x_min, rect.y_min)),
    ]
    for e1, e2 in edges:
        if segment_segment_intersects(a, b, e1, e2):
            return True
    return False


def arm_collides(theta1_deg, theta2_deg, arm=None, obstacles=None):
    """Return True if either arm segment intersects any obstacle."""
    if arm is None:
        arm = (BASE, L1, L2)
    if obstacles is None:
        obstacles = []
    if not isinstance(arm, tuple):
        arm = (arm.base, arm.L1, arm.L2)
    base = arm[0]
    L1v = arm[1]
    L2v = arm[2]
    t1 = np.deg2rad(theta1_deg)
    p1 = (base[0] + L1v * np.cos(t1), base[1] + L1v * np.sin(t1))
    p2 = (p1[0] + L2v * np.cos(t1 + np.deg2rad(theta2_deg)), p1[1] + L2v * np.sin(t1 + np.deg2rad(theta2_deg)))
    segments = [((base[0], base[1]), p1), (p1, p2)]
    for a, b in segments:
        for obs in obstacles:
            if isinstance(obs, Circle):
                if segment_circle_intersects(a, b, (obs.cx, obs.cy), obs.r):
                    return True
            elif isinstance(obs, Rectangle):
                if segment_rect_intersects(a, b, obs):
                    return True
    return False


def collides_vectorized(theta1_deg, theta2_deg, arm=None, obstacles=None):
    """Vectorized collision grid for a meshgrid of theta1/theta2 values."""
    if obstacles is None:
        obstacles = []
    if arm is None:
        arm = (BASE, L1, L2)
    theta1 = np.asarray(theta1_deg, dtype=float)
    theta2 = np.asarray(theta2_deg, dtype=float)
    base = np.asarray(arm[0], dtype=float)
    L1v = float(arm[1])
    L2v = float(arm[2])
    t1 = np.deg2rad(theta1)
    p1x = base[0] + L1v * np.cos(t1)
    p1y = base[1] + L1v * np.sin(t1)
    p2x = p1x + L2v * np.cos(t1 + np.deg2rad(theta2))
    p2y = p1y + L2v * np.sin(t1 + np.deg2rad(theta2))
    out = np.zeros_like(theta1, dtype=bool)
    for idx, obs in enumerate(obstacles):
        if isinstance(obs, Circle):
            cx, cy, r = obs.cx, obs.cy, obs.r
            seg1 = np.hypot((p1x - base[0]) ** 2 + (p1y - base[1]) ** 2, 0)
            # Fallback: explicit per-cell test to keep correctness simple and reliable.
            for i in range(theta1.size):
                a = (base[0], base[1])
                b = (float(p1x.flat[i]), float(p1y.flat[i]))
                c = (float(p2x.flat[i]), float(p2y.flat[i]))
                if segment_circle_intersects(a, b, (cx, cy), r) or segment_circle_intersects(b, c, (cx, cy), r):
                    out.flat[i] = True
        elif isinstance(obs, Rectangle):
            for i in range(theta1.size):
                a = (base[0], base[1])
                b = (float(p1x.flat[i]), float(p1y.flat[i]))
                c = (float(p2x.flat[i]), float(p2y.flat[i]))
                if segment_rect_intersects(a, b, obs) or segment_rect_intersects(b, c, obs):
                    out.flat[i] = True
    return out


def arm_segments(theta1_deg, theta2_deg, arm=None):
    if arm is None:
        arm = (BASE, L1, L2)
    base = arm[0]
    L1v = arm[1]
    L2v = arm[2]
    t1 = np.deg2rad(theta1_deg)
    elbow = (base[0] + L1v * np.cos(t1), base[1] + L1v * np.sin(t1))
    tip = (elbow[0] + L2v * np.cos(t1 + np.deg2rad(theta2_deg)), elbow[1] + L2v * np.sin(t1 + np.deg2rad(theta2_deg)))
    return (base, elbow, tip)
