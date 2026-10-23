from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float


@dataclass(frozen=True)
class Rectangle:
    x_min: float
    y_min: float
    width: float
    height: float

    @property
    def x_max(self) -> float:
        return self.x_min + self.width

    @property
    def y_max(self) -> float:
        return self.y_min + self.height

    def contains(self, point):
        x, y = point
        return (
            self.x_min - 1e-9 <= x <= self.x_max + 1e-9
            and self.y_min - 1e-9 <= y <= self.y_max + 1e-9
        )


Obstacle = Circle | Rectangle


def point_to_segment_distance(a, b, p):
    """Distance from point p to segment ab."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        return float(np.hypot(px - ax, py - ay))
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = min(1.0, max(0.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return float(np.hypot(px - qx, py - qy))


def segment_circle_intersects(a, b, c, r):
    """True if segment ab touches or intersects circle center c radius r."""
    return point_to_segment_distance(a, b, c) <= r + 1e-9


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, c):
    return (
        min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
        and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
    )


def segment_segment_intersects(a, b, c, d):
    """Segment intersection test with collinear overlap handling."""
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if abs(o1) <= 1e-9 and abs(o2) <= 1e-9 and abs(o3) <= 1e-9 and abs(o4) <= 1e-9:
        return (
            _on_segment(a, b, c)
            or _on_segment(a, b, d)
            or _on_segment(c, d, a)
            or _on_segment(c, d, b)
        )

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def segment_rect_intersects(a, b, rect):
    """True if segment ab meets or touches the rectangle boundary or endpoints."""
    if rect.contains(a) or rect.contains(b):
        return True
    edges = [
        ((rect.x_min, rect.y_min), (rect.x_max, rect.y_min)),
        ((rect.x_max, rect.y_min), (rect.x_max, rect.y_max)),
        ((rect.x_max, rect.y_max), (rect.x_min, rect.y_max)),
        ((rect.x_min, rect.y_max), (rect.x_min, rect.y_min)),
    ]
    return any(segment_segment_intersects(a, b, e1, e2) for e1, e2 in edges)
