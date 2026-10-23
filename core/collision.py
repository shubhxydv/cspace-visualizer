from __future__ import annotations

import numpy as np

from config import EPS


def segment_circle_intersects(a, b, c, r):
    """Distance from centre c to segment ab is <= r, with zero-length support."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx = bx - ax
    dy = by - ay
    if abs(dx) <= EPS and abs(dy) <= EPS:
        return np.hypot(cx - ax, cy - ay) <= r + EPS
    t = ((cx - ax) * dx + (cy - ay) * dy) / (dx * dx + dy * dy)
    t = min(1.0, max(0.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return np.hypot(cx - qx, cy - qy) <= r + EPS


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, c):
    return (
        min(a[0], b[0]) - EPS <= c[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= c[1] <= max(a[1], b[1]) + EPS
    )


def segment_segment_intersects(a, b, c, d):
    """Return True if segments ab and cd intersect or are collinear and overlapping."""
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if abs(o1) <= EPS and abs(o2) <= EPS and abs(o3) <= EPS and abs(o4) <= EPS:
        return _on_segment(a, b, c) or _on_segment(a, b, d) or _on_segment(c, d, a) or _on_segment(c, d, b)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def segment_rect_intersects(a, b, rect):
    """Return True if segment ab intersects rectangle edges or endpoints fall inside."""
    if rect.contains(a) or rect.contains(b):
        return True
    edges = [
        ((rect.x_min, rect.y_min), (rect.x_max, rect.y_min)),
        ((rect.x_max, rect.y_min), (rect.x_max, rect.y_max)),
        ((rect.x_max, rect.y_max), (rect.x_min, rect.y_max)),
        ((rect.x_min, rect.y_max), (rect.x_min, rect.y_min)),
    ]
    return any(segment_segment_intersects(a, b, e1, e2) for e1, e2 in edges)
