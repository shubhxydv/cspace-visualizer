from __future__ import annotations

import numpy as np
from dataclasses import dataclass

BASE = (0.0, 0.0)
L1 = 1.0
L2 = 0.8
GRID_N = 360
EPS = 1e-9


def normalize_angle_deg(angle_deg: float) -> float:
    """Wrap an angle into the continuous domain [0, 360)."""
    value = angle_deg % 360.0
    if np.isclose(value, 360.0):
        return 0.0
    return float(value)


def angle_to_index(angle_deg: float, N: int = GRID_N) -> int:
    """Map an angle in degrees to the nearest cell index on a torus grid."""
    if N <= 0:
        raise ValueError("N must be positive")
    angle = normalize_angle_deg(angle_deg)
    return int(np.rint(angle / (360.0 / N)) % N)


def index_to_angle(index: int, N: int = GRID_N) -> float:
    """Map a grid index back to a degree value."""
    if N <= 0:
        raise ValueError("N must be positive")
    return float((index % N) * (360.0 / N))


def torus_distance_axis(a: int, b: int, N: int) -> int:
    """Minimum torus distance in cell counts between two indices on one axis."""
    d = abs(int(a) - int(b))
    return min(d, N - d)


def torus_neighbours(i: int, j: int, N: int, connectivity: int = 8):
    """Return valid 4/8-connected torus neighbours in a deterministic order."""
    if N <= 0:
        raise ValueError("N must be positive")
    cells = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni = (i + di) % N
            nj = (j + dj) % N
            if connectivity == 4 and abs(di) + abs(dj) != 1:
                continue
            cells.append((ni, nj))
    # deterministic order: up, right, down, left, diagonals, etc.
    ordered = []
    for di, dj in [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        if connectivity == 4 and abs(di) + abs(dj) != 1:
            continue
        if connectivity == 8:
            ni = (i + di) % N
            nj = (j + dj) % N
            ordered.append((ni, nj))
        else:
            if abs(di) + abs(dj) == 1:
                ni = (i + di) % N
                nj = (j + dj) % N
                ordered.append((ni, nj))
    return ordered


def wrap_360(value: float) -> float:
    """Alias for normalize_angle_deg."""
    return normalize_angle_deg(value)


@dataclass(frozen=True)
class Arm:
    base: tuple[float, float] = BASE
    L1: float = L1
    L2: float = L2

    def fk_scalar(self, theta1_deg: float, theta2_deg: float):
        """Return elbow and tip position for a single configuration."""
        t1 = np.deg2rad(normalize_angle_deg(theta1_deg))
        t2 = np.deg2rad(normalize_angle_deg(theta2_deg))
        elbow = (
            self.base[0] + self.L1 * np.cos(t1),
            self.base[1] + self.L1 * np.sin(t1),
        )
        tip = (
            elbow[0] + self.L2 * np.cos(t1 + t2),
            elbow[1] + self.L2 * np.sin(t1 + t2),
        )
        return elbow, tip

    def fk_vectorized(self, theta1_deg, theta2_deg):
        """Vectorized FK for arrays of theta1/theta2. Works for 2D mesh arrays."""
        theta1 = np.asarray(theta1_deg, dtype=float)
        theta2 = np.asarray(theta2_deg, dtype=float)
        t1 = np.deg2rad(normalize_angle_deg(theta1))
        t2 = np.deg2rad(normalize_angle_deg(theta2))
        elbow_x = self.base[0] + self.L1 * np.cos(t1)
        elbow_y = self.base[1] + self.L1 * np.sin(t1)
        tip_x = elbow_x + self.L2 * np.cos(t1 + t2)
        tip_y = elbow_y + self.L2 * np.sin(t1 + t2)
        return np.stack([elbow_x, elbow_y], axis=-1), np.stack([tip_x, tip_y], axis=-1)


ARM_DEFAULT = Arm()
