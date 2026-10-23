from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

L1 = 1.0
L2 = 0.8
BASE = (0.0, 0.0)
GRID_N = 360
CONNECTIVITY = 8
MAX_STEPS = 5 * GRID_N
ZETA = 1.0
ETA = 2000.0
RHO0 = 25
WORKSPACE_LIMIT = L1 + L2 + 0.3
COLORS = {
    "free": "white",
    "collision": "#d62728",
    "path_pf": "#ff7f0e",
    "path_wave": "#2ca02c",
    "escape": "#9467bd",
    "start": "#2ca02c",
    "goal": "#1f77b4",
}
ANIMATION_FPS = 30
EPS = 1e-9


def normalize_angle_deg(angle_deg: float) -> float:
    """Convert any degree input into the canonical interval [0, 360)."""
    value = float(angle_deg) % 360.0
    if math.isclose(value, 360.0, abs_tol=1e-12):
        return 0.0
    return value


def wrap_360(angle_deg: float) -> float:
    return normalize_angle_deg(angle_deg)


def angle_to_index(angle_deg: float, n: int = GRID_N) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    return int(round(normalize_angle_deg(angle_deg) * n / 360.0)) % n


def index_to_angle(index: int, n: int = GRID_N) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    return float((int(index) % n) * 360.0 / n)


def torus_distance_axis(a: int, b: int, n: int) -> int:
    d = abs(int(a) - int(b))
    return int(min(d, n - d))


def torus_neighbours(i: int, j: int, n: int, connectivity: int = 8):
    neighbours = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            if connectivity == 4 and abs(di) + abs(dj) != 1:
                continue
            ni = (i + di) % n
            nj = (j + dj) % n
            if connectivity == 8 or abs(di) + abs(dj) == 1:
                neighbours.append((ni, nj))
    return neighbours


@dataclass(frozen=True)
class Arm:
    base: Tuple[float, float] = BASE
    L1: float = L1
    L2: float = L2

    def fk_scalar(self, theta1_deg: float, theta2_deg: float):
        t1 = math.radians(normalize_angle_deg(theta1_deg))
        t2 = math.radians(normalize_angle_deg(theta2_deg))
        elbow = (
            self.base[0] + self.L1 * math.cos(t1),
            self.base[1] + self.L1 * math.sin(t1),
        )
        tip = (
            elbow[0] + self.L2 * math.cos(t1 + t2),
            elbow[1] + self.L2 * math.sin(t1 + t2),
        )
        return elbow, tip

    def fk_vectorized(self, theta1_deg, theta2_deg):
        theta1 = np.asarray(theta1_deg, dtype=float)
        theta2 = np.asarray(theta2_deg, dtype=float)
        t1 = np.deg2rad(np.mod(theta1, 360.0))
        t2 = np.deg2rad(np.mod(theta2, 360.0))
        elbow_x = self.base[0] + self.L1 * np.cos(t1)
        elbow_y = self.base[1] + self.L1 * np.sin(t1)
        tip_x = elbow_x + self.L2 * np.cos(t1 + t2)
        tip_y = elbow_y + self.L2 * np.sin(t1 + t2)
        elbow = np.stack([elbow_x, elbow_y], axis=-1)
        tip = np.stack([tip_x, tip_y], axis=-1)
        return elbow, tip


ARM_DEFAULT = Arm()
