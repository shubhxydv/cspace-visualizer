from __future__ import annotations

import numpy as np

from config import BASE, L1, L2, GRID_N, normalize_angle_deg
from core.arm import Arm
from core.collision import collides_vectorized


class CSpace:
    """Configuration-space occupancy grid and torus-aware helper methods."""

    def __init__(self, N: int = GRID_N, arm=None, obstacles=None):
        self.N = int(N)
        if self.N <= 0:
            raise ValueError("N must be positive")
        self.deg_per_cell = 360.0 / self.N
        self.arm = arm if arm is not None else Arm(BASE, L1, L2)
        self.obstacles = list(obstacles or [])

        theta = np.arange(self.N) * self.deg_per_cell
        t1, t2 = np.meshgrid(theta, theta, indexing="ij")
        self.occ = collides_vectorized(t1, t2, self.arm, self.obstacles)

    def angle_to_index(self, deg: float) -> int:
        angle = normalize_angle_deg(deg)
        return int(np.rint((angle / 360.0) * self.N)) % self.N

    def index_to_angle(self, idx: int) -> float:
        return float((int(idx) % self.N) * self.deg_per_cell)

    def neighbours(self, i: int, j: int, connectivity: int = 8):
        out = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                if connectivity == 4 and abs(di) + abs(dj) != 1:
                    continue
                ni = (i + di) % self.N
                nj = (j + dj) % self.N
                if connectivity == 8 or abs(di) + abs(dj) == 1:
                    out.append((ni, nj))
        return out


def build_cspace(occ, arm=None, obstacles=None):
    cspace = CSpace(N=occ.shape[0], arm=arm, obstacles=obstacles)
    cspace.occ = np.asarray(occ, dtype=bool)
    return cspace


def empty_grid(N: int = GRID_N):
    return np.zeros((N, N), dtype=bool)
