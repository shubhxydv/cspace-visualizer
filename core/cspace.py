from __future__ import annotations

import numpy as np

from config import BASE, L1, L2, GRID_N
from core.arm import Arm


class CSpace:
    """Simple C-space object with helpers used by the planner and UI layers."""

    def __init__(self, N: int = GRID_N, arm=None, obstacles=None):
        self.N = int(N)
        self.deg_per_cell = 360.0 / self.N
        self.arm = arm or Arm(BASE, L1, L2)
        self.obstacles = list(obstacles or [])
        theta = np.arange(self.N) * self.deg_per_cell
        T1, T2 = np.meshgrid(theta, theta, indexing="ij")
        from core.collision import collides_vectorized

        self.occ = collides_vectorized(T1, T2, self.arm, self.obstacles)

    def angle_to_index(self, deg: float) -> int:
        return int(np.rint((float(deg) % 360.0) * self.N / 360.0)) % self.N

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


def build_cspace(occ):
    c = CSpace(N=occ.shape[0])
    c.occ = np.asarray(occ, dtype=bool)
    return c
