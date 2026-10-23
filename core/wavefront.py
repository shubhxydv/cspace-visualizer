from __future__ import annotations

import numpy as np
from scipy import ndimage

from config import GRID_N, BASE, L1, L2
from core.arm import Arm
from core.collision import collides_vectorized


class CSpace:
    """A torus-aware occupancy grid for the 2-link manipulator configuration space."""

    def __init__(self, N=GRID_N, arm=None, obstacles=None):
        self.N = int(N)
        self.deg_per_cell = 360.0 / self.N
        theta = np.arange(self.N) * self.deg_per_cell
        T1, T2 = np.meshgrid(theta, theta, indexing="ij")
        self.arm = arm or Arm(BASE, L1, L2)
        self.obstacles = list(obstacles or [])
        self.occ = collides_vectorized(T1, T2, self.arm, self.obstacles)

    def angle_to_index(self, deg):
        return int(np.rint(deg / self.deg_per_cell) % self.N)

    def index_to_angle(self, idx):
        return float((int(idx) % self.N) * self.deg_per_cell)

    def neighbours(self, i, j, connectivity=8):
        out = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                if connectivity == 4 and abs(di) + abs(dj) != 1:
                    continue
                if connectivity == 8:
                    out.append(((i + di) % self.N, (j + dj) % self.N))
                else:
                    if abs(di) + abs(dj) == 1:
                        out.append(((i + di) % self.N, (j + dj) % self.N))
        return out

    def torus_distance_transform(self):
        """Distance to obstacle for free cells using a 3x3 torus tile."""
        occ = self.occ.astype(np.uint8)
        free = (~occ).astype(bool)
        tiled = np.tile(free, (3, 3))
        # distance from each free cell to nearest obstacle, in the tiled map
        dist_tiled = ndimage.distance_transform_edt(~tiled)
        center = dist_tiled[1 * self.N:2 * self.N, 1 * self.N:2 * self.N]
        return center.astype(float)


def build_cspace(occ, arm=None, obstacles=None):
    """Compatibility wrapper returning a CSpace object for a known occupancy grid."""
    obj = CSpace(N=occ.shape[0], arm=arm, obstacles=obstacles)
    obj.occ = np.asarray(occ, dtype=bool)
    return obj


def torus_distance_between(i1, j1, i2, j2, N):
    """Torus-aware Euclidean distance between two cells in the grid."""
    di = min(abs(i1 - i2), N - abs(i1 - i2))
    dj = min(abs(j1 - j2), N - abs(j1 - j2))
    return float(np.hypot(di, dj))
