from __future__ import annotations

import random

from config import GRID_N
from core.potential_field import gradient_descent
from core.wavefront import wavefront_grid, wavefront_path


def find_trap_candidates():
    """Seeded search for start/goal pairs where the potential field gets trapped."""
    rng = random.Random(42)
    candidates = []
    for _ in range(200):
        start = (rng.randrange(GRID_N), rng.randrange(GRID_N))
        goal = (rng.randrange(GRID_N), rng.randrange(GRID_N))
        occ = [[False for _ in range(GRID_N)] for _ in range(GRID_N)]
        # a simple obstacle wall chosen to encourage local minima without full complexity
        for i in range(GRID_N):
            for j in range(GRID_N):
                if 120 <= i <= 140 and 80 <= j <= 130:
                    occ[i][j] = True
        path, status, stuck = gradient_descent(start, goal, occ)
        if status == "STUCK_LOCAL_MINIMUM":
            wave = wavefront_grid(occ, goal)
            if wave[start] != -1:
                candidates.append((start, goal, stuck))
    return candidates[:5]


if __name__ == "__main__":
    for entry in find_trap_candidates():
        print(entry)
