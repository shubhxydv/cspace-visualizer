from __future__ import annotations

from collections import deque

import numpy as np


def neighbours_torus(i, j, N, connectivity=8):
    out = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            if connectivity == 4 and abs(di) + abs(dj) != 1:
                continue
            ni = (i + di) % N
            nj = (j + dj) % N
            if connectivity == 8:
                out.append((ni, nj))
            elif abs(di) + abs(dj) == 1:
                out.append((ni, nj))
    return out


def wavefront_grid(occ, goal, connectivity=8):
    """BFS flood fill on a torus, with free cells = False in occ."""
    occ = np.asarray(occ, dtype=bool)
    N = occ.shape[0]
    wave = np.full((N, N), -1, dtype=np.int32)
    if not (0 <= goal[0] < N and 0 <= goal[1] < N):
        raise ValueError("goal out of bounds")
    if occ[goal]:
        return wave
    q = deque([tuple(goal)])
    wave[goal] = 0
    while q:
        i, j = q.popleft()
        for ni, nj in neighbours_torus(i, j, N, connectivity):
            if occ[ni, nj]:
                continue
            if wave[ni, nj] != -1:
                continue
            wave[ni, nj] = wave[i, j] + 1
            q.append((ni, nj))
    return wave


def wavefront_path(occ, start, goal, connectivity=8):
    """Return a path from start to goal using the wave-front grid."""
    occ = np.asarray(occ, dtype=bool)
    N = occ.shape[0]
    start = tuple(start)
    goal = tuple(goal)
    if occ[start]:
        return [], "NO_PATH"
    if occ[goal]:
        return [], "NO_PATH"
    wave = wavefront_grid(occ, goal, connectivity)
    if wave[start] == -1:
        return [], "NO_PATH"
    if start == goal:
        return [start], "PATH_FOUND"
    path = [start]
    cur = start
    while cur != goal:
        best = None
        best_val = None
        for ni, nj in neighbours_torus(cur[0], cur[1], N, connectivity):
            if occ[ni, nj]:
                continue
            value = int(wave[ni, nj])
            if value == -1:
                continue
            if value < int(wave[cur]):
                if best_val is None or value < best_val:
                    best = (ni, nj)
                    best_val = value
        if best is None:
            return path, "NO_PATH"
        path.append(best)
        cur = best
    return path, "PATH_FOUND"


def escape_path(occ, from_cell, goal, connectivity=8):
    """Wave-front route from a stranded cell to the goal."""
    return wavefront_path(occ, from_cell, goal, connectivity)
