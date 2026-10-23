from __future__ import annotations

import numpy as np
from scipy import ndimage

from config import EPS, ETA, ZETA, RHO0


def torus_distance_to_goal(goal, shape):
    """Torus-aware Euclidean distance to the goal from every cell."""
    N = shape[0]
    i = np.arange(N)
    I, J = np.meshgrid(i, i, indexing="ij")
    di = np.minimum(np.abs(I - goal[0]), N - np.abs(I - goal[0]))
    dj = np.minimum(np.abs(J - goal[1]), N - np.abs(J - goal[1]))
    return np.hypot(di, dj)


def torus_distance_to_obstacles(occ):
    """Distance from each free cell to nearest occupied cell, accounting for torus seams."""
    occ = np.asarray(occ, dtype=bool)
    free = ~occ
    tiled = np.tile(free, (3, 3))
    dist_tiled = ndimage.distance_transform_edt(~tiled)
    center = dist_tiled[occ.shape[0]:2 * occ.shape[0], occ.shape[1]:2 * occ.shape[1]]
    out = np.zeros_like(occ, dtype=float)
    out[~occ] = center[~occ]
    return out


def potential_grid(occ, goal, zeta=ZETA, eta=ETA, rho0=RHO0):
    """Compute the scalar potential field over the whole torus grid."""
    occ = np.asarray(occ, dtype=bool)
    if occ.shape[0] != occ.shape[1]:
        raise ValueError("Occupancy grid must be square")
    N = occ.shape[0]
    d_goal = torus_distance_to_goal(goal, occ.shape)
    rho = torus_distance_to_obstacles(occ)
    U_att = 0.5 * zeta * d_goal ** 2
    U_rep = np.zeros_like(U_att, dtype=float)
    mask = (rho > 0) & (rho <= rho0) & (~occ)
    U_rep[mask] = 0.5 * eta * (1.0 / rho[mask] - 1.0 / rho0) ** 2
    U_rep[occ] = 1e12
    return U_att + U_rep


def valid_descent_moves(occ, cur, connectivity=8):
    """Return free torus neighbours of cur in deterministic order."""
    N = occ.shape[0]
    out = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            if connectivity == 4 and abs(di) + abs(dj) != 1:
                continue
            ni = (cur[0] + di) % N
            nj = (cur[1] + dj) % N
            if connectivity == 8 or abs(di) + abs(dj) == 1:
                if not occ[ni, nj]:
                    out.append((ni, nj))
    return out


def gradient_descent(start, goal, occ, max_steps=None, zeta=ZETA, eta=ETA, rho0=RHO0):
    """Follow the local gradient of the potential field until a stop condition is reached."""
    occ = np.asarray(occ, dtype=bool)
    start = tuple(start)
    goal = tuple(goal)
    if max_steps is None:
        max_steps = 5 * occ.shape[0]
    if occ[start]:
        return [start], "STUCK_LOCAL_MINIMUM", start
    U = potential_grid(occ, goal, zeta=zeta, eta=eta, rho0=rho0)
    path = [start]
    cur = start
    for step in range(max_steps):
        if torus_distance_to_goal(goal, occ.shape)[cur] <= 1.0:
            return path, "GOAL_REACHED", None
        moves = valid_descent_moves(occ, cur, connectivity=8)
        if not moves:
            return path, "STUCK_LOCAL_MINIMUM", cur
        better = [(n, U[n]) for n in moves if U[n] < U[cur] - EPS]
        if not better:
            return path, "STUCK_LOCAL_MINIMUM", cur
        next_cell = min(better, key=lambda item: (item[1], item[0][0], item[0][1]))[0]
        path.append(next_cell)
        cur = next_cell
    return path, "MAX_STEPS", cur
