# Configuration-space theory

A 2-link planar arm is described by two joint angles, so every configuration is a point in a 2D map: the C-space. A physical obstacle in the room becomes a set of forbidden joint-angle pairs, called C-obstacles.

The arm is a pair of links anchored at the base. The forward kinematics are computed from the joint angles and the link lengths. In configuration space, a collision means the arm touches an obstacle, and every (θ1, θ2) pair is either free or blocked.

Because angles wrap modulo 360°, the map is a torus. Moving across the left edge continues on the right edge, and the top edge connects to the bottom. That matters for wave-front planning and distance calculations, and it is why C-space plots must consider seam handling.

The potential field planner combines an attractive term that pulls the robot toward the goal and a repulsive term that pushes it away from obstacles. It is simple but can stop in a pocket: a local minimum. The wave-front planner avoids that by propagating distances from the goal outward across free cells and then descending from the start.

The grid planner is resolution complete: if a path exists on the chosen discretization, it will be found; if the resolution is too coarse, some narrow paths can be missed.
