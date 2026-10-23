from __future__ import annotations

from dataclasses import dataclass

from config import BASE, L1, L2
from core.arm import Arm, Circle, Rectangle


@dataclass(frozen=True)
class World:
    name: str
    arm: Arm
    obstacles: tuple
    start: tuple[int, int]
    goal: tuple[int, int]


def simple_world():
    return World(
        name="Simple",
        arm=Arm(BASE, L1, L2),
        obstacles=(
            Circle(1.0, 0.9, 0.25),
            Rectangle(-1.2, -0.6, 0.5, 0.4),
        ),
        start=(10, 10),
        goal=(300, 260),
    )


def trap_world():
    return World(
        name="Trap (local minimum)",
        arm=Arm(BASE, L1, L2),
        obstacles=(
            Circle(0.2, 0.5, 0.4),
            Circle(-0.3, -0.5, 0.4),
            Rectangle(-0.9, -0.4, 0.9, 0.5),
        ),
        start=(30, 40),
        goal=(300, 300),
    )


def wrap_world():
    return World(
        name="Wrap-around path",
        arm=Arm(BASE, L1, L2),
        obstacles=(
            Circle(0.8, 0.4, 0.35),
            Rectangle(-1.1, -0.8, 0.6, 0.8),
        ),
        start=(330, 170),
        goal=(20, 170),
    )


def narrow_world():
    return World(
        name="Narrow passage",
        arm=Arm(BASE, L1, L2),
        obstacles=(
            Rectangle(-0.2, -1.2, 0.5, 2.0),
            Rectangle(0.8, -1.2, 0.5, 2.0),
        ),
        start=(80, 60),
        goal=(280, 260),
    )


def no_path_world():
    return World(
        name="No path",
        arm=Arm(BASE, L1, L2),
        obstacles=(
            Circle(0.0, 0.0, 1.1),
            Rectangle(-1.5, -0.4, 3.2, 0.8),
        ),
        start=(40, 40),
        goal=(320, 300),
    )


SCENARIOS = {
    "Simple": simple_world,
    "Trap (local minimum)": trap_world,
    "Wrap-around path": wrap_world,
    "Narrow passage": narrow_world,
    "No path": no_path_world,
}


def scenario(name):
    return SCENARIOS[name]()
