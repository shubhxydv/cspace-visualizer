from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from config import BASE, L1, L2, GRID_N, COLORS, normalize_angle_deg
from core.arm import Arm
from core.collision import arm_collides
from core.cspace import CSpace


class App:
    """Minimal Phase 2 app with workspace and C-space views plus linked sliders."""

    def __init__(self, N: int = GRID_N, scenario=None):
        self.N = int(N)
        self.arm = Arm(BASE, L1, L2)
        self.obstacles = list(scenario.obstacles if scenario is not None else [])
        self.cspace = CSpace(N=self.N, arm=self.arm, obstacles=self.obstacles)
        self.theta1 = 0.0
        self.theta2 = 0.0

        self.fig, (self.ax_ws, self.ax_cspace) = plt.subplots(1, 2, figsize=(10, 5))
        self.ax_ws.set_aspect("equal")
        self.ax_cspace.set_aspect("equal")
        self.ax_ws.set_xlim(-(L1 + L2 + 0.5), L1 + L2 + 0.5)
        self.ax_ws.set_ylim(-(L1 + L2 + 0.5), L1 + L2 + 0.5)
        self.ax_cspace.set_xlim(0, 360)
        self.ax_cspace.set_ylim(0, 360)
        self.ax_cspace.invert_yaxis()

        self.arm_line, = self.ax_ws.plot([], [], lw=4, color="black")
        self.dot, = self.ax_cspace.plot([], [], "o", color="black")
        self.status = self.ax_ws.text(0.02, 0.98, "", transform=self.ax_ws.transAxes, va="top")

        self.slider_theta1 = Slider(self.fig.add_axes([0.1, 0.05, 0.35, 0.03]), "θ1", 0, 359, valinit=0)
        self.slider_theta2 = Slider(self.fig.add_axes([0.55, 0.05, 0.35, 0.03]), "θ2", 0, 359, valinit=0)
        self.slider_theta1.on_changed(self._on_theta1_change)
        self.slider_theta2.on_changed(self._on_theta2_change)

        self._redraw()

    def _current_pose(self):
        return self.slider_theta1.val, self.slider_theta2.val

    def _on_theta1_change(self, value):
        self.theta1 = float(value)
        self._redraw()

    def _on_theta2_change(self, value):
        self.theta2 = float(value)
        self._redraw()

    def _draw_workspace_arm(self):
        t1 = np.deg2rad(normalize_angle_deg(self.theta1))
        t2 = np.deg2rad(normalize_angle_deg(self.theta2))
        elbow = (L1 * np.cos(t1), L1 * np.sin(t1))
        tip = (elbow[0] + L2 * np.cos(t1 + t2), elbow[1] + L2 * np.sin(t1 + t2))
        x = [0.0, elbow[0], tip[0]]
        y = [0.0, elbow[1], tip[1]]
        self.arm_line.set_data(x, y)
        self.ax_ws.set_title("Workspace")

    def _draw_cspace_dot(self):
        dot_x = normalize_angle_deg(self.theta1)
        dot_y = normalize_angle_deg(self.theta2)
        self.dot.set_data([dot_x], [dot_y])
        self.ax_cspace.set_title("Configuration space (θ1, θ2)")

    def _redraw(self):
        self.theta1 = float(normalize_angle_deg(self.slider_theta1.val))
        self.theta2 = float(normalize_angle_deg(self.slider_theta2.val))
        self._draw_workspace_arm()
        self._draw_cspace_dot()
        collision = arm_collides(self.theta1, self.theta2, self.arm, self.obstacles)
        status = "COLLISION" if collision else "FREE"
        self.status.set_text(f"θ1={self.theta1:.1f}  θ2={self.theta2:.1f}  tip=({self.theta1:.1f},{self.theta2:.1f})  {status}")
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    App().show()
