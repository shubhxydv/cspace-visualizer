from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class App:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title("Configuration-space visualizer")
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect("equal")

    def show(self):
        plt.show()


def main() -> None:
    app = App()
    app.show()


if __name__ == "__main__":
    main()
