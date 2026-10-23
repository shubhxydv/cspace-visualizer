from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import COLORS


def export_all(output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], color=COLORS["path_pf"])
    plt.title("workspace")
    plt.savefig(os.path.join(output_dir, "workspace.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], color=COLORS["collision"])
    plt.title("cspace")
    plt.savefig(os.path.join(output_dir, "cspace.png"), dpi=200)
    plt.close()

    print(f"Exported files to {output_dir}")


if __name__ == "__main__":
    export_all()
