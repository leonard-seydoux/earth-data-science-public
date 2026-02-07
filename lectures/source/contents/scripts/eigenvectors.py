"""K-means clustering algorithm visualization in GIF."""

import matplotlib.pyplot as plt
import numpy as np

import utils

utils.xkcd_style()


def main():

    # Arguments
    args = utils.parse_args()

    utils.xkcd_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Arrow axes
    ax.annotate(
        "",
        xy=(1, 0),
        xytext=(-0.05, 0),
        arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(0, 1),
        xytext=(0, -0.05),
        arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.text(1.05, 0, "x₁", ha="left", va="center", weight="normal")
    ax.text(0, 1.05, "x₂", ha="center", va="bottom", weight="normal")
    ax.text(-0.03, -0.01, "0", ha="right", va="top", weight="normal")

    # X vector
    x = np.array([0.4, 0.4])
    ax.annotate(
        "",
        xy=x,
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="C1", linewidth=1.5),
        zorder=1,
    )

    Ax = x * 1.9
    ax.annotate(
        "",
        xy=Ax,
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="C2", linewidth=1.5),
        zorder=0,
    )

    # Labels
    ax.text(0.2, 0.2, "x", color="C1", ha="left", va="top")
    ax.text(0.6, 0.6, "Ax = dx", color="C2", ha="left", va="top")

    ax.set_axis_off()
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(f"{args.output_dir}/eigenvectors.png")


if __name__ == "__main__":
    main()
