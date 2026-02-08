"""K-means clustering algorithm visualization in GIF."""

import matplotlib.pyplot as plt
import numpy as np

import utils


def main():

    # Arguments
    args = utils.parse_args()

    # Set up figure
    fig, ax = utils.square_canvas()

    # Data and model
    A = np.array([[1, 0.5], [0.5, 1]]) * 1.2
    v = np.linalg.eig(A)[1][:, 0]

    # Annotate plenty of vectors
    for x, y in zip([0.1, 0.5, 0.5], [0.5, 0.5, 0.1]):
        x_vec = np.array([x, y])
        Ax_vec = A @ x_vec
        color = "C1" if x == y else "C0"
        ax.annotate(
            "",
            xy=Ax_vec,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color),
            zorder=0,
        )
        ax.annotate(
            "",
            xy=x_vec,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, alpha=0.5),
            zorder=0,
        )

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

    # Labels
    ax.text(0.2, 0.2, "x", color="C1", ha="left", va="top")
    ax.text(0.6, 0.6, "Ax = dx", color="C1", ha="left", va="top")

    ax.set_axis_off()
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")

    fig.savefig(args.output_dir / "eigenvectors.png")


if __name__ == "__main__":
    main()
