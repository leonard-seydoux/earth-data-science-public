"""Matrix eigenvector visualization."""

import numpy as np

import utils


def main():

    # Arguments
    args = utils.parse_args()

    # Set up figure
    fig, ax = utils.square_canvas()

    # Data and model
    A = np.array([[1, 0.5], [0.5, 1]]) * 1.1

    # Annotate plenty of vectors
    for i, (x, y) in enumerate(zip([0.1, 0.5, 0.5], [0.5, 0.5, 0.1])):
        x_vec = np.array([x, y])
        Ax_vec = A @ x_vec
        ax.annotate(
            "",
            xy=Ax_vec,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=f"C{i + 1}"),
            zorder=0,
        )
        ax.annotate(
            "",
            xy=x_vec,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=f"C{i + 1}"),
            zorder=0,
        )
        angle_with_x = np.arctan2(y, x)
        angle_with_Ax = np.arctan2(Ax_vec[1], Ax_vec[0])
        ax.text(
            *Ax_vec,
            "Ax = dx" if i == 1 else "Ax",
            color=f"C{i + 1}",
            ha="left",
            rotation=np.degrees(angle_with_Ax),
            va="baseline",
        )
        ax.text(
            *x_vec,
            "x",
            color=f"C{i + 1}",
            ha="left",
            rotation=np.degrees(angle_with_x),
            va="center",
        )

    # Labels and limits
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.text(-0.03, -0.01, "0", ha="right", va="top")
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_aspect("equal")

    # Save figure
    fig.savefig(args.output_dir / "eigenvectors.png")


if __name__ == "__main__":
    main()
