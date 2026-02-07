import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import utils

np.random.seed(42)


def main():

    # Arguments
    args = utils.parse_args()

    # Figure
    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw={"projection": "3d"})
    fig.set_facecolor("none")

    # Remove ax background white
    ax.set_facecolor("none")

    # Data samples
    x = np.random.normal(scale=0.6, size=(100, 3))

    # Sort x with respect to distance to origin
    x = x[np.argsort(np.linalg.norm(x, axis=1))]
    for sample in x:

        # If point in sphere, color is C0 elsea C1
        if np.linalg.norm(sample) < 1:
            color = "C0"
        else:
            color = "C1"

        ax.plot(
            sample[0],
            sample[1],
            sample[2],
            linestyle="",
            marker="o",
            color=color,
            mec="k",
            ms=5,
        )

    # Equal aspect ratio    ax.set_box_aspect([1, 1, 1])
    ax.set_box_aspect([1, 1, 1])

    # Remove ticks
    ax.set(xticks=[], yticks=[], zticks=[])
    ax.set_xlabel("x", labelpad=-10)
    ax.set_ylabel("y", labelpad=-10)
    ax.set_zlabel("z", labelpad=-10)
    ax.view_init(elev=35, azim=45)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

    # Save
    # fig.tight_layout(pad=0)
    fig.savefig(
        args.output_dir / "notebook_3.png", bbox_inches="tight", pad_inches=0.2
    )


if __name__ == "__main__":
    main()
