"""Regression plot script."""

import matplotlib.pyplot as plt
import matplotlib.patheffects as mpath
import numpy as np
from sklearn.datasets import make_circles


import utils


def main():

    # Arguments
    args = utils.parse_args()

    # Seed randomness
    np.random.seed(42)

    # Style
    utils.xkcd_style()

    # Create accuracy matrix
    features, labels = make_circles(n_samples=100, noise=0.1, factor=0.1)

    # Initialize plot
    fig = plt.figure(figsize=(3.2, 6))
    ax = [
        fig.add_subplot(2, 1, 1),
        fig.add_subplot(2, 1, 2, projection="3d"),
    ]

    # Data samples
    utils.samples(ax[0], features, labels)

    # Regression line
    for cls in np.unique(labels):
        ax[1].plot(
            features[labels == cls, 0],
            features[labels == cls, 1],
            np.exp(
                -5
                * (
                    (
                        features[labels == cls, 0] ** 2
                        + features[labels == cls, 1] ** 2
                    )
                )
            ),
            linestyle="",
            marker="o",
            color=f"C{cls}",
            mec="k",
            ms=5,
            zorder=10 * cls,
        )

    # Remove labels
    ax[0].set_xlabel("feature x₁")
    ax[0].set_ylabel("feature x₂")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # Loss
    ax[1].set_xlabel("x₁", labelpad=-8)
    ax[1].set_ylabel("x₂", labelpad=-8)
    ax[1].set_zlabel("exp(-r²)", labelpad=-8)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_zticks([])
    ax[1].set_facecolor("none")
    ax[1].view_init(elev=30, azim=45)
    fig.tight_layout(pad=1, h_pad=0.4)

    # Plane that splits
    xx = np.linspace(-1.5, 1.5, 100)
    YY, XX = np.meshgrid(xx, xx)
    Z = 0.5 * np.ones(XX.shape)
    ax[1].plot_surface(
        XX,
        YY,
        Z,
        color="C3",
        zorder=0.5,
        alpha=0.3,
        edgecolor="C3",
        linewidth=0,
    )
    # Decision boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    x_boundary = np.cos(theta) / np.sqrt(2)
    y_boundary = np.sin(theta) / np.sqrt(2)
    z_boundary = 0.5 * np.ones(x_boundary.shape)
    ax[1].plot3D(
        x_boundary,
        y_boundary,
        z_boundary,
        color="C3",
        linewidth=1.5,
        zorder=0.5,
    )

    # Decision boundary in the original space
    circle = plt.Circle(
        (0, 0),
        1 / np.sqrt(2),
        color="C3",
        fill=False,
        linewidth=1.5,
    )
    ax[0].add_artist(circle)

    # Arrow top axis to bottom axis
    arrowprops = dict(
        arrowstyle="<-",
        connectionstyle="arc3,rad=-0.2",
        shrinkA=5,
        shrinkB=10,
        linewidth=1.5,
    )
    annotate = dict(
        arrowprops=arrowprops,
        ha="center",
        va="center",
        xycoords="axes fraction",
        fontweight="normal",
    )
    ax[0].annotate("kernel", xy=(0, 0), xytext=(0, -0.35), **annotate)

    # Arrow bottom axis to top axis
    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="arc3,rad=0.2",
        shrinkA=5,
        shrinkB=5,
        linewidth=1.5,
    )
    annotate = dict(
        arrowprops=arrowprops,
        ha="center",
        va="center",
        xycoords="axes fraction",
        fontweight="normal",
    )
    ax[1].annotate("back", xy=(1, 1.2), xytext=(1, 0.83), **annotate)

    # Save figure
    fig.savefig(f"{args.output_dir}/kernel_trick.png")


if __name__ == "__main__":
    main()
