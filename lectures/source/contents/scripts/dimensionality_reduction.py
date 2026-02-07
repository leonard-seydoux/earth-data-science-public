import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

import plot


def make_reduction(x, y):
    """Generate synthetic classification data."""
    model = PCA(n_components=1)
    reduced = model.fit_transform(x)
    return reduced, model


def main():

    # Customize matplotlib style
    plot.xkcd_style()

    # Create accuracy matrix
    np.random.seed(42)
    x1 = np.random.uniform(-3, 3, size=(30,))
    x2 = x1 + np.random.normal(0, 0.7, size=x1.shape)
    X = np.vstack([x1, x2]).T

    # Initialize plot
    fig, ax = plt.subplots(
        ncols=2,
        figsize=(3.8, 3),
        gridspec_kw={"wspace": 0.5, "width_ratios": [1, 0.0001]},
    )

    # Show scatter plot
    plot.samples(ax[0], X)

    # Dimensionality reduction
    reduced, model = make_reduction(X, np.zeros(X.shape[0]))

    ax[1].plot(
        np.zeros(reduced.shape[0]),
        reduced,
        "o",
        color="C1",
        mec="k",
        clip_on=False,
        zorder=10,
    )

    # Labels
    ax[0].set_xlabel("feature x₁")
    ax[0].set_ylabel("feature x₂")
    ax[1].set_ylabel("latent variable z")
    ax[1].spines[:].set_visible(False)
    ax[1].spines["left"].set_visible(True)
    ax[1].set_xlim(-0.5, 0.5)

    # Annotate
    z_continuous = np.linspace(reduced.min(), reduced.max(), 100)
    ax[0].plot(
        *model.inverse_transform(z_continuous[:, np.newaxis]).T,
        color=plot.brighter("C5", -1),
        lw=2,
        zorder=0
    )

    # Arrow between axes
    ax[0].annotate(
        "",
        xy=(1.12, 0.5),
        xytext=(0.7, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->", lw=1.5, color="k", connectionstyle="arc3,rad=0.0"
        ),
    )

    # Collapse all samples to their projections
    for i in range(X.shape[0]):
        x_orig = X[i : i + 1]
        x_proj = model.inverse_transform(reduced[i : i + 1])
        ax[0].annotate(
            "",
            xy=(x_orig[0, 0], x_orig[0, 1]),
            xytext=(x_proj[0, 0], x_proj[0, 1]),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-",
                lw=0.8,
                color=plot.brighter("C5", -1),
                connectionstyle="arc3,rad=0.0",
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=0,
        )

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    ax[0].set_aspect("equal")
    lim_min = min(ax[0].get_xlim()[0], ax[0].get_ylim()[0])
    lim_max = max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])
    ax[0].set_xlim(lim_min, lim_max)
    ax[0].set_ylim(lim_min, lim_max)
    # ax[0].set_ylim(ax[0].get_xlim())

    # Save figure
    fig.savefig("contents/figures/dimensionality_reduction.png")


if __name__ == "__main__":
    main()
