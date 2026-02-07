import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from sklearn import datasets

import plot


def main():

    # Customize matplotlib style
    plot.xkcd_style()

    # Create representation matters matrix
    fig, ax = plt.subplots(2, 1, figsize=(2.5, 6), gridspec_kw={"hspace": 0.4})

    # Data
    X, y = datasets.make_circles(
        n_samples=100, factor=0.5, noise=0.05, random_state=42
    )

    # Plot original data (cmap C0 and C1)
    plot.samples(ax[0], X, y)
    ax[0].set_xlabel("feature x₁")
    ax[0].set_ylabel("feature x₂")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # Plot transformed data
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    theta = np.arctan2(X[:, 1], X[:, 0])
    X_transformed = np.column_stack((theta, r))
    plot.samples(ax[1], X_transformed, y)
    ax[1].set_xlabel("a = atan(x₂, x₁)")
    ax[1].set_ylabel("r² = x₁² + x₂²")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    # Save
    fig.savefig(
        "contents/figures/representation_matters.png", bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
