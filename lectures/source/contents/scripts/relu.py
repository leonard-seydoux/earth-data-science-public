"""Regression plot script."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA, KernelPCA


import plot


def main():

    # Seed randomness
    np.random.seed(42)

    # Style
    plot.xkcd_style()

    # Initialize plot
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Relu function
    z = np.linspace(-10, 10, 100)
    relu = np.maximum(0, z)
    ax.plot(z, relu)

    # Labels
    ax.set(
        xlabel="input z",
        ylabel="relu(z)",
        xlim=(-10, 10),
        ylim=(-1, 10.5),
        xticks=[-10, -5, 0, 5, 10],
        yticks=[0, 5, 10],
    )
    ax.text(0.05, 0.15, "inactive", transform=ax.transAxes)

    # Save figure
    fig.tight_layout(h_pad=0.2)
    fig.savefig("contents/figures/relu.png")


if __name__ == "__main__":
    main()
