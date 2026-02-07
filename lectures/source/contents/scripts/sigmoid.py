"""Regression plot script."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA, KernelPCA


import utils


def main():

    # Arguments
    args = utils.parse_args()

    # Seed randomness
    np.random.seed(42)

    # Style
    utils.xkcd_style()

    # Initialize plot
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Sigmoid function
    z = np.linspace(-10, 10, 100)
    sigmoid = 1 / (1 + np.exp(-z))
    ax.plot(z, sigmoid)

    # Labels
    ax.set(
        xlabel="input z",
        ylabel="sigmoid(z)",
        xlim=(-10, 10),
        ylim=(-0.1, 1.1),
        xticks=[-10, -5, 0, 5, 10],
        yticks=[0, 0.5, 1],
    )
    ax.text(0.02, 0.15, "inactive", transform=ax.transAxes)
    ax.text(0.95, 0.8, "active", transform=ax.transAxes, ha="right")

    # Save figure
    fig.tight_layout(h_pad=0.2)
    fig.savefig(f"{args.output_dir}/sigmoid.png")


if __name__ == "__main__":
    main()
