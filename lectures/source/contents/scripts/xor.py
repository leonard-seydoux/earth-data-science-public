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

    # Generate Random XOR Data
    n_samples = 200
    X = np.random.rand(n_samples, 2) * 2 - 1
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

    # Relu function
    utils.samples(ax=ax, X=X, y=y)

    # Labels
    ax.set(
        xlabel="feature x₁",
        ylabel="feature x₂",
        xticks=[-1, 0, 1],
        yticks=[-1, 0, 1],
    )

    # Save figure
    fig.tight_layout(h_pad=0.2)
    fig.savefig(f"{args.output_dir}/xor.png")


if __name__ == "__main__":
    main()
