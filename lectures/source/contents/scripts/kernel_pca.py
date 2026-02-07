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

    # Create accuracy matrix
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.1)

    # Initialize plot
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(5, 5.2))
    ax = ax.ravel()

    # Plot data
    utils.samples(ax=ax[0], X=X, y=y)

    # Regular PCA
    model = PCA(n_components=2)
    X_pca = model.fit_transform(X)
    utils.samples(ax=ax[1], X=X_pca, y=y)

    # Kernel PCA
    model = KernelPCA(n_components=2, kernel="cosine", gamma=15)
    X_kpca = model.fit_transform(X)
    utils.samples(ax=ax[2], X=X_kpca, y=y)

    # Kernel PCA
    model = KernelPCA(n_components=2, kernel="rbf", gamma=15)
    X_kpca = model.fit_transform(X)
    utils.samples(ax=ax[3], X=X_kpca, y=y)

    # Labels
    for a in ax:
        a.set(xticks=[], yticks=[], xmargin=0.2, ymargin=0.2)
    ax[0].set(xlabel="feature x₁", ylabel="feature x₂", title="original")
    ax[1].set(xlabel="component z₁", ylabel="component z₂", title="pca")
    ax[2].set(
        xlabel="component z₁",
        ylabel="component z₂",
        title="kernel pca (cosine)",
    )
    ax[3].set(
        xlabel="component z₁",
        ylabel="component z₂",
        title="kernel pca (rbf)",
    )

    # Save figure
    fig.tight_layout(h_pad=0.2)
    fig.savefig(f"{args.output_dir}/kernel_pca.png")


if __name__ == "__main__":
    main()
