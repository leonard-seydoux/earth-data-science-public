"""K-means clustering algorithm visualization in GIF."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import SpectralClustering

import utils

utils.xkcd_style()


def main():

    # Arguments
    args = utils.parse_args()

    fig, ax = plt.subplots(
        nrows=2, figsize=(3.5, 5.2), gridspec_kw={"height_ratios": [3, 1.5]}
    )

    X, _ = make_blobs(
        n_samples=300, centers=4, cluster_std=1.2, random_state=40
    )
    cls = SpectralClustering(
        n_clusters=4,
        assign_labels="discretize",
        random_state=42,
        affinity="nearest_neighbors",
    )
    cls.fit(X)

    # Represent graph
    aff = cls.affinity_matrix_
    for i in range(aff.shape[0]):
        for j in range(aff.shape[1]):
            if aff[i, j] > 0:
                ax[0].plot(
                    [X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], lw=0.5, c="k"
                )

    # Show eigenvectors
    utils.samples(ax[0], X, cls.labels_)

    # Show eigenvalues
    eigenvalues = np.linalg.eigvalsh(aff.toarray())[::-1][:10]
    ax[1].plot(eigenvalues, "o-", color="C4")
    for i in range(4):
        ax[1].plot(i, eigenvalues[i], "o", color=f"C{i}")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel("feature x₁")
    ax[0].set_ylabel("feature x₂")
    ax[1].set_xlabel("first 10 eigenvalues")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    fig.tight_layout()
    fig.savefig(f"{args.output_dir}/spectral_clustering.png")


if __name__ == "__main__":
    main()
