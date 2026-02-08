"""Clustering with k-means"""

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import utils


def cluster_data(features, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=0)
    model.fit(features)
    return model


def main():

    # Arguments
    args = utils.parse_args()

    # Data and model
    X, _ = make_blobs(centers=3, cluster_std=1.5, random_state=42)
    model = cluster_data(X, n_clusters=3)
    y = model.labels_
    centroids = model.cluster_centers_

    # Figure
    fig, ax = utils.square_canvas()

    # Plot samples and boundaries
    utils.scatter_samples(ax, X, y)
    utils.plot_boundary_decision(ax, model)

    # Cluster annotation
    for centroid in centroids:
        ax.annotate(
            "clusters",
            xy=centroid,
            xytext=(0.35, 0.5),
            arrowprops={**utils.DEFAULT_ARROWPROPS, "shrinkB": 30},
            textcoords="axes fraction",
            ha="center",
            va="center",
            xycoords="data",
        )

    # Axes labels
    ax.set(xlabel="feature x₁", ylabel="feature x₂")

    # Save figure
    fig.savefig(f"{args.output_dir}/clustering.png")


if __name__ == "__main__":
    main()
