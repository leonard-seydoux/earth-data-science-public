"""Clustering with K-Means"""

import matplotlib.pyplot as plt
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

    # Generate data
    X, _ = make_blobs(centers=3, cluster_std=1.5, random_state=42)

    # Infer labels with clustering
    model = cluster_data(X, n_clusters=3)
    y = model.labels_

    # Figure
    utils.xkcd_style()
    fig, ax = plt.subplots(figsize=(3, 3))

    # Plot samples and boundaries
    utils.samples(ax, X, y)
    utils.draw_boundaries(ax, model)

    # Annotate centroids
    centroids = model.cluster_centers_
    for centroid in centroids:
        ax.annotate(
            "clusters",
            xy=centroid,
            xytext=(0.2, 0.5),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0.2",
                mutation_scale=15,
                lw=1.2,
                shrinkB=30,
            ),
            textcoords="axes fraction",
            ha="center",
            va="bottom",
            xycoords="data",
            fontweight="normal",
        )

    # Axes labels
    ax.set(xlabel="feature x₁", ylabel="feature x₂", xticks=[], yticks=[])

    # Save figure
    fig.savefig(f"{args.output_dir}/clustering.png")


if __name__ == "__main__":
    main()
