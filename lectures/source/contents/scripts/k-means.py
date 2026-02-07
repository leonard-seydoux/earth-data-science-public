"""K-means clustering algorithm visualization in GIF."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import plot

plot.xkcd_style()
ax_kwargs = dict(
    xlabel="feature x₁", ylabel="feature x₂", xticks=[], yticks=[]
)


def kmeans_animation(X, n_clusters=3, n_frames=12, interval=800):
    """Create a GIF visualizing K-means clustering."""

    # Initial centroids
    centroids = np.array([[1, 1, 10, 10], [10, 10, 1, 1]]).T

    # Initial KMeans model
    kmeans = KMeans(n_clusters=n_clusters, init=centroids, max_iter=0)

    # Initialize figure and axes
    fig, ax = plt.subplots(figsize=(3, 3), tight_layout=True)
    ax.set(**ax_kwargs)
    plot.samples(ax, X)
    fig.savefig("contents/figures/k-means-initial.png")

    def update(frame):

        # Clear axis
        ax.clear()
        ax.set(**ax_kwargs)

        # Update number of iterations
        kmeans.max_iter = frame + 1
        y = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        # Plot samples and centroids
        plot.samples(ax, X, y)
        ax.plot(centroids[:, 0], centroids[:, 1], "ko", mfc="w")

        # Plot decision boundaries
        plot.draw_boundaries(ax, kmeans)

        # Inertia text
        intertia_label = f"inertia = {kmeans.inertia_:.1f}"
        ax.text(0.5, 0.05, intertia_label, transform=ax.transAxes, ha="center")

    # Create and save animation
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval)
    ani.save("contents/figures/k-means.gif")

    # Final frame
    plt.close()
    fig.savefig("contents/figures/k-means-final.png")


if __name__ == "__main__":

    # Parameters
    n_clusters = 4

    # Generate standardized features
    X, _ = make_blobs(centers=n_clusters, cluster_std=1.2, random_state=40)
    X = StandardScaler().fit_transform(X)

    # Clustering animation
    kmeans_animation(X, n_clusters=n_clusters)
