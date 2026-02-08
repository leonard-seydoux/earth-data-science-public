"""K-means clustering algorithm visualization in GIF."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import utils


def main(n_frames=12, interval=800):

    # Arguments
    args = utils.parse_args()

    # Data and model
    n_clusters = 4
    X, _ = make_blobs(centers=n_clusters, cluster_std=1.2, random_state=40)
    X = StandardScaler().fit_transform(X)
    init_centroids = np.array([[1, 1, 10, 10], [10, 10, 1, 1]]).T
    model = KMeans(n_clusters=n_clusters, init=init_centroids, max_iter=0)

    # Initialize figure and axes
    fig, ax = utils.square_canvas()
    ax_kwargs = dict(
        xlabel="feature x₁", ylabel="feature x₂", xticks=[], yticks=[]
    )
    ax.set(**ax_kwargs)
    utils.scatter_samples(ax, X)
    fig.savefig(args.output_dir / "k_means_initial.png")

    def update(frame):

        # Clear axis
        ax.clear()
        ax.set(**ax_kwargs)

        # Update number of iterations
        model.max_iter = frame + 1
        y = model.fit_predict(X)
        centroids = model.cluster_centers_

        # Plot samples and centroids
        utils.scatter_samples(ax, X, y)
        ax.plot(centroids[:, 0], centroids[:, 1], "ko", mfc="w")

        # Plot decision boundaries
        utils.plot_boundary_decision(ax, model)

        # Inertia text
        inertia_label = f"inertia = {model.inertia_:.1f}"
        ax.text(0.5, 0.05, inertia_label, transform=ax.transAxes, ha="center")

    # Create and save animation
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval)
    ani.save(args.output_dir / "k_means.gif")

    # Final frame
    plt.close()
    fig.savefig(args.output_dir / "k_means_final.png")


if __name__ == "__main__":
    main()
