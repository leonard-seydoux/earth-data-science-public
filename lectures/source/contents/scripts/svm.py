"""Support Vector Machine (SVM) example."""

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.svm import SVC

import utils


def make_classification(x, y):
    model = SVC(kernel="linear")
    model.fit(x, y)
    return model


def main():

    # Arguments
    args = utils.parse_args()

    # Create accuracy matrix
    features, labels = make_blobs(
        n_samples=70, centers=2, cluster_std=2.5, random_state=1
    )

    # Initialize plot
    fig, ax = utils.square_canvas()

    # Show scatter plot
    utils.scatter_samples(ax, features, labels)

    # Labels
    ax.set_xlabel("feature x₁")
    ax.set_ylabel("feature x₂")
    fig.savefig(f"{args.output_dir}/svm_data.png", dpi=300)

    # SVM decision boundary
    model = make_classification(features, labels)

    # Get support vectors (closest points to the decision boundary)
    sv = model.support_vectors_
    for point in sv:
        ax.annotate(
            "Support\nvectors",
            xy=(point[0], point[1]),
            xytext=(0.9, 0.15),
            arrowprops=dict(
                arrowstyle="->",
                color="k",
                connectionstyle="arc3,rad=-0.2",
                shrinkA=0,
                shrinkB=5,
            ),
            va="center",
            textcoords="axes fraction",
            fontsize="small",
            ha="right",
        )

    fig.savefig(args.output_dir / "svm_support_vectors.png")

    utils.plot_boundary_decision(ax, model, colors="w")
    utils.colorized_boundaries(ax, model)

    # Now annotate the margins
    w = model.coef_[0]
    b = model.intercept_[0]
    margin = 1 / np.sqrt(np.sum(w**2))
    xx = np.linspace(features[:, 0].min(), features[:, 0].max(), 100)
    yy_down = -(w[0] * xx + b - margin) / w[1]
    yy_up = -(w[0] * xx + b + margin) / w[1]
    ax.plot(xx, yy_down, "k--", linewidth=0.8)
    ax.plot(xx, yy_up, "k--", linewidth=0.8)

    # Save figure
    fig.savefig(args.output_dir / "svm.png")


if __name__ == "__main__":
    main()
