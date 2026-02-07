"""Classification with Support Vector Machines (SVM)"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

import utils


def main():

    # Arguments
    args = utils.parse_args()

    # Generate data
    X, y = make_blobs(n_samples=70, centers=2, cluster_std=2.5, random_state=1)

    # Train classifier
    model = SVC(kernel="linear")
    model.fit(X, y)

    # Initialize plot
    fig, ax = plt.subplots(figsize=(3, 3))

    # Show scatter plot and boundaries
    utils.samples(ax, X, y)
    utils.draw_boundaries(ax, model)

    # Annotate labels
    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="arc3,rad=0.2",
        mutation_scale=15,
        lw=1.2,
    )
    annotate = dict(
        arrowprops=arrowprops,
        ha="center",
        va="center",
        xycoords="axes fraction",
        fontweight="normal",
    )
    ax.annotate("label y=0", xy=(0.51, 0.75), xytext=(0.3, 0.9), **annotate)
    ax.annotate("label y=1", xy=(0.45, 0.25), xytext=(0.65, 0.07), **annotate)

    # Axes labels
    ax.set(xlabel="feature x₁", ylabel="feature x₂", xticks=[], yticks=[])

    # Save
    fig.savefig(args.output_dir / "classification.png")


if __name__ == "__main__":
    main()
