"""Classification with Support Vector Machines (SVM)"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

import plot


def make_classification(x, y):
    model = SVC(kernel="linear")
    model.fit(x, y)
    return model


def main():

    # Generate data
    X, y = make_blobs(n_samples=70, centers=2, cluster_std=2.5, random_state=1)

    # Classification model
    model = make_classification(X, y)

    # Initialize plot
    plot.xkcd_style()
    fig, ax = plt.subplots(figsize=(3, 3))

    # Show scatter plot and boundaries
    plot.samples(ax, X, y)
    plot.draw_boundaries(ax, model)

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
    fig.savefig("contents/figures/classification.png")


if __name__ == "__main__":
    main()
