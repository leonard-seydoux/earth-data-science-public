"""Classification with linear support vector machine"""

from sklearn.datasets import make_blobs
from sklearn.svm import SVC

import utils


def main():

    # Arguments
    args = utils.parse_args()

    # Data and model
    X, y = make_blobs(n_samples=70, centers=2, cluster_std=2.5, random_state=1)
    model = SVC(kernel="linear")
    model.fit(X, y)

    # Initialize plot
    fig, ax = utils.square_canvas()

    # Show scatter plot and boundaries
    utils.scatter_samples(ax, X, y)
    utils.plot_boundary_decision(ax, model)

    # Class annotation
    annotate = dict(
        arrowprops=utils.DEFAULT_ARROWPROPS,
        ha="center",
        va="center",
        xycoords="axes fraction",
        fontweight="normal",
    )
    ax.annotate(
        "label $y=0$",
        xy=(0.51, 0.75),
        xytext=(0.3, 0.9),
        color="C0",
        **annotate
    )
    ax.annotate(
        "label $y=1$",
        xy=(0.45, 0.25),
        xytext=(0.65, 0.07),
        color="C1",
        **annotate
    )

    # Axes labels
    ax.set(
        xlabel="feature $x_1$", ylabel="feature $x_2$", xticks=[], yticks=[]
    )

    # Save
    fig.savefig(args.output_dir / "classification.png")


if __name__ == "__main__":
    main()
