import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay

import utils

KERNELS = "linear", "rbf", "poly"


def make_classification(x, y, kernel="linear"):
    """Generate synthetic classification data."""
    model = SVC(kernel=kernel, gamma=1, degree=3)
    model.fit(x, y)
    return model


def main():

    # Arguments
    args = utils.parse_args()

    # Customize matplotlib style
    utils.xkcd_style()

    # Create accuracy matrix
    data = load_iris()
    features = data.data
    labels = data.target

    # Initialize plot
    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(5.5, 6.5))

    # Show scatter plot
    for j in range(len(KERNELS)):

        for i in range(len(np.unique(labels))):
            samples = features[labels == i]
            utils.samples(ax[j, 0], samples, labels[labels == i])

        # SVM decision boundary
        model = make_classification(features[:, :2], labels, kernel=KERNELS[j])
        utils.colorized_boundaries(ax[j, 0], model)
        utils.draw_boundaries(ax[j, 0], model, colors="w")

    # Labels
    for a, name in zip(ax[:, 0], KERNELS):
        a.set_ylabel("sepal width")
        a.set_xticks([])
        a.set_yticks([])
        a.text(-0.3, 0.5, name, transform=a.transAxes, ha="right")

    # Confusion matrices
    for j in range(len(KERNELS)):
        model = make_classification(features[:, :2], labels, kernel=KERNELS[j])
        y_pred = model.predict(features[:, :2])
        disp = ConfusionMatrixDisplay.from_predictions(
            labels,
            y_pred,
            ax=ax[j, 1],
            cmap="Blues",
            colorbar=False,
            # display_labels=data.target_names,
        )
        disp.ax_.set_title("")
        ax[j, 1].set_ylabel("true")
    for a, name in zip(ax[:, 1], KERNELS):
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlabel("")

    ax[2, 0].set_xlabel("sepal length")
    ax[2, 1].set_xlabel("predicted")

    # Save figure
    fig.tight_layout(w_pad=2.0)
    fig.savefig(f"{args.output_dir}/iris_confusion.png")

    # Hide confusion matrices and show decision boundaries only
    for a in ax[:, 1]:
        a.set_visible(False)
    # fig.tight_layout(w_pad=2.0)
    fig.savefig(f"{args.output_dir}/iris.png")


if __name__ == "__main__":
    main()
