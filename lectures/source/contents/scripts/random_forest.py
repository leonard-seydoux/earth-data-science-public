import numpy as np
from matplotlib import pyplot as plt

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import utils

utils.xkcd_style()


def main():

    # Arguments
    args = utils.parse_args()

    iris = load_iris()
    X = iris.data[:, :3]
    y = iris.target
    clf = RandomForestClassifier(
        random_state=0, criterion="log_loss", max_features=3, max_depth=5
    )
    clf.fit(X, y)

    fig, ax = plt.subplots(5, 4, figsize=(5, 6))
    for estimator, ax in zip(clf.estimators_, ax.flatten()):
        annotations = tree.plot_tree(
            estimator,
            ax=ax,
            filled=True,
            rounded=True,
            fontsize=3,
            label="none",
        )
        for item in annotations:
            item.set_text(7 * " ")

    # Save
    fig.tight_layout()
    fig.savefig(args.output_dir / "random_forest.png")


if __name__ == "__main__":
    main()
