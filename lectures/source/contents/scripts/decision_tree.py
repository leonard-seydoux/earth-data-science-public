import numpy as np
from matplotlib import pyplot as plt

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

import plot

plot.xkcd_style()


def main():

    iris = load_iris()
    X = iris.data[:, :3]
    y = iris.target
    iris.feature_names = [
        name.replace(" (cm)", "") for name in iris.feature_names
    ]

    clf = DecisionTreeClassifier(
        random_state=42, criterion="log_loss", max_features=2, max_depth=4
    )
    clf.fit(X, y)
    fig, ax = plt.subplots(figsize=(6, 5))
    annotations = tree.plot_tree(
        clf,
        proportion=True,
        ax=ax,
        filled=True,
        rounded=True,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        fontsize=11,
    )
    for item in annotations:
        if isinstance(item, plt.Text):

            if "class" in item.get_text():
                text = item.get_text()
                text = text.replace("class", "label")
                item.set_text(text)

            if "value" in item.get_text():
                text = item.get_text()
                lines = text.split("\n")
                lines = [line for line in lines if "value" not in line]
                item.set_text("\n".join(lines))

            if "log_loss" in item.get_text():
                text = item.get_text()
                lines = text.split("\n")
                lines = [line for line in lines if "log_loss" not in line]
                item.set_text("\n".join(lines))

    # Save
    fig.tight_layout()
    fig.savefig(
        "contents/figures/decision_tree.png",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
