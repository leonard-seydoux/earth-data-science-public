# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import utils

args = utils.parse_args()

names = [
    "nearest\nneighbors",
    "support vector\nmachine\n(linear kernel)",
    "support vector\nmachine\n(radial kernel)",
    "decision\ntrees",
    "random\nforest",
    "multi-layer\nperceptron",
]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear", random_state=42),
    SVC(kernel="rbf", random_state=42),
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(max_depth=6, n_estimators=20, random_state=42),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
]

X, y = make_classification(
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
    n_samples=200,
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    make_moons(noise=0.3, random_state=0, n_samples=200),
    make_circles(noise=0.2, factor=0.5, random_state=1, n_samples=200),
    linearly_separable,
]

figure = plt.figure(figsize=(12, 6))
i = 1
# iterate over datasets
for dataset in datasets:

    # Split into training and test part
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42
    )

    # Domain bounds
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Plot dataset
    cmap = utils.cycler_cmap(2)
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if dataset is datasets[0]:
        ax.set_title("labeled\ndataset")

    # Plot the testing points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=cmap,
        s=20,
        edgecolors="k",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            cmap=cmap,
            ax=ax,
            plot_method="contourf",
            levels=[-10, 0, 10],
            alpha=0.3,
            zorder=0,
        )
        DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            colors="w",
            ax=ax,
            plot_method="contour",
            levels=[0],
            zorder=0,
        )

        # Plot testing set
        ax.scatter(*X_test.T, c=y_test, cmap=cmap, ec="k", s=20)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if dataset is datasets[0]:
            ax.set_title(name)
        ax.text(
            0.92,
            0.11,
            f"{score:.0%}",
            ha="right",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
            size="small",
        )
        i += 1

plt.tight_layout()
plt.savefig(args.output_dir / "classification_comparison.png")
