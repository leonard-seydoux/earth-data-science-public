"""Regression plot script."""

import matplotlib.pyplot as plt
import matplotlib.patheffects as mpath
import numpy as np
from sklearn.linear_model import LinearRegression

import plot


def generate_data(n_samples=30):
    feature = np.random.normal(loc=0, scale=1, size=n_samples)
    label = 2 * feature + np.random.normal(0, 0.8, size=feature.shape)
    return feature, label


def linear_regression_line(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = model.predict(x_fit.reshape(-1, 1))
    return x_fit, y_fit


def main():

    # Generate data and fit model
    np.random.seed(42)
    feature, label = generate_data()
    x, y = linear_regression_line(feature, label)

    # Initialize plot
    plot.xkcd_style()
    fig, ax = plt.subplots(figsize=(3, 3))

    # Data samples
    ax.plot(x, y, color=plot.brighter("C5", -1), lw=2)
    ax.plot(feature, label, "o", color="C0", mec="k")

    # Remove labels
    ax.set_xlabel("feature x")
    ax.set_ylabel("label y")
    ax.set_xticks([])
    ax.set_yticks([])

    # Annotate model
    ax.annotate(
        "model",
        xy=(1.5, 1.5),
        xytext=(0.1, 0.95),
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=-0.2",
            mutation_scale=15,
            lw=1.2,
            shrinkA=10,
            shrinkB=30,
        ),
        textcoords="axes fraction",
        ha="left",
        va="top",
        xycoords="data",
        fontweight="normal",
    )

    # Save figure
    fig.savefig("contents/figures/regression.png")


if __name__ == "__main__":
    main()
