"""Regression plot script."""

import matplotlib.pyplot as plt
import matplotlib.patheffects as mpath
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

import utils


def generate_data(n_samples=30):
    """Generate synthetic regression data."""
    feature = np.random.normal(loc=0, scale=1, size=n_samples)
    label = 2 * feature**3 + np.random.normal(0, 2, size=feature.shape)
    return feature, label


def make_fit(model, x, y):
    """Fit linear regression and return line coordinates."""
    model.fit(x.reshape(-1, 1), y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = model.predict(x_fit.reshape(-1, 1))
    return x_fit, y_fit


def fit(model, test=False):

    # Seed randomness
    np.random.seed(0)

    # Style
    utils.xkcd_style()

    # Create accuracy matrix
    feature, label = generate_data()
    x, y = make_fit(model, feature, label)
    x_test, y_test = generate_data(n_samples=20)

    # Initialize plot
    fig, ax = plt.subplots(figsize=(3, 3))

    # Data samples
    ax.plot(feature, label, "o", color="C0", mec="k")
    if test:
        ax.plot(x_test, y_test, "o", color="C1", mec="k")

    # Regression line
    ax.plot(x, y, color="C3", lw=2)

    # Remove labels
    ax.set_xlabel("feature x")
    ax.set_ylabel("label y")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(
        0.5,
        0.05,
        "train loss: {:.2f}".format(
            np.mean((model.predict(feature.reshape(-1, 1)) - label) ** 2)
        ),
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize="small",
    )

    # Text loss
    if test:
        ax.text(
            0.5,
            0.15,
            "test loss: {:.2f}".format(
                np.mean((model.predict(x_test.reshape(-1, 1)) - y_test) ** 2)
            ),
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize="small",
        )

    return fig


if __name__ == "__main__":

    args = utils.parse_args()

    model = LinearRegression()
    fit(model).savefig(f"{args.output_dir}/fit_under.png")
    fit(model, test=True).savefig(f"{args.output_dir}/fit_under_test.png")

    model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    fit(model).savefig(f"{args.output_dir}/fit_right.png")
    fit(model, test=True).savefig(f"{args.output_dir}/fit_right_test.png")

    model = KNeighborsRegressor(n_neighbors=1)
    fit(model).savefig(f"{args.output_dir}/fit_over.png")
    fit(model, test=True).savefig(f"{args.output_dir}/fit_over_test.png")
