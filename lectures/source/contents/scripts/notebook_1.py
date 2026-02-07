import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import RANSACRegressor, LinearRegression

import utils


def main():

    # Arguments
    args = utils.parse_args()

    # Generate random data
    X, y = make_regression(
        n_samples=50, n_features=1, noise=20, random_state=0
    )

    # Normalize data
    X = (X - X.min()) / (X.max() - X.min())
    y = (y - y.min()) / (y.max() - y.min())

    # Add outliers
    n_outliers = 10
    np.random.seed(42)
    outliers_X = np.random.uniform(
        low=X.max(), high=X.max() * 3, size=(n_outliers, 1)
    )
    outliers_y = np.random.uniform(
        low=y.max(), high=1.5 * y.max(), size=n_outliers
    )
    X = np.vstack([X, outliers_X])
    y = np.hstack([y, outliers_y])

    # Fit normal and robust linear regression models
    ransac = RANSACRegressor(random_state=0)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    linear = LinearRegression()
    linear.fit(X, y)

    # Figure
    fig, ax = plt.subplots(figsize=(4, 3))

    # Plot data
    ax.plot(X, y, "o", c="C0", mec="k", label="data")
    ax.plot(
        X[~inlier_mask],
        y[~inlier_mask],
        "o",
        color="C1",
        mec="k",
        label="outliers",
    )

    # Plot model
    x_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_fit = ransac.predict(x_fit)
    ax.plot(x_fit, y_fit, color="C2", lw=2, label="robust model")
    y_fit = linear.predict(x_fit)
    ax.plot(x_fit, y_fit, color="C3", lw=2, label="linear model")

    # Labels
    ax.set(xticks=[], yticks=[])
    ax.set_xlabel("water turbidity (T)")
    ax.set_ylabel("suspended load (C)")
    ax.legend(loc="upper center", fontsize="small", ncol=2)

    fig.tight_layout(pad=0.2)
    fig.savefig(args.output_dir / "notebook_1.png")


if __name__ == "__main__":
    main()
