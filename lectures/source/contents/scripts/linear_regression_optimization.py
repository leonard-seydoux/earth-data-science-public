import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib.animation import PillowWriter

import utils

GRID_SEARCH_A = np.linspace(-2, 2, 12)
GRID_SEARCH_B = np.linspace(-1, 4, 12)


def make_data(n_samples=100, noise=0.2, random_state=0):
    """Generate synthetic linear regression data."""
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0, 1, size=(n_samples, 1))
    y = 1 + X + rng.normal(0, noise, size=X.shape)
    return X, y


def base_figure(X, y):

    fig = plt.figure(figsize=(3.2, 6))
    ax = [
        fig.add_subplot(2, 1, 1),
        fig.add_subplot(2, 1, 2, projection="3d"),
    ]

    utils.plot_samples(ax[0], np.c_[X.flatten(), y.flatten()], color="C0")

    # labels
    ax[0].set_xlabel("feature x₁", weight="normal")
    ax[0].set_ylabel("feature x₂", weight="normal")
    ax[0].set_title("data space", weight="bold")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # Loss
    # Plot loss function
    A, B = np.meshgrid(GRID_SEARCH_A, GRID_SEARCH_B)
    Z = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            y_pred = A[i, j] * X + B[i, j]
            Z[i, j] = np.mean((y - y_pred) ** 2)

    ax[1].plot_wireframe(A, B, Z, color="C4", alpha=1)
    ax[1].set_xlabel("slope a", labelpad=-8, weight="normal")
    ax[1].set_ylabel("intercept b", labelpad=-8, weight="normal")
    ax[1].set_title("parameter space", weight="bold")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_zticks([])
    ax[1].set_zlabel("loss", labelpad=-8)
    ax[1].set_facecolor("none")
    ax[1].view_init(elev=30, azim=45)
    ax[1].set_zlim(0, np.max(Z) * 1.5)
    fig.tight_layout(pad=1, h_pad=0.4)

    # Fix axes limits
    ax[0].set_xlim(*ax[0].get_xlim())
    ax[0].set_ylim(*ax[0].get_ylim())

    return fig, ax


def show_test(a, b, ax, color="C1", linewidth=1, markersize=4):

    # Plot initial regression line
    x_line = np.linspace(-1, 2, 300)
    y_line = a * x_line + b
    y_line[y_line < y.min()] = np.nan
    y_line[y_line > y.max()] = np.nan
    loss = np.mean((y - (a * X + b)) ** 2)
    alpha = 1 - np.sqrt((np.atan(loss) / (np.pi / 2)))
    alpha = 0.1 if alpha < 0.1 else alpha
    ax[0].plot(x_line, y_line, color, alpha=alpha, linewidth=linewidth)
    ax[1].plot(a, b, loss, "o", color=color, mec="k", ms=markersize, zorder=3)


if __name__ == "__main__":

    args = utils.parse_args()
    utils.xkcd_style()

    X, y = make_data()
    fig, ax = base_figure(X, y)
    show_test(1, 1, ax)
    fig.savefig(f"{args.output_dir}/linear_regression_true.png")

    # Create GIF for gradient descent
    fig, ax = base_figure(X, y)
    writer = PillowWriter(fps=7)
    with writer.saving(
        fig,
        f"{args.output_dir}/linear_regression_gradient_descent.gif",
        dpi=300,
    ):
        a, b = -2, 4
        learning_rate = 0.6
        for _ in range(50):
            y_pred = a * X + b
            error = y_pred - y
            grad_a = (2 / len(X)) * np.sum(error * X)
            grad_b = (2 / len(X)) * np.sum(error)
            a -= learning_rate * grad_a
            b -= learning_rate * grad_b
            show_test(a, b, ax, color="C3")
            writer.grab_frame()
        show_test(a, b, ax, color="C1", linewidth=2, markersize=8)
        for _ in range(20):
            writer.grab_frame()
    # Create GIF for grid search
    fig, ax = base_figure(X, y)
    writer = PillowWriter(fps=7)
    # Find best parameters
    best_loss = float("inf")
    best_a, best_b = 0, 0
    for a in GRID_SEARCH_A:
        for b in GRID_SEARCH_B:
            loss = np.mean((y - (a * X + b)) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_a, best_b = a, b
    with writer.saving(
        fig, f"{args.output_dir}/linear_regression_grid_search.gif", dpi=300
    ):
        for a in GRID_SEARCH_A:
            for b in GRID_SEARCH_B:
                show_test(a, b, ax, color="C3")
                writer.grab_frame()
        # Show final best model in red and hold for 2 seconds
        show_test(best_a, best_b, ax, color="C1", linewidth=2, markersize=8)
        for _ in range(20):  # 2 seconds at 7 fps
            writer.grab_frame()

    # Create GIF for random search
    rng = np.random.default_rng(0)
    fig, ax = base_figure(X, y)
    writer = PillowWriter(fps=7)
    # Track best parameters
    best_loss = float("inf")
    best_a, best_b = 0, 0
    with writer.saving(
        fig, f"{args.output_dir}/linear_regression_random_search.gif", dpi=300
    ):
        for _ in range(100):  # Reduced from 200 for faster generation
            a = rng.uniform(-2, 2)
            b = rng.uniform(-1, 4)
            loss = np.mean((y - (a * X + b)) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_a, best_b = a, b
            show_test(a, b, ax, color="C3")
            writer.grab_frame()
        # Show final best model in red and hold for 2 seconds
        show_test(best_a, best_b, ax, color="C1", linewidth=2, markersize=8)
        for _ in range(20):  # 2 seconds at 7 fps
            writer.grab_frame()
