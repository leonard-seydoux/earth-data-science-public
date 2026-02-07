import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.animation as animation


import figures
from models import LinearModel

SAVE_PATH = "../lectures-2/images/regression_linear"
CMAP = "Greys"
LOSS_FUNCTION = mean_squared_error
np.random.seed(0)


def plot_loss_surface(x, model, ax=None):
    """Plot the loss surface of the linear model.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        The input data.
    model : LinearModel
        The linear model.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on.

    Returns
    -------
    vmax : float
        The maximum loss value.
    """
    # Get axes
    ax = ax or plt.gca()

    # Extract loss surface
    y = model.forward(x)
    loss, a, b = model.loss_surface(x, y)

    # Plot loss surface
    vmax = loss.max()
    levels = np.linspace(0, 1, 9) ** 2
    levels *= vmax
    mappable = ax.contourf(
        a, b, loss.T, cmap=CMAP, vmin=0, vmax=vmax, levels=levels
    )

    # Plot true parameters
    ax.plot(model.slope, model.intercept, "+", mec="k", ms=10)

    # Labels
    colorbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=15)
    colorbar.set_label(r"$\mathcal{L}(\theta_1, \theta_2)$")
    colorbar.set_ticks([])

    return vmax


def plot_model(slope, intercept, ax=None, **kwargs):
    """Plot the model."""
    # Get axes
    ax = ax or plt.gca()

    # Plot the true linear model
    ax.axline((0, intercept), slope=slope, **kwargs)


def plot(x, y, model, slopes=None, intercepts=None, filename=None):
    """Plot linear data with labels."""
    # Create a figure and axes
    fig, ax = figures.linear_regression()

    # Plot the data points
    ax[0].plot(x, y, ".", ms=2)

    # Plot the true linear model
    ax[0].axline((0, model.intercept), slope=model.slope, color="k")

    # Plot true loss and parameters
    plot_loss_surface(x, model, ax=ax[1])

    # If provided, plot the linear models
    if slopes is not None and intercepts is not None:

        losses = []

        # Plot the linear models
        for slope, intercept in zip(slopes, intercepts):
            loss = LOSS_FUNCTION(y, model.f(x, slope, intercept))
            losses.append(loss)
            plot_model(slope, intercept, ax=ax[0], color="0.9", zorder=0)
            ax[1].plot(slope, intercept, ".", c="0.5", ms=2)

        # Highlight the best model
        best = np.argmin(losses)
        plot_model(slopes[best], intercepts[best], ax=ax[0], color="C1")
        ax[1].plot(slopes[best], intercepts[best], "*", color="C1")

    # Add mock legend
    ax[0].plot([], [], "k-", label="True model")
    ax[0].plot([], [], "-", c="0.9", label="Tested model")
    ax[0].plot([], [], "C1-", label="Best model")
    ax[0].legend(loc="upper left", fontsize="x-small")
    ax[1].plot([], [], ".", c="0.5", ms=1, label="Tested model")
    ax[1].plot([], [], "C1*", label="Best model")
    ax[1].legend(loc="upper left", fontsize="x-small")

    # Save the figure
    if filename is not None:
        figures.savefig(f"{SAVE_PATH}/{filename}", fig)


def init_plot(x, y, model):
    """Initialize the plot for the animation.

    Parameters
    ----------
    x : np.ndarray
        The input data.
    y : np.ndarray
        The target data.
    model : LinearModel
        The linear model.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : np.ndarray
        The axes.
    best_line : matplotlib.lines.Line2D
        The line representing the model.
    searched_models : matplotlib.lines.Line2D
        The points representing the search trajectory.
    best_model : matplotlib.lines.Line2D
        The best point so far.
    """
    fig, ax = figures.linear_regression()

    # Plot the data points and true line
    ax[0].plot(x, y, ".", ms=2)
    ax[0].axline((0, model.intercept), slope=model.slope, color="k")

    # True loss surface
    plot_loss_surface(x, model, ax=ax[1])

    # Add mock legend
    ax[0].plot([], [], "k-", label="True model")
    ax[0].plot([], [], "-", c="0.9", label="Tested model")
    ax[0].plot([], [], "C1-", label="Best model")
    ax[0].legend(loc="upper left", fontsize="x-small")
    ax[1].plot([], [], ".", c="0.5", ms=1, label="Tested model")
    ax[1].plot([], [], "C1*", label="Best model")
    ax[1].legend(loc="upper left", fontsize="x-small")

    # Add plot elements for animation
    best_line = ax[0].axline((0, -10), slope=0, color="C1", zorder=10)
    (searched_models,) = ax[1].plot([], [], ".", color="0.5", ms=1)
    (best_model,) = ax[1].plot([], [], "*", color="C1", ms=6)
    return fig, ax, searched_models, best_line, best_model


def animate_frame(
    i,
    x,
    y,
    model,
    slopes,
    intercepts,
    ax,
    searched_models,
    best_line,
    best_model,
):
    """Animate a frame of the optimization process.

    Parameters
    ----------
    i : int
        The frame number.
    x : np.ndarray
        The input data.
    y : np.ndarray
        The target data.
    model : LinearModel
        The linear model.
    slopes : np.ndarray
        The slopes of the models.
    intercepts : np.ndarray
        The intercepts of the models.
    ax : np.ndarray
        The axes.
    searched_models : matplotlib.lines.Line2D
        The points representing the search trajectory.
    best_line : matplotlib.lines.Line2D
        The line representing the best model.
    best_model : matplotlib.lines.Line2D
        The best point so far.
    """
    # Plot currently tested model, and update the points of the search trajectory
    ax[0].axline((0, intercepts[i]), slope=slopes[i], color="0.9")
    searched_models.set_data(slopes[: i + 1], intercepts[: i + 1])

    # Find best model so far, and update the best model point
    if i > 1:
        losses = [
            LOSS_FUNCTION(y, model.f(x, a, b))
            for a, b in zip(slopes[: i + 1], intercepts[: i + 1])
        ]
        best_idx = np.argmin(losses)
        best_model.set_data([slopes[best_idx]], [intercepts[best_idx]])
        best_line.set_slope(slopes[best_idx])
        best_line.set_xy1(0, intercepts[best_idx])


def create_animation(x, y, model, slopes, intercepts, filename):
    """Create an animation of the optimization process.

    Parameters
    ----------
    x : np.ndarray
        The input data.
    y : np.ndarray
        The target data.
    model : LinearModel
        The linear model.
    slopes : np.ndarray
        The slopes of the models.
    intercepts : np.ndarray
        The intercepts of the models.
    filename : str
        The filename to save the animation to.
    """
    # Initialize plot
    fig, ax, searched_models, best_line, best_model = init_plot(x, y, model)

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        animate_frame,
        fargs=(
            x,
            y,
            model,
            slopes,
            intercepts,
            ax,
            searched_models,
            best_line,
            best_model,
        ),
        frames=len(slopes),
        repeat=False,
    )

    # Save as GIF using PillowWriter
    figures.saveani(f"{SAVE_PATH}/{filename}.gif", ani)


def main():

    # Generate data
    model = LinearModel(noise_variance=0.1)
    x = np.random.normal(0.5, 0.25, 200)
    y = model.forward(x)

    # Just data
    plot(x, y, model, filename="data")

    # Random search
    num_samples = 100
    slopes = np.random.uniform(-1, 3, num_samples)
    intercepts = np.random.uniform(-1, 1, num_samples)
    create_animation(x, y, model, slopes, intercepts, filename="random_search")
    plot(x, y, model, slopes, intercepts, filename="random_search")

    # Grid search
    grid_size = 8
    slopes = np.linspace(-0.8, 2.8, grid_size)
    intercepts = np.linspace(-0.8, 0.8, grid_size)
    grid = np.meshgrid(slopes, intercepts, indexing="ij")
    slopes, intercepts = grid[0].ravel(), grid[1].ravel()
    create_animation(x, y, model, slopes, intercepts, filename="grid_search")
    plot(x, y, model, slopes, intercepts, filename="grid_search")

    # Gradient descent
    learning = 0.1
    num_steps = 15
    a, b = 2.2, 1
    slopes, intercepts = [a], [b]
    for _ in range(num_steps):
        gradient = 2 * np.array([a - model.slope, b - model.intercept])
        a, b = (a - learning * gradient[0], b - learning * gradient[1])
        slopes.append(a)
        intercepts.append(b)
    create_animation(x, y, model, slopes, intercepts, filename="descend")
    plot(x, y, model, slopes, intercepts, filename="descend")

    # Least-squares fit
    a_star, b_star = model.fit(x, y)
    plot(x, y, model, slopes=[a_star], intercepts=[b_star], filename="fit")


if __name__ == "__main__":
    main()
