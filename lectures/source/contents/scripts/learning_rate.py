import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.svm import SVC

import plot


def loss(parameters):
    """Simple quadratic loss function."""
    return (
        parameters**2
        + 5
        + 1 * np.sin(3 * parameters) * np.exp(-0.1 * parameters**2)
    )


def descent(learning_rate=0.5):

    # Customize matplotlib style
    plot.xkcd_style()

    # Create learning rate matrix
    fig, ax = plt.subplots(figsize=(3, 3))
    parameters = np.linspace(-10, 10, 100)

    # Plot loss
    ax.plot(parameters, loss(parameters), color="C0")

    # Slow learning rate
    initial_param = 9.5
    for step in range(4):
        grad = loss(initial_param + 0.01) - loss(initial_param - 0.01)
        new_param = initial_param - learning_rate * grad
        ax.plot(
            [initial_param, new_param],
            [loss(initial_param), loss(new_param)],
            "o",
            color="C1",
            lw=2,
            mec="k",
        )
        sign = np.sign(new_param - initial_param)
        mag = np.abs(new_param - initial_param)
        if mag > 1.25:
            ax.annotate(
                "",
                xy=(initial_param, loss(initial_param)),
                xytext=(new_param, loss(new_param)),
                arrowprops=dict(
                    arrowstyle="<-",
                    shrinkA=4,
                    shrinkB=6,
                    color="k",
                    mutation_scale=10,
                    connectionstyle=f"arc3,rad={sign * (mag / 200 + 1) * 0.5}",
                ),
            )
        initial_param = new_param

    # Labels
    ax.set_xlabel("parameter a")
    ax.set_ylabel("loss function  L(a)")
    ax.set_xticks([])
    ax.set_yticks([])

    # Save figure
    return fig


if __name__ == "__main__":
    fig = descent(learning_rate=4)
    fig.savefig("contents/figures/learning_rate_slow.png")

    fig = descent(learning_rate=45)
    fig.savefig("contents/figures/learning_rate_fast.png")

    fig = descent(learning_rate=14)
    fig.savefig("contents/figures/learning_rate_right.png")
