"""Gradient descent with momentum optimization algorithm."""

import matplotlib.pyplot as plt
import numpy as np

import utils


def loss(parameters):
    """Loss function with multiple local minima."""
    p1 = (parameters - 3) ** 2
    p2 = -np.sin(parameters) * 40 * np.exp(-0.1 * parameters**2)
    return p1 + p2


def descent(learning_rate=1, momentum=0.6):

    args = utils.parse_args()

    # Customize matplotlib style
    utils.xkcd_style()

    # Create learning rate matrix
    fig, ax = plt.subplots(figsize=(3, 3))
    parameters = np.linspace(-10, 10, 100)

    # Plot loss
    ax.plot(parameters, loss(parameters), color="C0")

    # Slow learning rate
    initial_param = -7.5
    velocity = 0
    for _ in range(10):
        grad = loss(initial_param + 0.01) - loss(initial_param - 0.01)
        velocity = momentum * velocity + learning_rate * grad
        new_param = initial_param - velocity
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
    ax.set_xlabel("p")
    ax.set_ylabel("loss function  L(p)")
    ax.set_xticks([])
    ax.set_yticks([])

    # Save figure
    fig.savefig(
        args.output_dir
        / f"gradient_descent_momentum_lr{learning_rate}_mom{momentum}.png"
    )


if __name__ == "__main__":
    descent(learning_rate=1, momentum=0.6)
    descent(learning_rate=0.5, momentum=1)
