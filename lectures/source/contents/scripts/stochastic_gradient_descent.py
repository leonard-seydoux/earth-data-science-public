"""Gradient descent with stochastic optimization algorithm."""

import matplotlib.pyplot as plt
import numpy as np

import utils


def loss(parameters):
    """Loss function with multiple local minima."""
    p1 = (parameters - 3) ** 2
    p2 = -np.sin(parameters) * 40 * np.exp(-0.1 * parameters**2)
    return p1 + p2


np.random.seed(0)


def descent(learning_rate=0.5):

    args = utils.parse_args()

    # Customize matplotlib style
    utils.xkcd_style()

    # Create learning rate matrix
    fig, ax = plt.subplots(figsize=(3, 3))
    parameters = np.linspace(-10, 10, 100)

    # Plot loss
    ax.plot(parameters, loss(parameters), color="C0")

    # Slow learning rate
    initial_param = -6.5
    for _ in range(5):
        grad = loss(initial_param + 0.01) - loss(initial_param - 0.01)
        new_param = (
            initial_param
            - learning_rate * grad * np.abs(np.random.randn()) * 3
        )

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

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("p")
    ax.set_ylabel("loss function  L(p)")

    # Save figure
    fig.savefig(f"{args.output_dir}/stochastic_gradient_descent.png")


if __name__ == "__main__":
    descent(learning_rate=4)
