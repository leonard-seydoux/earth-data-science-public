import matplotlib.pyplot as plt
import numpy as np

import utils


def main():

    # Arguments
    args = utils.parse_args()

    fig, ax = plt.subplots(figsize=(3.5, 4))

    # Architectures
    n = [4, 7, 2]

    # Input layer
    for layer, n_neurons in enumerate(n):
        for neuron in range(n_neurons):
            ax.add_patch(
                plt.Circle(
                    (layer * 3, neuron - n_neurons / 2 + 0.5),
                    0.3,
                    facecolor="w",
                    edgecolor="k",
                )
            )

    # Connections
    for layer in range(len(n) - 1):
        for neuron in range(n[layer]):
            for next_neuron in range(n[layer + 1]):
                ax.plot(
                    [layer * 3, (layer + 1) * 3],
                    [
                        neuron - n[layer] / 2 + 0.5,
                        next_neuron - n[layer + 1] / 2 + 0.5,
                    ],
                    color="k",
                    linewidth=1.5,
                    zorder=0,
                )

    # Labels
    ax.text(0, -n[0] / 2, "input\nx", ha="center", va="top")
    ax.text(3, -n[1] / 2, "hidden", ha="center", va="top")
    ax.text(6, -n[2] / 2, "output\ny", ha="center", va="top")

    # Labels
    ax.set_xlim(-0.5, 3 * len(n) - 2.5)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Save figure
    fig.savefig(
        f"{args.output_dir}/neural_net.png",
        bbox_inches="tight",
        pad_inches=0.2,
    )


if __name__ == "__main__":
    main()
