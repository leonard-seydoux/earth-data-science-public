"""Reinforcement learning diagram script."""

import matplotlib.pyplot as plt

import utils


def main():

    # Arguments
    args = utils.parse_args()

    # Figure
    utils.xkcd_style()
    fig, ax = plt.subplots(figsize=(3, 3))

    # Labels
    ax.text(0.5, 0.2, "agent", ha="center", va="top")
    ax.text(0.5, 0.8, "environment", ha="center")
    ax.text(0.31, 0.52, "action", va="center", ha="right", weight="normal")
    ax.text(0.67, 0.52, "reward", va="center", ha="left", weight="normal")

    # Agent patch
    utils.space_invaders(ax, facecolor="C1", center=(0.5, 0.3), size=0.17)

    # Arrows
    arrowprops = dict(
        color="k",
        lw=1.5,
        shrinkA=10,
        shrinkB=25,
        mutation_scale=15,
    )
    ax.annotate(
        "",
        xy=(0.57, 0.2),
        xytext=(0.58, 0.8),
        arrowprops={
            **arrowprops,
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=-0.22",
        },
    )
    ax.annotate(
        "",
        xy=(0.43, 0.2),
        xytext=(0.42, 0.8),
        arrowprops={
            **arrowprops,
            "arrowstyle": "<-",
            "connectionstyle": "arc3,rad=0.22",
        },
    )

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Save figure
    fig.savefig(f"{args.output_dir}/reinforcement_learning.png")


if __name__ == "__main__":
    main()
