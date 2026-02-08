"""Reinforcement learning"""

import utils


def main():

    # Arguments
    args = utils.parse_args()

    # Figure
    fig, ax = utils.square_canvas()

    # Labels
    ax.text(0.5, 0.2, "agent", ha="center", va="top")
    ax.text(0.5, 0.8, "environment", ha="center")
    ax.text(0.31, 0.52, "reward", va="center", ha="right")
    ax.text(0.67, 0.52, "action", va="center", ha="left")

    # Agent patch
    utils.space_invaders(ax, facecolor="C1", center=(0.5, 0.3), size=0.17)

    # Arrows
    ax.annotate(
        "",
        xy=(0.58, 0.75),
        xytext=(0.58, 0.35),
        arrowprops=utils.DEFAULT_ARROWPROPS,
    )
    ax.annotate(
        "",
        xy=(0.42, 0.35),
        xytext=(0.42, 0.75),
        arrowprops=utils.DEFAULT_ARROWPROPS,
    )

    # Save figure
    fig.savefig(args.output_dir / "reinforcement_learning.png")


if __name__ == "__main__":
    main()
