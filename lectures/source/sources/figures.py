"""Module for creating figures for the lecture."""

import os

from matplotlib import pyplot as plt


def savefig(filepath, figure):
    """Create a directory for figures."""
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    print(f"Saving figure to {filepath}")
    figure.savefig(filepath + ".png", format="png")


def saveani(filepath, animation):
    """Create a directory for animations."""
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    print(f"Saving animation to {filepath}")
    animation.save(
        filepath,
        writer="ffmpeg",
        fps=6,
        savefig_kwargs={"transparent": True},
        dpi=300,
        # bitrate=1800,
    )


def linear_regression(margins=0.1):
    """Create axes for half the width of a slide."""
    # Generate axes
    fig, ax = plt.subplots(
        ncols=2, figsize=(4.2, 2.1), constrained_layout=True
    )

    # Set limits
    limits = -margins, 1 + margins
    ax[0].set(xticks=[], yticks=[], xlim=limits, ylim=limits)
    ax[1].set(xticks=[], yticks=[])
    ax[1].margins(margins)

    # Labels
    ax[0].set(xlabel="Feature $x$", ylabel="Target $y$", title="Data space")
    ax[1].set(
        xlabel=r"$a = \theta_1$", ylabel=r"$b = \theta_2$", title="Model space"
    )

    return fig, ax
