"""Illustration of a confusion matrix."""

import matplotlib.pyplot as plt
import numpy as np

import utils


def main():

    # Arguments
    args = utils.parse_args()

    # Generate data
    confusion_matrix = np.eye(2)
    labels = ["positive", "negative"]
    cases = ["true", "false"]

    # Initialize plot
    fig, ax = utils.square_canvas(margin=0.22)

    # Show matrix
    cmap = utils.cycler_cmap(2).reversed()
    ax.matshow(confusion_matrix, cmap=cmap, alpha=0.6)

    # Annotate
    for column, label in enumerate(labels):
        for row, case in enumerate(cases):
            ax.text(row, column, f"{case}\n{label}s", va="center", ha="center")
        cases = cases[::-1]

    # Labels
    ax.set_xlabel("observed")
    ax.set_ylabel("predicted")
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels, rotation=90, va="center")
    ax.xaxis.set_ticks_position("bottom")

    # Save
    fig.savefig(args.output_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
