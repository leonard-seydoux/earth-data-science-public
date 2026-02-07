import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

import plot


def accuracy_matrix():
    """Plot accuracy matrix with annotations."""
    accuracy = np.array([[10, 3], [2, 8]])
    total = accuracy.sum()
    labels = ["positive", "negative"]
    cases = ["true", "false"]
    return accuracy, total, labels, cases


def main():

    # Generate data
    accuracy, total, labels, cases = accuracy_matrix()

    # Initialize plot
    plot.xkcd_style()
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    # Show matrix
    ax.matshow(accuracy, cmap="Greys")

    # Anotate cells
    for i, label in enumerate(labels):
        for j, case in enumerate(cases):
            accuracy_value = accuracy[i, j]
            color = "white" if accuracy_value > total / 4 else "black"
            display = f"{case}\n{label.lower()}s"
            ax.text(
                j,
                i,
                display,
                va="center",
                ha="center",
                color=color,
                weight="normal",
            )
        cases = cases[::-1]

    # Labels
    ax.set_xlabel("observed")
    ax.set_ylabel("predicted")
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels, rotation=90, va="center")
    ax.xaxis.set_ticks_position("bottom")

    fig.savefig("contents/figures/accuracy.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
