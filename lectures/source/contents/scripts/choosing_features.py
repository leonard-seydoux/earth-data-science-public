import matplotlib.pyplot as plt
import utils


def main():

    # Arguments
    args = utils.parse_args()

    # Figure
    fig, ax = plt.subplots(figsize=(7, 7))

    # Names
    names = [
        ["input", "hand-crafted\nfeatures", "output"],
        ["input", "hand-crafted\nfeatures", "mapping", "output"],
        ["input", "features", "mapping", "output"],
        [
            "input",
            "simple\nfeatures",
            "abstract\nfeatures",
            "mapping",
            "output",
        ],
    ]

    # Large rectangle for representation learning
    rectangle = plt.Rectangle(
        [1.9, -0.1],
        width=2,
        height=4.7,
        facecolor="C5",
        edgecolor="k",
        lw=1.5,
    )
    ax.add_patch(rectangle)
    ax.text(
        2.9,
        4.7,
        "representation learning",
        ha="center",
        va="bottom",
    )

    # Positions
    for column in range(4):
        for row in range(5):

            # Check if the cell is empty
            if row >= len(names[column]):
                continue

            facecolor = (
                "white"
                if names[column][row]
                in ["input", "output", "hand-crafted\nfeatures"]
                else "C3"
            )

            rectangle = plt.Rectangle(
                [column, row],
                width=0.8,
                height=0.5,
                facecolor=facecolor,
                edgecolor="k",
            )

            ax.text(
                column + 0.4,
                row + 0.25,
                names[column][row],
                ha="center",
                va="center",
            )
            ax.add_patch(rectangle)

    # Arrows
    for column in range(4):
        for row in range(5):

            # Check if the cell is empty
            if row + 1 >= len(names[column]):
                continue

            ax.annotate(
                "",
                xy=(column + 0.4, row + 0.5),
                xytext=(column + 0.4, row + 1),
                arrowprops=dict(
                    arrowstyle="<-", color="k", lw=1.5, shrinkA=4, shrinkB=6
                ),
            )

    # Limits
    ax.set_xlim(-0.11, 4)
    ax.set_ylim(-0.11, 5)
    ax.set_axis_off()

    # Save
    fig.tight_layout(pad=0.11)
    fig.savefig(args.output_dir / "choosing_features.png")


if __name__ == "__main__":
    main()
