import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import numpy as np

import plot

equilateral_triangle = (
    np.array(
        [
            [0, np.sqrt(3) / 1.5],
            [-1, -np.sqrt(3) / 3],
            [1, -np.sqrt(3) / 3],
            [0, np.sqrt(3) / 1.5],
        ]
    )
    / 3
)


def main():

    plot.xkcd_style()

    fig, ax = plt.subplots(figsize=(3, 3))

    # Circles
    radius = 0.3
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors_light = [mcolor.to_rgba(c, alpha=0.1) for c in colors]

    circle1 = plt.Circle(
        equilateral_triangle[0],
        radius,
        facecolor=colors_light[0],
        edgecolor="k",
    )
    circle2 = plt.Circle(
        equilateral_triangle[1],
        radius,
        facecolor=colors_light[1],
        edgecolor="k",
    )
    circle3 = plt.Circle(
        equilateral_triangle[2],
        radius,
        facecolor=colors_light[3],
        edgecolor="k",
    )

    # Tetx
    ax.text(
        *equilateral_triangle[0],
        "experience",
        fontsize=12,
        ha="center",
        va="center",
    )
    ax.text(
        *equilateral_triangle[1],
        "task",
        fontsize=12,
        ha="center",
        va="center",
    )
    ax.text(
        *equilateral_triangle[2],
        "performance",
        fontsize=12,
        ha="center",
        va="center",
    )

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.margins(7)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Arrows
    arrow_props = dict(
        arrowstyle="<->",
        linewidth=1.2,
        color="k",
        connectionstyle="arc3,rad=0.5",
        shrinkA=20,
        shrinkB=20,
        mutation_scale=10,
    )
    ax.annotate(
        "",
        xy=equilateral_triangle[1],
        xytext=equilateral_triangle[0],
        arrowprops=arrow_props,
    )
    ax.annotate(
        "",
        xy=equilateral_triangle[2],
        xytext=equilateral_triangle[1],
        arrowprops=arrow_props,
    )
    ax.annotate(
        "",
        xy=equilateral_triangle[0],
        xytext=equilateral_triangle[2],
        arrowprops=arrow_props,
    )

    # Annotations
    fig.savefig("contents/figures/exp_task_perf_venn.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
