import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA

import utils


def main():

    # Arguments
    args = utils.parse_args()

    # Data and model
    np.random.seed(42)
    x1 = np.random.uniform(-3, 3, size=(30,))
    x2 = x1 + 0.8 * np.random.normal(0, 0.7, size=x1.shape)
    X = np.vstack([x1, x2]).T
    model = PCA(n_components=1)
    reduced = model.fit_transform(X)

    # Figure
    fig, ax = utils.square_canvas(right=0.8)
    ax = [ax, fig.add_axes([0.95, 0.1, 0, 0.8])]

    # Show scatter plot
    utils.scatter_samples(ax[0], X)
    ax[1].plot(
        np.zeros_like(reduced),
        reduced,
        "o",
        color="C1",
        mec="k",
        clip_on=False,
        zorder=10,
    )

    # Labels
    ax[0].set_xlabel("feature x₁")
    ax[0].set_ylabel("feature x₂")
    ax[1].set_ylabel("latent variable z")

    # Annotate
    z_continuous = np.linspace(reduced.min(), reduced.max(), 100)
    ax[0].plot(
        *model.inverse_transform(z_continuous[:, np.newaxis]).T,
        color=utils.brighter("C5", -1),
        lw=2,
        zorder=0,
    )

    # Arrow between axes
    ax[0].annotate(
        "",
        xy=(1.12, 0.5),
        xytext=(0.7, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=utils.DEFAULT_ARROWPROPS,
    )

    # Collapse all samples to their projections
    for i in range(X.shape[0]):
        x_orig = X[i : i + 1]
        x_proj = model.inverse_transform(reduced[i : i + 1])
        ax[0].annotate(
            "",
            xy=(x_orig[0, 0], x_orig[0, 1]),
            xytext=(x_proj[0, 0], x_proj[0, 1]),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-",
                lw=0.8,
                color=utils.brighter("C5", -1),
                connectionstyle="arc3,rad=0.0",
            ),
            zorder=0,
        )

    # Labels
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    lim_min = min(ax[0].get_xlim()[0], ax[0].get_ylim()[0])
    lim_max = max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])
    ax[0].set_xlim(lim_min, lim_max)
    ax[0].set_ylim(lim_min, lim_max)
    ax[1].set_ylim(reduced.min() - 0.5, reduced.max() + 0.5)

    # Save figure
    fig.savefig(args.output_dir / "dimensionality_reduction.png")
    fig.savefig(args.output_dir / "dimensionality_reduction.svg")


if __name__ == "__main__":
    main()
