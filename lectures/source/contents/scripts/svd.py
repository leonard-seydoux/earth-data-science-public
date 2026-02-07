"""K-means clustering algorithm visualization in GIF."""

import matplotlib.pyplot as plt
import numpy as np

import utils

utils.xkcd_style()


def main():

    # Arguments
    args = utils.parse_args()

    # Matrix
    X = np.random.rand(6, 3)
    U, D, VT = np.linalg.svd(X, full_matrices=True)

    utils.xkcd_style()
    fig, ax = plt.subplots(
        ncols=4,
        figsize=(6, 2.5),
        width_ratios=[2, 2, 1, 1],
        gridspec_kw={"wspace": 0.5},
    )

    # Plot
    ax[0].matshow(X.T, cmap="Blues", vmin=0, vmax=1)
    ax[1].matshow(U, cmap="Blues", vmin=-1, vmax=1)
    ax[2].matshow(np.diag(D), cmap="Blues", vmin=0, vmax=1)
    ax[3].matshow(VT.T, cmap="Blues", vmin=-1, vmax=1)
    ax[0].set_title("X (nxm)", weight="bold")
    ax[1].set_title("U (nxn)", weight="bold")
    ax[2].set_title("S (mxm)", weight="bold")
    ax[3].set_title("V (mxm)", weight="bold")

    # Text math suymbols between
    ax[0].text(1.1, 0.5, "=", transform=ax[0].transAxes)
    ax[0].text(1.1, 0.5, "x", transform=ax[1].transAxes, weight="normal")
    ax[0].text(1.2, 0.5, "x", transform=ax[2].transAxes, weight="normal")

    # Remove ticks
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    fig.savefig(f"{args.output_dir}/svd.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
