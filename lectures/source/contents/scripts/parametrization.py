"""Illustration of parametrization over training."""

import numpy as np

import utils


def main():

    # Parse
    args = utils.parse_args()

    # Figure
    fig, ax1, ax2 = utils.vertical_squares_canvas()

    # Train/test versus number of samples
    samples = np.linspace(0, 1, 100)
    train = 1 - np.exp(-5 * samples)
    test = np.exp(-5 * samples) + 1.2
    ax1.plot(samples, train)
    ax1.plot(samples, test)
    ax1.set_xlabel("number of samples")
    ax1.set_ylabel("error")
    ax1.set_aspect("auto")
    ax1.text(
        0.95,
        0.42,
        "train",
        ha="right",
        va="top",
        transform=ax1.transAxes,
        c="C0",
    )
    ax1.text(
        0.95,
        0.57,
        "test",
        ha="right",
        va="bottom",
        transform=ax1.transAxes,
        c="C1",
    )

    # Parametrization
    params = np.linspace(0, 1, 100)
    train = np.exp(-5 * params) + 0.2
    test = np.exp(-5 * params) + 0.3 + 0.5 * params**2
    ax2.plot(params, train)
    ax2.plot(params, test)
    ax2.set_xlabel("number of parameters")
    ax2.set_ylabel("error")
    ax2.set_aspect("auto")
    ax2.text(
        0.95,
        0.07,
        "train",
        ha="right",
        va="bottom",
        transform=ax2.transAxes,
        c="C0",
    )
    ax2.text(
        0.95,
        0.55,
        "test",
        ha="right",
        va="bottom",
        transform=ax2.transAxes,
        c="C1",
    )

    # Save
    fig.savefig(args.output_dir / "parametrization.png")


if __name__ == "__main__":
    main()
