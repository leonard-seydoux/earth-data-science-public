import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import utils

np.random.seed(42)


def main():

    # Arguments
    args = utils.parse_args()

    # Figure
    fig, ax = plt.subplots(
        figsize=(6, 5),
        nrows=2,
        ncols=2,
        gridspec_kw={"hspace": 0.3, "wspace": 0.2},
        sharex=True,
        sharey=True,
    )

    t = np.linspace(0, 10, 800)

    # Long-period
    y = np.random.normal(size=t.shape)
    start = 3
    b, a = signal.butter(2, 0.4)
    y = signal.filtfilt(b, a, y)
    b, a = signal.butter(4, 0.1, btype="highpass")
    y = signal.filtfilt(b, a, y)
    y[t < start] /= 10
    y[t > start] *= np.exp(-0.5 * (t[t > start] - start))
    y = (y - y.min()) / (y.max() - y.min())
    y -= y.mean()

    ax[0, 0].plot(t, y, color="C0")

    # Volcano-tectonic
    y = np.random.normal(size=t.shape)
    b, a = signal.butter(4, 0.4)
    y = signal.filtfilt(b, a, y)
    b, a = signal.butter(4, 0.3, btype="highpass")
    y = signal.filtfilt(b, a, y)
    start = 3
    y[t < start] /= 10
    y[t > start] *= 10 * np.exp(-4 * (t[t > start] - start))
    y += np.random.normal(size=t.shape) * 0.1
    y = (y - y.min()) / (y.max() - y.min())
    y -= y.mean()
    ax[0, 1].plot(t, y, color="C1")

    # Tremor
    y = np.random.normal(size=t.shape)
    b, a = signal.butter(4, 0.2)
    y = signal.filtfilt(b, a, y)
    b, a = signal.butter(4, 0.1, btype="highpass")
    y = signal.filtfilt(b, a, y)
    y = (y - y.min()) / (y.max() - y.min()) - 0.5
    y -= y.mean()
    y = signal.windows.tukey(len(y), 0.2) * y
    ax[1, 0].plot(t, y, color="C2")

    # Very long-period
    y = np.random.normal(size=t.shape)
    start = 3
    y[t < start] /= 10
    y[t > start] *= np.exp(-0.5 * (t[t > start] - start))
    b, a = signal.butter(4, 0.1)
    y = signal.filtfilt(b, a, y)
    b, a = signal.butter(4, 0.05, btype="highpass")
    y = signal.filtfilt(b, a, y)
    y = (y - y.min()) / (y.max() - y.min())
    y -= y.mean()
    ax[1, 1].plot(t, y, color="C3")

    # Titles
    ax[0, 0].set_title("long-period")
    ax[0, 1].set_title("volcano-tectonic")
    ax[1, 0].set_title("tremor")
    ax[1, 1].set_title("very long-period")

    # Remove ticks
    for a in ax.flatten():
        a.set_xticks([])
        a.set_yticks([])
        a.set_ylim(-0.7, 0.7)

    # Save
    fig.savefig(args.output_dir / "notebook_2.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
