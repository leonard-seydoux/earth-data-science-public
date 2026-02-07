import matplotlib.pyplot as plt
import numpy as np

import utils


def main():

    # Arguments
    args = utils.parse_args()

    utils.xkcd_style()
    fig, ax = plt.subplots(figsize=(4, 5))

    # Data
    x = np.linspace(0, 10, 400)
    y1 = np.exp(-0.1 * x) + 0.05 * np.random.randn(len(x))
    y2 = np.exp(-0.5 * x) + 0.05 * np.random.randn(len(x))
    y2 += np.exp(0.1 * x) / 2
    y3 = np.exp(-0.5 * x) + 0.05 * np.random.randn(len(x))

    ax.plot(x, y1, label="too slow")
    ax.plot(x, y2, label="diverging!")
    ax.plot(x, y3, label="just right")
    ax.set_xlim(0, 10)
    ax.set_xlabel("epoch")
    ax.set_ylabel("testing loss")
    ax.legend(loc="lower center", frameon=False, bbox_to_anchor=(0.5, 1))
    ax.set_xticks([])
    ax.set_yticks([])

    # Save
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_curve.png")


if __name__ == "__main__":
    main()
