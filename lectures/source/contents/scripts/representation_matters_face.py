import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

import utils

args = utils.parse_args()

# Read face from deep learning book
img = plt.imread(f"{args.data_dir}/deep-learning-figure-x-x-1.jpg")

# Create figure with 3x3 subplots
fig, ax = plt.subplots(3, 3, figsize=(4, 4))

# Create GIF with random shuffles
writer = PillowWriter(fps=10)
n_frames = 40

with writer.saving(
    fig, f"{args.output_dir}/representation_matters_face.gif", dpi=150
):
    for frame in range(n_frames):
        for i in range(3):
            for j in range(3):
                ax[i, j].clear()
                ax[i, j].set_axis_off()

                img_shuffle = np.copy(img)
                np.random.seed(frame * 9 + i * 3 + j)
                np.random.shuffle(img_shuffle.reshape(-1, 3))

                ax[i, j].imshow(img_shuffle)

        writer.grab_frame()
