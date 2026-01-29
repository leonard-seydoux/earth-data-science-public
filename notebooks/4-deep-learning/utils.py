"""Plot utils for the practicals with SeisBench."""

import matplotlib.pyplot as plt
import numpy as np

PHASES = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}

COLORS = {"P": "C0", "S": "C1"}


def plot_waveforms(waveforms, metadata, phases=PHASES, ax=None):
    # Prepare figure
    ax = ax or plt.gca()

    # Normalize waveforms
    waveforms = 0.8 * waveforms / np.max(np.abs(waveforms))

    # Labels to info
    sampling_rate = metadata["trace_sampling_rate_hz"]
    times = np.arange(waveforms.shape[1]) / sampling_rate
    component = metadata["trace_channel"]
    channels = [
        component.replace("*", ch) for ch in metadata["trace_component_order"]
    ]

    # Plot waveforms
    for i, waveform in enumerate(waveforms):

        # Plot waveform
        ax.plot(times, waveform + i, color="k", linewidth=0.5)

    # Common labels
    ax.grid()
    ax.set_yticks(range(len(channels)), channels)
    ax.set_ylabel("Normalized seismograms")
    ax.set_xlabel("Time (seconds)")


def add_picks(labels, phases=PHASES, ax=None):
    # Prepare figure
    ax = ax or plt.gca()

    # Time
    sampling_rate = labels["trace_sampling_rate_hz"]

    # Plot picked arrivals
    if labels is not None:
        for phase_key, phase_name in phases.items():
            if phase_key in labels:
                ax.axvline(
                    labels[phase_key] / sampling_rate,
                    label=rf"${phase_name}$",
                    color=COLORS[phase_name],
                )

    # Common labels
    ax.legend(loc="upper right", title="Picked arrival")

    return ax


def plot_waveforms_and_labels(
    waveforms, labels_true, labels, metadata=None, phases=PHASES
):
    """Plot waveforms in a single figure.

    Parameters
    ----------
    waveforms : list of numpy.ndarray
        Waveforms to plot.
    labels : list of str
        Labels for the waveforms.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    """
    # Prepare figure
    gs = dict(height_ratios=[2, 1, 1])
    _, ax = plt.subplots(3, sharex=True, dpi=300, gridspec_kw=gs)

    # Plot waveform
    plot_waveforms(waveforms, metadata, phases=phases, ax=ax[0])

    # Plot labels
    times = np.arange(waveforms.shape[1]) / metadata["trace_sampling_rate_hz"]
    for (label, name) in zip(labels_true, "PSN"):
        ax[1].plot(times, label, label=name)
    for (label, name) in zip(labels, "PSN"):
        ax[2].plot(times, label, label=name)

    # Labels
    ax[0].set_xlabel("")
    ax[2].set_xlabel("Time (seconds)")
    ax[1].grid()
    ax[2].grid()
    ax[1].legend()

    return ax
