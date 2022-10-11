import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_traces(traces, figsize=None, protocol=None, num_traces=10):
    if protocol is None:
        protocol = [
            "bAP.soma.v",
            "bAP.dend1.v",
            "bAP.dend2.v",
            "Step3.soma.v",
            "Step2.soma.v",
            "Step1.soma.v",
        ]
    if figsize is None:
        figsize = (12, 0.7 * num_traces)

    traces = traces[:num_traces]

    fig, ax = plt.subplots(num_traces + 1, len(protocol), figsize=figsize)
    xo_trace = return_xo(summstats=False)[0]
    for i, p in enumerate(protocol):
        _ = ax[0, i].plot(
            xo_trace[p].response["time"].to_numpy(),
            xo_trace[p].response["voltage"].to_numpy(),
            c="#fc4e2a",
            alpha=0.9,
        )
        ax[0, i].set_title(p)
    for trace_ind, t in enumerate(traces):
        for i, p in enumerate(protocol):
            traces_protocol = t[p].response["voltage"].to_numpy()
            time_protocol = t[p].response["time"].to_numpy()
            _ = ax[trace_ind + 1, i].plot(
                np.asarray(time_protocol).T,
                np.asarray(traces_protocol).T,
                c="k",
                alpha=0.8,
            )
    for i in range(num_traces):
        for j in range(len(protocol)):
            ax[i, j].spines["bottom"].set_visible(False)
            ax[i + 1, j].spines["left"].set_visible(False)
            ax[i, j].set_xticks([])
            ax[i + 1, j].set_yticks([])
            xo = xo_trace[protocol[j]].response["voltage"].to_numpy()
            ax[i + 1, j].set_ylim([np.min(xo) - 15.0, np.max(xo) + 15.0])
            ax[0, j].set_ylim([np.min(xo) - 15.0, np.max(xo) + 15.0])
