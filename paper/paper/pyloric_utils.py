import numpy as np
import matplotlib.pyplot as plt


def compare_voltage_low_and_high_energy_trace(out_target, t, figsize, offset=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    iii = 0
    time_len = int(3 * 1000 / 0.025)  # 3 seconds
    if offset is None:
        offset = [164000, 149000]
    print("Showing :  ", time_len / 40000, "seconds")
    print("Scalebar indicates:  50mV")

    current_col = 0
    Vx = out_target["voltage"]
    axV = ax
    for j in range(3):
        if time_len is not None:
            axV.plot(
                t[:time_len:5] / 1000,
                Vx[j, 10000 + offset[iii] : 10000 + offset[iii] + time_len : 5]
                + 130.0 * (2 - j),
                linewidth=0.6,
                c="k",
            )
        else:
            axV.plot(t / 1000, Vx[j] + 120.0 * (2 - j), lw=0.6, c="k")
        current_col += 1

    box = axV.get_position()

    axV.set_position([box.x0, box.y0, box.width, box.height])

    axV.spines["right"].set_visible(False)
    axV.spines["top"].set_visible(False)
    axV.set_yticks([])
    # if iii == 0:
    #     axV.set_ylabel("Voltage")
    # axV.set_xlabel("Time (seconds)")
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # axV.set_ylabel("Voltage")
    axV.set_ylim([-90, 320])
    axV.set_xlim([-0.1, (t[:time_len:5] / 1000)[-1] + 0.2])
    axV.set_xticks([])

    # scale bar
    end_val_x = (t[:time_len:5] / 1000)[-1] + 0.15
    axV.plot([end_val_x, end_val_x], [260, 310], c="k")

    plt.subplots_adjust(wspace=0.1)
