import matplotlib.pyplot as plt
import numpy as np
import sys

with open("sys",'r') as f:
    sys=f.read()

# energy consumption (KJ)
energy = {
 "20news": [700.3475999999999,
  250.9596,
  344.303648,
  160.04764799999992,
  52.506880000000066],
 "agnews": [1082.472, 304.584, 274.0186399999999, 69.57064000000004, 33.716760000000036],
 "semeval":  [1824.636, 494.868, 1098.153440000001, 270.11103999999983, 73.13740000000013],
 "onto": [1177.512,
  441.74760000000003,
  1325.6281600000007,
  480.7281599999993,
  376.3580000000005]
}

energy_normalized = {
 "20news": np.array([700.3475999999999,
  250.9596,
  344.303648,
  160.04764799999992,
  52.506880000000066]) / 52.506880000000066,

 "agnews": np.array([1082.472,
  304.584,
  236.48184000000037,
  54.94983999999992,
  33.716760000000036]) / 33.716760000000036,
 "semeval": np.array([1824.636, 494.868, 1098.153440000001, 270.11103999999983, 73.13740000000013]) / 73.13740000000013,
 "onto": np.array([1177.512,
  441.74760000000003,
  1325.6281600000007,
  480.7281599999993,
  376.3580000000005]) / 376.3580000000005
}


def plot_fps_stream_number(type):
    """
        plotting max stream numbers (online) or fps (offline) for each video
        type: 
            - online: plotting [libx264 (SoC-CPU), mediacodec(SoC-HW), libx264 (CPU), nvenc (GPU)]
            - offline: plotting [libx264 (SoC-CPU), libx264 (CPU), nvenc (GPU)]
    """
    label_font_conf = {
        # "weight": "bold",
        "size": "15"
    }
    bar_confs = {
        "color": ["white", "white", "silver", "grey", "black"],
        "linewidth": 1,
        "hatch": ["", "//", "", "//", ""],
        "edgecolor": "black",
    }

    figure_mosaic = """
    AAAAA.BBBBB.CCCCC.DDDDD
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(9, 2), dpi=100)
    bar_width = 0.03

    x = [0.1, 0.1+bar_width*2, 0.1+bar_width*4, 0.1+bar_width*6, 0.1+bar_width*8]
    data = energy_normalized

    xlabels = ["20news", "agnews", "semeval", "onto"]
    xlabels_fig = ["20NEWS", "AGNEWS", "SEMEVAL", "ONTONOTES"]
    ax = [axes["A"], axes["B"], axes["C"], axes["D"]]

    for i in range(len(axes)):
        ax[i].set_xlabel(xlabels_fig[i], **label_font_conf)
        ax[i].set_xticks([])
        dataset = xlabels[i]  # video name
        fps = data[dataset]
        ax[i].bar(x, fps, width=bar_width, **bar_confs)

        ax[i].grid(axis="y", alpha=0.3)
        ax[i].set_xlim(min(x)-bar_width*1.5, max(x)+bar_width*1.5)

        # # tag value on the last bar.
        # rect = ax[i].patches
        # height = rect[-1].get_height()
        # absolute_value = round(energy[dataset][-1] * 5 / 100, 1)
        # if i == 0:
        #     ax[i].text(x[-1]-0.025, height+0.2,absolute_value) 
        # elif i == 1:
        #     ax[i].text(x[-1]-0.025, height+0.5,absolute_value) 
        # elif i == 2:
        #     ax[i].text(x[-1]-0.025, height+0.30,absolute_value) 
        # elif i == 3:
        #     absolute_value = round(energy[dataset][-1] * 5 / 600, 1)
        #     ax[i].text(x[-1]-0.025, height+0.04,absolute_value) 

    
    ax[0].set_ylabel(r"Normalized Energy ($\times$)", **label_font_conf)
    # https://matplotlib.org/stable/api/container_api.html#module-matplotlib.container
    bars = ax[0].containers[0].get_children()
    labels = ["FT", "FTQ", r"LF$_{oracle}$", r"LFQ$_{oracle}$", sys]
    ax[0].legend(bars, labels, ncol=5, loc="lower left", bbox_to_anchor=(0, 1),frameon=False,fontsize=15,columnspacing = 1.5,handletextpad=0.5)

    plt.subplots_adjust(wspace=2.5)
    # plt.show()
    plt.savefig('../figs/eval-energy.pdf', bbox_inches="tight")


if __name__ == '__main__':
    plot_fps_stream_number("offline")
    print(energy_normalized)
