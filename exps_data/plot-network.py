import matplotlib.pyplot as plt
import numpy as np
import sys

with open("sys",'r') as f:
    sys=f.read()
    
# network traffic (G)
network = {
 "BERT": [107.7890625, 253.21875, 432.8671875, 159.1171875], # BERT
 "Quantize": [26.947265625, 63.3046875, 108.216796875, 39.779296875], # Quantize
 "Freeze": [30.75625, 50.990625, 212.46953125, 82.8328125], # Freeze
 "Q-Freeze": [4.061328125, 6.518359375, 29.50498046875, 11.616796875], # Q-Freeze
 "Ours": [0.6628515625000008, 1.3853906249999999, 14.794570312500031, 5.487656250000004] # Ours
} 

network = {
 "20news": [107.7890625, 26.947265625, 30.75625, 4.061328125, 0.6628515625000008],
 "agnews": [253.21875, 63.3046875, 50.990625, 6.518359375, 1.3853906249999999],
 "semeval": [432.8671875,
  108.216796875,
  212.46953125,
  29.50498046875,
  3.3683593749999923],
 "onto": [257.49609375,
  64.3740234375,
  273.07890625000005,
  33.713964843750006,
  8.922558593750017]
}

network_total = {
 "20news": np.array([146.28515625, 36.5712890625, 55.615625, 7.674609375, 0.6628515625000008])*15,
 "agnews": np.array([253.21875, 63.3046875, 59.084375, 8.252734375, 1.3853906249999999])*15,
 "semeval": np.array([432.8671875,
  108.216796875,
  256.24296875000005,
  27.821386718750002,
  3.3683593749999923])*15,
 "onto": np.array([257.49609375,
  64.3740234375,
  273.07890625000005,
  33.713964843750006,
  8.922558593750017])*15
}

network_normalized = {
 "20news": np.array([146.28515625, 36.5712890625, 55.615625, 7.674609375, 0.6628515625000008]) / 0.6628515625000008,
 "agnews": np.array([253.21875, 63.3046875, 59.084375, 8.252734375, 1.3853906249999999]) / 1.3853906249999999,
 "semeval": np.array([432.8671875,
  108.216796875,
  256.24296875000005,
  27.821386718750002,
  3.3683593749999923]) / 3.3683593749999923,
 "onto": np.array([257.49609375,
  64.3740234375,
  273.07890625000005,
  33.713964843750006,
  8.922558593750017]) / 8.922558593750017
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
    data = network_total

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
        # if i == 0:
        #     ax[i].text(x[-1]-0.025, height+0.1,round(network[dataset][-1],1)) 
        # elif i == 1:
        #     ax[i].text(x[-1]-0.025, height+0.1,round(network[dataset][-1],1)) 
        # elif i == 2:
        #     ax[i].text(x[-1]-0.025, height+0.1,round(network[dataset][-1],1)) 
        # elif i == 3:
        #     ax[i].text(x[-1]-0.025, height+0.1,round(network[dataset][-1],1)) 

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    ax[3].set_yscale('log')
    ax[0].set_ylabel(r"Network Traffic (GB)", **label_font_conf)
    # https://matplotlib.org/stable/api/container_api.html#module-matplotlib.container
    bars = ax[0].containers[0].get_children()
    labels = ["FT", "FTQ", r"LF$_{oracle}$", r"LFQ$_{oracle}$", sys]
    ax[0].legend(bars, labels, ncol=5, loc="lower left", bbox_to_anchor=(0, 1),frameon=False,fontsize=15,columnspacing = 1.5,handletextpad=0.5)

    plt.subplots_adjust(wspace=2.5)
    # plt.show()
    plt.savefig('../figs/eval-cost-network.pdf', bbox_inches="tight")


if __name__ == '__main__':
    plot_fps_stream_number("offline")
    print(network_normalized)
