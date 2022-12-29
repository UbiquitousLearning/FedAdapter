import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib as mpl

with open("sys",'r') as f:
    sys=f.read()

runtime_TX2 = {'20news': [32.3855, 6.147444444444437, 4.916249999999992, 1.4725000000000026],
 'agnews': [73.075, 7.255555555555534, 3.172222222222234, 1.547222222222225],
 'semeval': [124.32138888888889,
  2.9641111111111176,
  1.8330000000000028,
  1.6980000000000024],
 'onto': [72.29444444444445,
  8.484444444444428,
  8.484444444444428,
  8.484444444444428]}

runtime_RPI = {'20news': [88.445, 106.98044444444437, 96.51999999999994, 27.645000000000035],
 'agnews': [92.99333333333334,
  92.75555555555546,
  55.422222222222416,
  22.922222222222242],
 'semeval': [147.0211111111111,
  29.181333333333374,
  25.87466666666669,
  23.54966666666668],
 'onto': [130.0988888888889,
  139.58444444444416,
  139.58444444444416,
  139.58444444444416]}

runtime_TX2 = {'20news': [43.95175,
  6.147444444444437,
  4.916249999999992,
  1.4725000000000026],
 'agnews': [73.075, 7.255555555555534, 3.172222222222234, 1.547222222222225],
 'semeval': [124.32138888888889,
  2.676333333333339,
  1.578416666666668,
  1.4621666666666675],
 'onto': [76.08611111111111,
  8.484444444444428,
  8.484444444444428,
  8.484444444444428]}

runtime_hybrid = {'20news+TX2': [43.95175,
  6.147444444444437,
  4.916249999999992,
  1.4725000000000026],
 'agnews+TX2': [73.075, 7.255555555555534, 3.172222222222234, 1.547222222222225],
 '20news+RPI': [88.445, 106.98044444444437, 96.51999999999994, 27.645000000000035],
 'agnews+RPI': [92.99333333333334,
  92.75555555555546,
  55.422222222222416,
  22.922222222222242]}

def plot_fps_stream_number(type):
    """
        plotting max stream numbers (online) or fps (offline) for each video
        type: 
            - online: plotting [libx264 (SoC-CPU), mediacodec(SoC-HW), libx264 (CPU), nvenc (GPU)]
            - offline: plotting [libx264 (SoC-CPU), libx264 (CPU), nvenc (GPU)]
    """
    label_font_conf = {
        # "weight": "bold",
        "size": "14"
    }
    bar_confs = {
        "color": ["white", "white", "silver", "grey"],
        "linewidth": 1,
        "hatch": ["", "//", "", "//"],
        "edgecolor": "black",
    }

    figure_mosaic = """
    AAAA.BBBB.CCCC.DDDD
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(9, 2), dpi=100)
    bar_width = 0.03

    x = [0.1, 0.1+bar_width*2, 0.1+bar_width*4, 0.1+bar_width*6]
    data = runtime_hybrid

    xlabels = ["20news+TX2", "agnews+TX2", "20news+RPI", "agnews+RPI"]
    xlabels_fig = ["20NEWS+TX2", "AGNEWS+TX2", "20NEWS+RPI", "AGNEWS+RPI"]
    ax = [axes["A"], axes["B"], axes["C"], axes["D"]]

    for i in range(len(axes)):
        ax[i].set_xlabel(xlabels_fig[i], **label_font_conf)
        ax[i].set_xticks([])
        dataset = xlabels[i]  # video name
        fps = data[dataset]
        ax[i].bar(x, fps, width=bar_width, **bar_confs)

        ax[i].grid(axis="y", alpha=0.3)
        ax[i].set_xlim(min(x)-bar_width*1.5, max(x)+bar_width*1.5)

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].set_ylabel("Elapsed Training \nTime (hrs)", **label_font_conf)
    # https://matplotlib.org/stable/api/container_api.html#module-matplotlib.container
    bars = ax[0].containers[0].get_children()
    labels = ["FT", "Adapter", r"Adapter$_{Oracle}$", sys + r" (Cache+Adapter$_{Oracle}$)"]
    ax[0].legend(bars, labels, ncol=4, loc="lower left", bbox_to_anchor=(-0.6, 1),frameon=False,fontsize=15,columnspacing = 1.0,handletextpad=0.3)

    plt.subplots_adjust(wspace=2.5)
    # plt.show()
    plt.savefig('../figs/eval-ablation-adapter-cache-hybrid.pdf', bbox_inches="tight")


if __name__ == '__main__':
    plot_fps_stream_number("offline")
