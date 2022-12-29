import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.pyplot import MultipleLocator

runtime = {
    "tx2": [6.0316, 20, 210.74],
    "nano": [6.6079, 37, 222.5],
    "rpi": [6.632, 235.67, 359.7]
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
        "size": "38"
    }
    bar_confs = {
        "color": ["white", "white", "silver"],
        "linewidth": 1,
        "hatch": ["", "//", ""],
        "edgecolor": "black",
    }

    figure_mosaic = """
    AAA.BBB.CCC
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(9, 5.7), dpi=100)
    bar_width = 0.03
    
    x = [0.1, 0.1+bar_width*2, 0.1+bar_width*4]
    data = runtime

    xlabels = ["tx2", "nano", "rpi"]
    xlabels_fig = ["Jetson TX2", "Jetson Nano", "RPI 4B"]
    ax = [axes["A"], axes["B"], axes["C"]]

    for i in range(len(axes)):
        ax[i].yaxis.set_tick_params(labelsize=20)
        
        ax[i].set_xlabel(xlabels_fig[i], size=30,labelpad=30)
        # ax[i].set_ylabel(**label_font_conf)
        ax[i].set_xticks([])
        dataset = xlabels[i]  # video name
        fps = data[dataset]
        ax[i].bar(x, fps, width=bar_width, **bar_confs)

        ax[i].grid(axis="y", alpha=0.3)
        ax[i].set_xlim(min(x)-bar_width*1.5, max(x)+bar_width*1.5)

    y_major_locator=MultipleLocator(50)    
    ax[0].yaxis.set_major_locator(y_major_locator)
    y_major_locator=MultipleLocator(50)    
    ax[1].yaxis.set_major_locator(y_major_locator)
    y_major_locator=MultipleLocator(100)    
    ax[2].yaxis.set_major_locator(y_major_locator)

    ax[0].set_ylabel("Elapsed Training\nTime (hrs)", **label_font_conf)
    # https://matplotlib.org/stable/api/container_api.html#module-matplotlib.container
    bars = ax[0].containers[0].get_children()
    labels = ["CV1", "CV2", "NLP"]
    ax[0].legend( bars, labels, fontsize=30,ncol=3, loc="lower left", bbox_to_anchor=(0, 0.95),frameon=False,columnspacing = 1.0,handletextpad=0.3)

    plt.subplots_adjust(wspace=2.5)
    # plt.show()
    plt.savefig('../figs/motivation-convergence.pdf', bbox_inches="tight")


if __name__ == '__main__':
    plot_fps_stream_number("offline")
