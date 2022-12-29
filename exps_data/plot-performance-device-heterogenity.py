from cProfile import run
import matplotlib.pyplot as plt
import numpy as np
import sys

with open("sys",'r') as f:
    sys=f.read()

# target acc = 95%

# 20news runtime on various device.
runtime_20news = {'tx2': [43.95175, 12.744250000000001, 18.5163622222222, 5.160140000000001, 1.341133333333335], 'nano': [46.2935, 15.086, 21.21316888888888, 8.137280000000002, 2.493722222222222], 'rpi': [88.445, 57.2375, 69.75568888888883, 61.72579999999994, 23.240322222222233]}

# agnews runtime on various device.
runtime_agnews = {'tx2': [73.075, 19.055, 17.547172222222205, 3.1753944444444464, 1.0431916666666652], 'nano': [74.12333333333333, 20.10333333333333, 18.28812222222224, 4.003344444444441, 1.6923166666666678], 'rpi': [92.99333333333334, 38.973333333333336, 31.625222222222224, 18.906444444444443, 13.376566666666651]}

# semeval runtime on various device.
runtime_semeval = {'tx2': [89.92416666666666, 23.129166666666666, 46.779922222222254, 10.956977777777778, 1.1349333333333311], 'nano': [90.78833333333333, 23.993333333333332, 47.490733333333296, 11.593900000000009, 1.878933333333337], 'rpi': [106.34333333333333, 39.54833333333333, 60.285333333333334, 23.058500000000002, 15.270933333333303]}

# onto runtime on various device.
runtime_onto ={'tx2': [55.86388888888889, 15.53138888888889, 43.82791111111111, 12.86774444444445, 5.825194444444443], 'nano': [57.95111111111111, 17.61861111111111, 46.375600000000034, 15.415433333333322, 10.391194444444428], 'rpi': [95.52111111111111, 55.18861111111111, 92.23400000000001, 61.27383333333333, 92.57919444444443]}

def hybrid(runtime):
    runtime['hybrid'] =( np.array(runtime['tx2']) + np.array(runtime['nano']) + np.array(runtime['rpi']) )/ 3
    return runtime

runtime_20news = hybrid(runtime_20news)
runtime_agnews = hybrid(runtime_agnews)
runtime_semeval = hybrid(runtime_semeval)
runtime_onto = hybrid(runtime_onto)


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

    if type == "20news":
        data = runtime_20news
    if type == "semeval":
        data = runtime_semeval
    if type == "agnews":
        data = runtime_agnews
    if type == "onto":
        data = runtime_onto
    

    xlabels = ["tx2", "nano", "rpi",'hybrid']
    xlabels_fig = ["Jetson TX2", "Jetson Nano","RPI 4B", "Heterogeneous"]
    ax = [axes["A"], axes["B"], axes["C"], axes["D"]]

    for i in range(len(axes)):
        ax[i].set_xlabel(xlabels_fig[i], **label_font_conf)
        ax[i].set_xticks([])
        dataset = xlabels[i]  # video name
        fps = data[dataset]
        ax[i].bar(x, fps, width=bar_width, **bar_confs)

        ax[i].grid(axis="y", alpha=0.3)
        ax[i].set_xlim(min(x)-bar_width*1.5, max(x)+bar_width*1.5)

    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')
    # ax[2].set_yscale('log')
    ax[0].set_ylabel("Elapsed Training \nTime (hrs)", **label_font_conf)
    # https://matplotlib.org/stable/api/container_api.html#module-matplotlib.container
    bars = ax[0].containers[0].get_children()
    labels = ["FT", "FTQ", r"LF$_{oracle}$", r"LFQ$_{oracle}$", sys]
    ax[0].legend(bars, labels, ncol=5, loc="lower left", bbox_to_anchor=(0, 1),frameon=False,fontsize=15,columnspacing = 1.5,handletextpad=0.5)

    plt.subplots_adjust(wspace=2.5)
    # plt.show()
    plt.savefig('../figs/eval-performance-device-'+type+'.pdf', bbox_inches="tight")


if __name__ == '__main__':
    plot_fps_stream_number("20news")
    plot_fps_stream_number("agnews")
    plot_fps_stream_number("semeval")
    plot_fps_stream_number("onto")
