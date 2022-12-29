import matplotlib.pyplot as plt
import numpy as np
import sys

# video fps
video_fps = {
    "V1": 30, 
    "V2": 30, 
    "V3": 59, 
    "V4": 25, 
    "V5": 29, 
    "V6": 30,
}

# CPU-8-cores
# throughput from cpulogs-5-streams, avg-power from [cpulogs-power-2347, cpulogs-power-2249]
online_CPU = {
    "V1": [444.8398577, 0.06409843333],
    "V2": [470.2194357, 0.08334993333],
    "V3": [216.7205407, 0.1117019492],
    "V4": [200.2563281, 0.15289276],
    "V5": [110.3500761, 0.2471533103],
    "V6": [45.79034129, 0.8077265333],
}
# avg-throughput, avg-power from [cpulogs-power-2347, cpulogs-power-2249, cpulogs-power-2306]
offline_CPU = {
    "V1": [36.54798518, 0.9090719778],
    "V2": [73.45890531, 0.3051944],
    "V3": [27.934918, 1.536498893],
    "V4": [32.95127293, 0.81558112],
    "V5": [5.119455738, 9.58378731],
    "V6": [1.402792594, 31.57559867],
}
online_GPU = {
    "V1": [268.9457327, 0.162944489],
    "V2": [272.2323049, 0.1737266267],
    "V3": [433.4835423, 0.1138715314],
    "V4": [219.0420561, 0.2222783189],
    "V5": [211.9056898, 0.2434526418],
    "V6": [146.9051972, 0.3894796489],
}
offline_GPU = {
    "V1": [160.72757, 0.0729802935],
    "V2": [130.2125836, 0.1253644373],
    "V3": [184.7820511, 0.1586442454],
    "V4": [71.29777335, 0.4291134984],
    "V5": [46.33314749, 0.8162243807],
    "V6": [17.31348214, 2.918498794],
}
# SoC
online_SoC_SW = {
    "V1": [221.5246193, 0.005882070491],
    "V2": [243.9045655, 0.003574111376],
    "V3": [138.5714052, 0.01453769927],
    "V4": [134.8071184, 0.007920957786],
    "V5": [66.38720885, 0.04129256534],
    "V6": [66.38720885, 0.04129256534],
}
online_SoC_HW = {
    "V1": [104.844907, 0.003692106714],
    "V2": [77.41207102, 0.004928058676],
    "V3": [73.11227221, 0.004729872777],
    "V4": [35.78993904, 0.02564483036],
    "V5": [35.78993904, 0.02564483036],
    "V6": [-1, -1],
}
offline_SoC_SW = {
    "V1": [15.33371749, 0.2664353521],
    "V2": [46.11207283, 0.04202522712],
    "V3": [11.04619065, 0.4681093014],
    "V4": [22.55057783, 0.1195961168],
    "V5": [1.950201968, 2.741859043],
    "V6": [0.4814577027, 10.81307611],
}
offline_SoC_HW = {
    "V1": [-1, -1],
    "V2": [-1, -1],
    "V3": [-1, -1],
    "V4": [-1, -1],
    "V5": [-1, -1],
    "V6": [-1, -1],
}
online_data = [online_SoC_SW, online_SoC_HW, online_CPU, online_GPU]
offline_data = [offline_SoC_SW, offline_SoC_HW, offline_CPU, offline_GPU]


def plot_fps_stream_number(type):
    """
        plotting max stream numbers (online) or fps (offline) for each video
        type: 
            - online: plotting [libx264 (SoC-CPU), mediacodec(SoC-HW), libx264 (CPU), nvenc (GPU)]
            - offline: plotting [libx264 (SoC-CPU), libx264 (CPU), nvenc (GPU)]
    """
    label_font_conf = {
        "weight": "bold",
        "size": "11"
    }
    bar_confs = {
        "color": ["white", "white", "silver", "grey"],
        "linewidth": 1,
        "hatch": ["", "//", "", "//"],
        "edgecolor": "black",
    }

    figure_mosaic = """
    AAA.BBB.CCC.DDD.EEE.FFF
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(9, 2), dpi=100)
    bar_width = 0.03
    if type == "offline":
        x = [0.1, 0.1+bar_width*2, 0.1+bar_width*4]
        data = offline_data
    elif type == "online":
        x = [0.1, 0.1+bar_width*2, 0.1+bar_width*4, 0.1+bar_width*6]
        data = online_data


    xlabels = ["V1", "V2", "V3", "V4", "V5", "V6"]
    ax = [axes["A"], axes["B"], axes["C"], axes["D"], axes["E"], axes["F"]]

    for i in range(len(axes)):
        ax[i].set_xlabel(xlabels[i], **label_font_conf)
        ax[i].set_xticks([])
        video_name = xlabels[i]  # video name
        if type == "offline":
            fps = [hw[video_name][0] for hw in data if hw[video_name][0] != -1] 
            print(fps)
            bar_confs["color"] = bar_confs["color"][:3]
            bar_confs["hatch"] = bar_confs["hatch"][:3]
            ax[i].bar(x, fps, width=bar_width, **bar_confs)
        elif type == "online":
            snum = [hw[video_name][0] / video_fps[video_name] if hw[video_name][0] != -1 else 0 for hw in data]
            ax[i].bar(x, snum, width=bar_width, **bar_confs)

        ax[i].grid(axis="y", alpha=0.3)
        ax[i].set_xlim(min(x)-bar_width*1.5, max(x)+bar_width*1.5)

    
    if type == "online":
        ax[0].set_ylabel("Max Stream Num", **label_font_conf)
        # https://matplotlib.org/stable/api/container_api.html#module-matplotlib.container
        bars = ax[0].containers[0].get_children()
        labels = ["SoC-CPU", "SoC-HW", "CPU", "GPU"]
        ax[0].legend(bars, labels, ncol=4, loc="lower left", bbox_to_anchor=(1.8, 1))
    elif type == "offline":
        ax[0].set_ylabel("FPS (frame/s)", **label_font_conf)
        bars = ax[0].containers[0].get_children()
        labels = ["SoC-CPU", "CPU", "GPU"]
        ax[0].legend(bars, labels, ncol=3, loc="lower left", bbox_to_anchor=(2.5, 1))

    plt.subplots_adjust(wspace=2.5)
    plt.show()


if __name__ == '__main__':
    plot_fps_stream_number("offline")
