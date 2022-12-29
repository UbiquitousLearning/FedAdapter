from socketserver import DatagramRequestHandler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

label_font_confs = {
    "fontsize": 11,
    # "weight": "bold",
}


def parse_csv(filename: str):
    df = pd.read_csv(filename, sep=",", index_col=0)
    return df


def parse_raw_ping_log(file: str):
    with open(file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "rtt min/avg/max/mdev" in line:
            min, avg, max, mdev = line.split()[-2].split("/")
            return float(avg)
    return -1


def tile_values(df):
    l = list()
    for _, val in df.iterrows():
        l += val.to_list()
    l = [x for x in l if not np.isnan(x)]

    return l


def get_cdf_axis(x):
    x = sorted(x)
    x_accu = np.cumsum(x)
    x_accu = x_accu / x_accu[-1]
    return x, x_accu


def parse_csv_grouped(filename: str):
    intra_board, inter_board = [], []
    df = pd.read_csv(filename, sep=",", index_col=0)
    for row in df.itertuples():
        cur_index = int(row.Index[3:])
        for i in range(1, 56):
            if i == cur_index:
                continue
            if (i-1) // 5 == (cur_index-1) // 5:
                intra_board.append(row[i])
            else:
                inter_board.append(row[i])

    return intra_board, inter_board


def draw_latency():
    """latency 
        1. between x86 server and all socs
        2. all socs inside soc server"""

    # x86 server and all socs
    edge_soc_latency = []
    folder = "linux-soc-logs"
    for f in os.listdir(folder):
        edge_soc_latency.append(parse_raw_ping_log(os.path.join(folder, f)))

    print(f"[x86-soc] avg latency: {np.mean(edge_soc_latency)}, stderr: {np.std(edge_soc_latency)}")
    edge_soc_latency, edge_soc_latency_accu = get_cdf_axis(edge_soc_latency)

    # intra-board, inter-board socs
    intra, inter = parse_csv_grouped("soc-soc-logs/ping.csv")
    print(f"[intra-board socs] avg latency: {np.mean(intra)}, stderr: {np.std(intra)}")
    print(f"[inter-board socs] avg latency: {np.mean(inter)}, stderr: {np.std(inter)}")
    intra_latency, intra_latency_accu = get_cdf_axis(intra)
    inter_latency, inter_latency_accu = get_cdf_axis(inter)

    plt.figure(figsize=(3, 2), dpi=100)
    confs = {
        "linewidth": 1.8,
    }

    plt.plot(edge_soc_latency, edge_soc_latency_accu, label="x86-SoCs", color="black", linestyle="-", **confs)
    plt.plot(intra_latency, intra_latency_accu, label="Intra-board SoCs", color="blue", linestyle="-.", **confs)
    plt.plot(inter_latency, inter_latency_accu, label="Inter-board SoCs", color="orange", linestyle="--", **confs)

    plt.xlabel("Latency (ms)", **label_font_confs)
    plt.ylabel("CDF", **label_font_confs)
    plt.legend(fontsize=8, loc="lower right")
    # plt.show()
    plt.savefig("latency.pdf", bbox_inches="tight", pad_inches=0)


def draw_throughput():
    """protocol: tcp or udp
    """
    tcp_intra, tcp_inter = parse_csv_grouped("soc-soc-logs/tcp.csv")
    print(f"[intra-board socs] tcp avg throughput: {np.mean(tcp_intra)}, stderr: {np.std(tcp_intra)}")
    print(f"[inter-board socs] tcp avg throughput: {np.mean(tcp_inter)}, stderr: {np.std(tcp_inter)}")
    tcp_intra_throughput, tcp_intra_throughput_accu = get_cdf_axis(tcp_intra)
    tcp_inter_throughput, tcp_inter_throughput_accu = get_cdf_axis(tcp_inter)

    udp_intra, udp_inter = parse_csv_grouped("soc-soc-logs/udp.csv")
    print(f"[intra-board socs] udp avg throughput: {np.mean(udp_intra)}, stderr: {np.std(udp_intra)}")
    print(f"[inter-board socs] udp avg throughput: {np.mean(udp_inter)}, stderr: {np.std(udp_inter)}")
    udp_intra_throughput, udp_intra_throughput_accu = get_cdf_axis(udp_intra)
    udp_inter_throughput, udp_inter_throughput_accu = get_cdf_axis(udp_inter)

    figure_mosaic = """
    UT
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(3, 2), sharey=True, dpi=100)
    confs = {
        "linewidth": 1.8,
    }

    axes["U"].plot(udp_intra_throughput, udp_intra_throughput_accu, label="UDP Intra", color="black", linestyle=":", **confs)
    axes["U"].plot(udp_inter_throughput, udp_inter_throughput_accu, label="UDP Inter", color="red", alpha=0.5, linestyle="-", **confs)
    axes["U"].set_xlim(728, 761)
    axes["U"].spines.right.set_visible(False)
    # axes["U"].yaxis.tick_left()
    # axes["U"].tick_params(labelright=False)

    axes["T"].plot(tcp_intra_throughput, tcp_intra_throughput_accu, label="TCP Intra", color="blue", linestyle="-.", **confs)
    axes["T"].plot(tcp_inter_throughput, tcp_inter_throughput_accu, label="TCP Inter", color="orange", linestyle="--", **confs)
    axes["T"].set_xlim(931, 942)
    axes["T"].spines.left.set_visible(False)
    # axes["T"].yaxis.tick_right()
    axes["T"].tick_params(left=False)

    axes["U"].legend(fontsize=8)
    axes["T"].legend(fontsize=8)
    axes["U"].set_ylabel("CDF", **label_font_confs)
    fig.supxlabel("Throughput (Mbps)", y=-0.11, **label_font_confs)
    plt.subplots_adjust(wspace=0.08)

    d = 1.70  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=9,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axes["U"].plot([1, 1], [0, 1], transform=axes["U"].transAxes, **kwargs)
    axes["T"].plot([0, 0], [1, 0], transform=axes["T"].transAxes, **kwargs)

    # plt.show()
    plt.savefig("throughput.pdf", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    draw_latency()
    draw_throughput()
