# %%
# !pip install brewer2mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
plt.rc('font',family='Times New Roman')
from matplotlib.pyplot import MultipleLocator

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def border():
    bwith = 2 #边框宽度设置为2
    ax = plt.gca()#获取边框
    ax.tick_params(axis="x", direction="in", length=6, width=5, color="k")
    ax.tick_params(axis="x", direction="in", length=6, width=5, color="k")
    # ax.spines['top'].set_color('grey')  # 设置上‘脊梁’为红色
    # ax.spines['right'].set_color('grey')  # 设置右‘脊梁’为无色
    # ax.spines['left'].set_color('grey')  # 设置上‘脊梁’为红色
    # ax.spines['bottom'].set_color('none')  # 设置右‘脊梁’为无色
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)


# %%
def freq():
    # Frequence-temperature
    ft = 40
    x = list(range(7))# 时间(s)

    # y1 = np.array([0.6, 7.7, 14.8, 21.9, 29.0, 36.0, 43.1])
    y1 = np.array([65.51, 47.04, 41.07, 36.99, 30.14, 25.10, 9.51])

    # y2 = ((0.8232 - np.array([0.7080, 0.7267, 0.7873, 0.8034, 0.8225, 0.8221, 0.8232]))) * 100
    y2 = ((0.8232 - np.array([0.8232, 0.8221, 0.8225, 0.8034, 0.7873, 0.7267, 0.7080]))) * 100
    fig = plt.figure(figsize=(8,5))
    # 下面的代码会导致多出来很奇怪的坐标轴刻度
    # plt.axes().get_xaxis().set_visible(False)
    # plt.axes().get_yaxis().set_visible(False)
    # plt.grid(linestyle = "--") 
    
    ax1 = fig.add_subplot(111)
    ax1.plot(np.array(x), y1, linewidth = 6, color='darkblue', linestyle = "--", label="Time")
    # x_major_locator=MultipleLocator(10)
    y_major_locator=MultipleLocator(20)
    ax=plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    # border()
    # plt.grid(color = 'k', linestyle = (0, (1, 1)), axis="y", linewidth = 0.5, alpha=0.4)
    plt.tick_params(axis='y',colors='darkblue')
    plt.xticks(fontsize=ft)
    plt.yticks(fontsize=ft)
    ax1.set_ylabel("Elapsed Training\nTime (hrs)",  color='darkblue', fontsize=ft)
    ax1.set_xlabel('Freezing Layers',  fontsize=ft)
    plt.legend(fontsize=ft, loc=3, bbox_to_anchor=(0.05, 1.03), ncol=1,borderaxespad = 0., frameon=False,handletextpad=0.3)

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(np.array(x), y2, 'r', linewidth = 6, color="darkred", label="Loss")
    # x_major_locator=MultipleLocator(10)
    y_major_locator=MultipleLocator(4)
    ax=plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.tick_params(axis='y',colors='darkred')
    plt.xticks(fontsize=ft)
    plt.yticks(fontsize=ft)
    ax2.set_ylabel("Accuracy Loss (%)", color='darkred',fontsize=ft)


    plt.legend(fontsize=ft, loc=3, bbox_to_anchor=(0.5, 1.03), ncol=1,borderaxespad = 0., frameon=False,handletextpad=0.3)
    plt.savefig("../figs/eval-preliminary-freeze.pdf", bbox_inches="tight")

freq()

# %%



