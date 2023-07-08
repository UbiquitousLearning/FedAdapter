# %%
import numpy as np
import os
import re
import pandas as pd
import random
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from matplotlib.pyplot import MultipleLocator

with open("sys",'r') as f:
    sys=f.read()

# %%
ft = 60 # font size
lw = 5 # line width
ms = 25 # marker size
color = ['darkblue', 'darkred',  'green', '#FF8C00', '#9370DB', 'hotpink']
marker = ["o", "v", "s", "^", "D", "o"]

# %%
plt.figure(figsize=(15,8))
# 设置刻度字体大小
# plt.title("20news(BERT)",fontsize=ft)
# 设置刻度字体大小
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.xlabel("# Clients", fontsize=ft)
plt.ylabel("Elapsed Training \nTime (hrs)", fontsize=ft)
plt.grid(color = 'k', axis="y", linestyle = '--', linewidth = 0.5, alpha=0.4)

x_major_locator=MultipleLocator(1)
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

runtime = [ # 3, 6, 15, 30
    [2.48, 2.18, 3.156, 3.95], # Vanilla BERT Freeze
    [2.1248277777777793, 1.5936972222222225, 1.34, 2.3097805555555575], # T&E
]

x = list(range(4))

type = ["Fixed", "T&E"]
for i in range(2):
    plt.plot(x, runtime[i], linewidth = 5, color=color[i], marker=marker[i], markersize = ms, label=type[i])
plt.xticks(x, labels = [3,6,15,30])
plt.legend(fontsize=60,ncol = 1,frameon=False, bbox_to_anchor=(-0.04, 0.52),loc="lower left",)
plt.savefig('../figs/eval-ablation-clients.pdf', bbox_inches="tight")

# %%
plt.figure(figsize=(15,10),facecolor='black',edgecolor='black')
# 设置刻度字体大小
# plt.title("20news(BERT)",fontsize=ft)
# 设置刻度字体大小
plt.xticks(fontsize=ft,color="white")
plt.yticks(fontsize=ft,color="white")
plt.xlabel("# clients", fontsize=ft,color="white")
plt.ylabel("Clock Time (Hour)", fontsize=ft,color="white")
plt.grid(color = 'k', axis="y", linestyle = '--', linewidth = 0.5, alpha=0.4)

x_major_locator=MultipleLocator(1)
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

runtime = [ # 3, 6, 15, 30
    [2.48, 2.18, 3.156, 3.95], # Vanilla BERT Freeze
    [2.1248277777777793, 1.5936972222222225, 1.34, 2.3097805555555575], # T&E
]

x = list(range(4))

type = ["Fixed", "T&E"]
for i in range(2):
    plt.plot(x, runtime[i], linewidth = 5, color=color[i], marker=marker[i], markersize = ms, label=type[i])
plt.xticks(x, labels = [3,6,15,30])
plt.legend(fontsize=40,ncol = 1)
plt.savefig('../figs/eval-ablation-clients-black.pdf', bbox_inches="tight")

# %%



