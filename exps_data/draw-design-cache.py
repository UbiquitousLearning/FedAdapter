# %%
import numpy as np
import os
import re
import pandas as pd
import random
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import math
from matplotlib.pyplot import MultipleLocator

with open("sys",'r') as f:
    sys=f.read()

# %%
ft = 60 # font size
lw = 5 # line width
ms = 25 # marker size
color = ['#9370DB', 'green','darkred','darkblue', 'green', '#FF8C00', '#9370DB', 'hotpink']
marker = ["D", "s","o", "v", "s", "^", "D", "o"]

# %%
plt.figure(figsize=(15,8))
# 设置刻度字体大小
# plt.title("DistilBERT",fontsize=ft)
# 设置刻度字体大小
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
# plt.ylim(-0.03,0.8)
plt.xlabel("Tuning Depth", fontsize=ft)
plt.ylabel("Per-batch Training \nTime (s)", fontsize=ft)
plt.grid(color = 'k', axis="y", linestyle = '--', linewidth = 0.5, alpha=0.4)

x_major_locator=MultipleLocator(20)
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

type = ["w/o cache", "w/ cache"]
runtime = np.array([
    [5.28, 8.02, 9.63, 11.83, 13.12, 15.03, 18.27],
    [0.14, 2.88, 5.7, 8.04, 10.89, 14.2, 18.27]
]) / 20
for i in range(2):
    # if i == 1 or i == 2 or i == 3:
    #     continue
    # log_runtime = [math.log(i, 10) for i in runtime[i]]
    plt.plot(range(7), runtime[i], linewidth = lw, color=color[i], marker=marker[i], markersize = ms, label=type[i])

plt.xticks(range(7))
print(runtime)

plt.legend(fontsize=60,ncol = 1,frameon=False, bbox_to_anchor=(-0.04, 0.52),loc="lower left",)
plt.savefig('../figs/design-cache-perf.pdf', bbox_inches="tight")

# %%



