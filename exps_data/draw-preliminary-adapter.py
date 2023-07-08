# %%
import matplotlib
import os
import numpy as np
from numpy import random
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()



# %%
figs_root_path = "../figs/"

# TC 20news adapter 
# DistilBERT
# Freeze heads
tc = np.array([[0.007169411,	0.017657993,	0.025624004,	0.039564525,	0.219596389,	0.291582581],
               [0,	            0.006372809,	0.012878386,	0.021242698,	0.04992034,	    0.148167817],
               [0,	            0,	            0.017525226,	0.019118428,	0.041290494,	0.098114711],
               [0,	            0,	            0,	            0.003319172,	0.021242698,	0.054168879],
               [0,	            0,	            0,	            0,	            0.010621349,	0.037307488],
               [0,	            0,	            0,	            0,	            0,	            0.016728625]]) * 100

fig = plt.figure()
plt.title("Text Classification (20news)",fontsize=20)
# 设置刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
sns_plot = sns.heatmap(tc,cmap= "YlGnBu",annot= True, vmax=20)
sns_plot.set(xlabel="Last ablated layer", ylabel="First ablated layer")
fig_name = "eval-heatmap-TC.pdf"
fig_path = os.path.join(figs_root_path, fig_name)
fig.savefig(fig_path, bbox_inches='tight') # 减少边缘空白


