# %%
cv_models = [("AlexNet",61,460),("DenseNet201",20,2800),("VGG-19",138,11000),("InceptionV3",24,100000),("ResNet152",55,11000),("Xception",23,450000),("ResNeXt101(64x4d)",83,12000)]
nlp_models = [("BERT Large",330,250000),("GPT-1",110,57000),("RoBERTa Large",1500,4300000),("Megatron",8300,8100000),("ELMo",94,3300),("T-NLG",17000,28000000),("GPT-3",175000,310000000)]
# ("Transformer",65,23000),

# %%
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# %%
fig = plt.figure(dpi=300)  
ax1 = fig.add_subplot(111) 
for name,params,PFLOPs in cv_models:
    ax1.scatter(params,PFLOPs,marker=".",label=name, s=200)
for name,params,PFLOPs in nlp_models:
    ax1.scatter(params,PFLOPs,marker="x",label=name, s=100)
ax1.set_yscale('log')
ax1.set_xscale('log')
plt.ylabel("Trainning Compute(PFLOPs)")
plt.xlabel("Nums of Model Parameters(M)")
plt.legend(ncol = 2,fontsize=8)
plt.savefig('../figs/motivation-cost.pdf', bbox_inches="tight")

# %%



