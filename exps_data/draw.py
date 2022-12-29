import numpy as np
import os
import re
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def get_eval_metric(mtcs, file): # 文件中共有多少次eval，以及最后一次eval的metric值
    i = 0
    metric = '\'acc\': '
    target_str = metric + '\d+.?\d+'
    number_str = '\d+.?\d+'
    for line in open(file,"r"):
        if metric in line:
            word = re.findall(target_str, line)[0]
            mtc = re.findall(number_str, word)[0]
            
            mtcs.append(mtc)
            i = i + 1
    return i, mtcs 

def insert_10(left, right):
    l = [left + 1]
    for i in range(left, right):
        if (i - left -1) % 10 == 0 and i != left + 1:
            l.append(i)
    l.append(right)
    return l

def merge_stack(lists):
    for i in range(len(lists)):
        merged_l = []
        l = lists[i]
        for a in l:
            for b in a:
                merged_l.append(b)
        lists[i] = merged_l
    return lists

def cut(x,y,upper_bound_acc):
    x = np.array(x) / 3600 # convert second to hour
    threshold = 0
    delete_y = [t for t in y if t > upper_bound_acc]
    if len(delete_y) > 0:
        if isinstance(y, list):
            threshold = y.index(delete_y[0])
        else:
            threshold = y.tolist().index(delete_y[0])
    else:
        threshold = len(y)
    y = y[:threshold]
    x = x[:threshold]
    return x,y
            
def sum_duration(depth, width, idx, tmp, time, type = "Dyna-A-Freeze"):
    batch_num = 20
    latency_tx2_baseline = np.array([0.5325, 0.612, 0.696, 0.791, 0.883, 0.9713, 1.064, 1.156, 1.2465, 1.33, 1.419, 1.51, 1.7])
    latency_tx2_cached = np.array([0.02, 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.90, 0.99, 1.08]) # msl = 256
    comm_bert = np.array([0.6, 7.7, 14.8, 21.9, 29.0, 36.0, 43.1, 50.2, 57.3, 64.4, 71.5, 78.6, 109.5]) * 32 / 8 # 这里面没有仅仅freeze embedding的数据
    adapter_para = 0.0125 * width / 8
    comm_adapter =np.array([0.02 + i * adapter_para for i in range(0,13)]) * 4
    
    if type == "BERT" or type == "Freeze":
        latency = latency_tx2_baseline
        comm = comm_bert * 2
    if type == "Q-Freeze": # 使用误差补偿的方法; Quantize Freeze; TODO:混合精度量化
        latency = latency_tx2_baseline
        if depth > 2:
            comm = comm_bert * 2 / 4  # INT8 量化 for depth >= 2
        else:
            comm = comm_bert * 2 / 8  # INT4 量化 for depth < 2
    if type == "Quantize" : # 使用误差补偿的方法; Quantize Freeze; TODO:混合精度量化
        latency = latency_tx2_baseline
        comm = comm_bert * 2 / 4  # INT8 量化
    if type == "A-Freeze" or type == "Dyna-A-Freeze":
        latency = latency_tx2_cached 
        comm = comm_adapter * 2

    comm_tmp = comm[depth]
    duration = 0
    for i in range(0, idx - tmp):
        comp_tmp = latency[depth] * batch_num
        duration = duration + comp_tmp + comm_tmp
        # print(comp_tmp,comm_tmp)
    if len(time) > 0:
        time.append(duration + time[-1])
    else:
        time.append(duration)
    return time


ft = 60 # font size
lw = 5 # line width
color = ['red', 'blue', 'green', '#FF8C00', '#9370DB', 'hotpink']
marker = ["o", "v", "s", "^", "D", "o"]
# onto
root_path = "/Users/cdq/Desktop/opensource/FedFinetuning/"
log_root_path = "data/nice_results/semeval_2010_task8-Trail-0-90"
log_file_name = "semeval_2010_task8-depth-0-freq-90.log"
file = os.path.join(root_path, log_root_path, log_file_name)
ft = 60 # font size

# 
dwrm = [[0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 4, 6, 6, 6, 8, 8, 8, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12], [8, 8, 8, 8, 16, 16, 16, 24, 32, 32, 32, 32, 32, 32, 40, 40, 40, 40, 48, 48, 56, 56, 56, 64, 72, 72, 72, 80, 88, 96, 96, 96, 104, 104, 104, 104, 112, 112, 120, 120], [-1, 158, 317, 476, 635, 655, 848, 1032, 1208, 1384, 1560, 1736, 1825, 1938, 2098, 2258, 2378, 2537, 2689, 2810, 2955, 3076, 3221, 3360, 3499, 3633, 3767, 3906, 4045, 4184, 4304, 4424, 4563, 4680, 4797, 4914, 5053, 5166, 5305, 5414],
[0, '0.242914979757085', '0.26499815973500185', '0.2826647037173353', '0.2937062937062937', '0.31137283768862717', '0.583364004416636', '0.6407802723592197', '0.6676481413323518', '0.68052999631947', '0.6912035333087965', '0.7037173352962827', '0.7059256532940743', '0.7118145012881855', '0.7202797202797203', '0.7280088332719912', '0.7405226352594774', '0.7445712182554288', '0.7523003312476997', '0.7607655502392344', '0.7662863452337136', '0.7769598822230401', '0.7773279352226721', '0.7835848362164152', '0.7865292602134707', '0.7868973132131026', '0.7902097902097902', '0.7924181082075819', '0.7920500552079499', '0.7975708502024291', '0.7957305852042694', '0.7979389032020611', '0.8027235921972764', '0.8008833271991167', '0.8034596981965403', '0.8019874861980125', '0.8005152741994848', '0.8034596981965403', '0.8041958041958042', '0.8078763341921237']]

flag = ["init"]
trial_num = len(dwrm[0])
for i in range(trial_num-1):
    depth = dwrm[0]
    width = dwrm[1]
    if depth[i+1] > depth[i]: # deeper
        flag.append("deep")
    elif width[i+1] > width[i]: # wider
        flag.append("wide")
    else:
        flag.append("shallow")
dwrm.append(flag)

new_dwrm = [[],[],[],[]]
for i in range(1, trial_num):
    mtcs = [] # metric
    log_file_name = "fednlp_tc_" + dwrm[-1][i] + "_" + str(i-1) + ".log"
    file = os.path.join(root_path, log_root_path, log_file_name)
    count, mtcs = get_eval_metric(mtcs, file)
    new_dwrm[0].append([dwrm[0][i]] * count)
    new_dwrm[1].append([dwrm[1][i]] * count)
    new_dwrm[2].append(insert_10(dwrm[2][i-1], dwrm[2][i]))
    new_dwrm[3].append(mtcs) # mtcs是对的，其它不知道
# print(new_dwrm)

merged_new_dwrm = merge_stack(new_dwrm)
# print(merged_new_dwrm)
# for depth in dwrm:

Origin = os.path.join(root_path,"data/Baseline/onto_niid_label_clients=600_alpha=1_lr=0.1_freeze=_quantize=False_adapter=False.txt")
Quant = os.path.join(root_path,"data/Baseline/onto_niid_label_clients=600_alpha=1_lr=0.1_freeze=_quantize=True_adapter=False.txt")
Q_Freeze = os.path.join(root_path,"data/Baseline/onto_niid_label_clients=600_alpha=1_lr=0.1_freeze=e,0,1,2,3,4,5_quantize=True_adapter=False.txt")
Freeze = os.path.join(root_path,"data/Baseline/onto_niid_label_clients=600_alpha=1_lr=0.1_freeze=e,0,1,2,3,4,5_quantize=False_adapter=False.txt")
def load_baseline(depth, file):
    baseline = []
    for line in open(file,"r"):
        baseline.append(float(line))
    baseline_len = len(baseline)
    baseline_drm = [[depth]*baseline_len, (np.array(range(0, baseline_len))*5).tolist(), baseline]
    return baseline_drm
baseline_origin_drm = load_baseline(12, Origin)
baseline_quant = load_baseline(12, Quant)
baseline_q_freeze_drm = load_baseline(6, Q_Freeze)
baseline_freeze_drm = load_baseline(6, Freeze)

# agnews plot
plt.figure(figsize=(15,15))
plt.title("Sequence Tagging (onto)",fontsize=ft)
# 设置刻度字体大小
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.xlabel("Time (h)", fontsize=ft)
plt.ylabel("Accuracy", fontsize=ft)
plt.grid(color = 'k', linestyle = '--', linewidth = 0.5, alpha=0.4)
plt.ylim(0,0.78)

x_major_locator=MultipleLocator(20)
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

max_acc = 0.75
target_acc = 1
max_acc = max_acc * target_acc

time = []
data_path = "/Users/cdq/Desktop/opensource/FedFinetuning/data/Baseline/onto-adapter-trial.csv"
raw_data = pd.read_csv(data_path,index_col=0)

tmp = -1 # 记录最后一个访问的idx
data = raw_data.iloc[:,0].dropna()
y = [float(i) for i in data]
round_idx = np.array(list(range(0,len(data)))) * 10
for idx in round_idx:
    d = 0
    w = 8
    time = sum_duration(d, w, idx, tmp, time, "Dyna-A-Freeze")
    tmp = idx

data = raw_data.iloc[:,1].dropna()
y = y + [float(i) for i in data]
round_idx = np.array(list(range(0,len(data)))) * 10
for idx in round_idx:
    d = 4
    w = 32
    time = sum_duration(d, w, idx, tmp, time, "Dyna-A-Freeze")
    tmp = idx

data = raw_data.iloc[:,2].dropna()
y = y + [float(i) for i in data]
round_idx = np.array(list(range(0,len(data)))) * 10
for idx in round_idx:
    d = 10
    w = 32
    time = sum_duration(d, w, idx, tmp, time, "Dyna-A-Freeze")
    tmp = idx

time, y = cut(time, y, max_acc)
plt.plot(np.array(time), y, label = "Dyna-A-Freeze", linewidth = lw, color = color[0])
print("Dyna-A-Freeze",time[-1])


time = []
tmp = -1 # 记录最后一个访问的idx
data = baseline_origin_drm
y = [float(i) for i in baseline_origin_drm[2]]
for idx in data[1]:
    id = data[1].index(idx)
    d = data[0][id]
    w = 0
    time = sum_duration(d, w, idx, tmp, time, "BERT")
    tmp = idx

time, y = cut(time, y, max_acc)
plt.plot(np.array(time), y, label = "BERT", linewidth = lw, color = color[1])
print("BERT",time[-1])


time = []
tmp = -1 # 记录最后一个访问的idx
data = baseline_origin_drm
y = [float(i) for i in baseline_origin_drm[2]]
for idx in data[1]:
    id = data[1].index(idx)
    d = data[0][id]
    w = 0
    time = sum_duration(d, w, idx, tmp, time, "Quantize")
    tmp = idx

time, y = cut(time, y, max_acc)
plt.plot(np.array(time), y, label = "Quant", linewidth = lw, color = color[2])
print("Quant",time[-1])

time = []
tmp = -1 # 记录最后一个访问的idx
data = baseline_freeze_drm
y = [float(i) for i in baseline_freeze_drm[2]]
for idx in data[1]:
    id = data[1].index(idx)
    d = data[0][id]
    w = 0
    time = sum_duration(d, w, idx, tmp, time, "Freeze")
    tmp = idx

time, y = cut(time, y, max_acc)
plt.plot(np.array(time), y, label = "Freeze", linewidth = lw, color = color[3])
print("Freeze",time[-1])


time = []
tmp = -1 # 记录最后一个访问的idx
data = baseline_q_freeze_drm
y = [float(i) for i in baseline_q_freeze_drm[2]]
for idx in data[1]:
    id = data[1].index(idx)
    d = data[0][id]
    w = 0
    time = sum_duration(d, w, idx, tmp, time, "Q-Freeze")
    tmp = idx

time, y = cut(time, y, max_acc)
plt.plot(np.array(time), y, label = "Q-Freeze", linewidth = lw, color = color[4])
print("Q-Freeze",time[-1])



plt.legend(fontsize=50,ncol = 1)
plt.show()
# if target_acc >= 0.99:
#     plt.savefig('../figs/eval-performance-bert-onto.pdf', bbox_inches="tight")