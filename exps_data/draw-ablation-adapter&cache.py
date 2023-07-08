# %%
import numpy as np
import os
import re
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

with open("sys",'r') as f:
    sys=f.read()

# %%
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
            
def sum_duration(depth, width, idx, tmp, time, type = "Dyna-A-Freeze", dataset = "20news"):
    if dataset == "20news":
        batch_num = 29
    if dataset == "agnews":
        batch_num = 30
    if dataset == "semeval":
        batch_num = 20
    if dataset == "onto":
        batch_num = 20

    if dataset == "20news" or dataset == "onto":
        latency_tx2_baseline = np.array([0.5325, 0.612, 0.696, 0.791, 0.883, 0.9713, 1.064, 1.156, 1.2465, 1.33, 1.419, 1.51, 1.7]) 
        latency_tx2_adapter = np.array([0.57, 0.57, 0.63, 0.66, 0.71, 0.75, 0.79, 0.84, 0.89, 0.94, 0.99, 1.03, 1.08])
        latency_tx2_cached = np.array([0.02, 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.90, 0.99, 1.08]) # msl = 256
    else:
        latency_tx2_baseline = np.array([0.5325, 0.612, 0.696, 0.791, 0.883, 0.9713, 1.064, 1.156, 1.2465, 1.33, 1.419, 1.51, 1.7]) / 4
        latency_tx2_adapter = np.array([0.57, 0.57, 0.63, 0.66, 0.71, 0.75, 0.79, 0.84, 0.89, 0.94, 0.99, 1.03, 1.08]) / 4
        latency_tx2_cached = np.array([0.02, 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.90, 0.99, 1.08]) / 4# msl = 64
    comm_bert = np.array([0.6, 7.7, 14.8, 21.9, 29.0, 36.0, 43.1, 50.2, 57.3, 64.4, 71.5, 78.6, 109.5]) * 32 / 8 # 这里面没有仅仅freeze embedding的数据
    adapter_para = 0.0125 * width / 8
    comm_adapter =np.array([0.02 + i * adapter_para for i in range(0,13)]) * 4
    
    if type == "BERT":
        latency = latency_tx2_baseline
        comm = comm_bert * 2
    elif type == "Adapter":
        latency = latency_tx2_adapter
        comm = comm_adapter * 2
    elif type == "Cache":
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


# %%
target_acc = 0.99
runtime = {
    "20news": [],
    "agnews": [],
    "semeval": [],
    "onto": []
    # TODO: 这里的onto应该是模拟的到达0.8准确率的时间,也不对，反正onto的数据有点问题
}

# %%
# 20news
root_path = "."

Origin = os.path.join(root_path,"Baseline/20news_uniform_lr=0.1_freeze=_quantize=False_adapter=False_MSL=256_workers15_rounds10.txt")
Quant = os.path.join(root_path,"Baseline/20news_uniform_lr=0.1_freeze=_quantize=True_adapter=False_MSL=256_workers15_rounds10.txt")
Q_Freeze = os.path.join(root_path,"Baseline/20news_uniform_lr=0.1_freeze=e,0,1,2,3,4,5,6,7,8,9_quantize=True_adapter=False_MSL=256_workers15_rounds10.txt")
Freeze = os.path.join(root_path,"Baseline/20news_uniform_lr=0.1_freeze=e,0,1,2,3,4,5,6,7,8,9_quantize=False_adapter=False_MSL=256_workers15_rounds10.txt")
def load_baseline(depth, file):
    baseline = []
    for line in open(file,"r"):
        baseline.append(float(line))
    baseline_len = len(baseline)
    baseline_drm = [[depth]*baseline_len, (np.array(range(0, baseline_len))*10).tolist(), baseline]
    return baseline_drm
baseline_origin_drm = load_baseline(12, Origin)
baseline_quant = load_baseline(12, Quant)
baseline_q_freeze_drm = load_baseline(2, Q_Freeze)
baseline_freeze_drm = load_baseline(2, Freeze)

max_acc = 0.8
max_acc = max_acc * target_acc

time = []
tmp = -1 # 记录最后一个访问的idx
data = baseline_origin_drm
y = [float(i) for i in baseline_origin_drm[2]]
for idx in data[1]:
    id = data[1].index(idx)
    d = data[0][id]
    w = 0
    time = sum_duration(d, w, idx, tmp, time, "BERT", "20news")
    tmp = idx

time, y = cut(time, y, max_acc)
print("BERT",time[-1])
runtime["20news"].append(time[-1])


data_path = "./Baseline/20news.csv"
raw_data = pd.read_csv(data_path,index_col=0)
column_name = raw_data.columns.values

col = "Depth-3-Width-32 - Evaluation Accuracy"
time = []
multiple = 10
w = int(col.split("-")[3])
d = int(col.split("-")[1])
d = 12
data = raw_data.iloc[:,column_name.tolist().index(col)].dropna()
round_idx = np.array(list(range(0,len(data)))) * multiple
tmp = -1 * multiple # 记录最后一个访问的idx
for idx in round_idx:
    time = sum_duration(d, w, idx, tmp, time, "Adapter", "20news")
    tmp = idx
time, data = cut(time, data, max_acc)
print("Adapter",time[-1])
runtime["20news"].append(time[-1])

col = "Depth-2-Width-8 - Evaluation Accuracy"
time = []
multiple = 10
w = int(col.split("-")[3])
d = int(col.split("-")[1])
print(d)
data = raw_data.iloc[:,column_name.tolist().index(col)].dropna()
round_idx = np.array(list(range(0,len(data)))) * multiple
tmp = -1 * multiple # 记录最后一个访问的idx
for idx in round_idx:
    time = sum_duration(d, w, idx, tmp, time, "Adapter", "20news")
    tmp = idx
time, data = cut(time, data, max_acc)
print("Adapter optimal",time[-1])
runtime["20news"].append(time[-1])

col = "Depth-2-Width-8 - Evaluation Accuracy"
time = []
multiple = 10
w = int(col.split("-")[3])
d = int(col.split("-")[1])
data = raw_data.iloc[:,column_name.tolist().index(col)].dropna()
round_idx = np.array(list(range(0,len(data)))) * multiple
tmp = -1 * multiple # 记录最后一个访问的idx
for idx in round_idx:
    time = sum_duration(d, w, idx, tmp, time, "Cache", "20news")
    tmp = idx
time, data = cut(time, data, max_acc)
print("Adapter optimal cached",time[-1])
runtime["20news"].append(time[-1])

# %%
# agnews
root_path = "."

Origin = os.path.join(root_path,"Baseline/agnews_niid_label_clients=1000_alpha=10.0_lr=0.1_freeze=_quantize=False_adapter=False_length=64.txt")
Quant = os.path.join(root_path,"Baseline/agnews_niid_label_clients=1000_alpha=10.0_lr=0.1_freeze=_quantize=True_adapter=False_length=64.txt")
Q_Freeze = os.path.join(root_path,"Baseline/agnews_niid_label_clients=1000_alpha=10.0_lr=0.1_freeze=e,0,1,2,3,4,5,6,7,8,9_quantize=True_adapter=False_MSL=64_workers15_rounds10.txt")
Freeze = os.path.join(root_path,"Baseline/agnews_niid_label_clients=1000_alpha=10.0_lr=0.1_freeze=e,0,1,2,3,4,5,6,7,8,9_quantize=False_adapter=False_MSL=64_workers15_rounds10.txt")
def load_baseline(depth, file):
    baseline = []
    for line in open(file,"r"):
        baseline.append(float(line))
    baseline_len = len(baseline)
    baseline_drm = [[depth]*baseline_len, (np.array(range(0, baseline_len))*5).tolist(), baseline]
    return baseline_drm
baseline_origin_drm = load_baseline(12, Origin)
baseline_quant = load_baseline(12, Quant)
def load_baseline(depth, file):
    baseline = []
    for line in open(file,"r"):
        baseline.append(float(line))
    baseline_len = len(baseline)
    baseline_drm = [[depth]*baseline_len, (np.array(range(0, baseline_len))*10).tolist(), baseline]
    return baseline_drm
baseline_q_freeze_drm = load_baseline(2, Q_Freeze)
baseline_freeze_drm = load_baseline(2, Freeze)
# print(baseline_quant)

max_acc = 0.9
max_acc = max_acc * target_acc


time = []
tmp = -1 # 记录最后一个访问的idx
data = baseline_origin_drm
y = [float(i) for i in baseline_origin_drm[2]]
for idx in data[1]:
    id = data[1].index(idx)
    d = data[0][id]
    w = 0
    time = sum_duration(d, w, idx, tmp, time, "BERT", "agnews")
    tmp = idx

time, y = cut(time, y, max_acc)
print("BERT",time[-1])
runtime["agnews"].append(time[-1])


data_path = "./Baseline/agnews-adapter.csv"
raw_data = pd.read_csv(data_path,index_col=0)
column_name = raw_data.columns.values

col = "depth-2"
time = []
multiple = 10
w = 32
d = 12
data = raw_data.iloc[:,column_name.tolist().index(col)].dropna()
round_idx = np.array(list(range(0,len(data)))) * multiple
tmp = -1 * multiple # 记录最后一个访问的idx
for idx in round_idx:
    time = sum_duration(d, w, idx, tmp, time, "Adapter", "agnews")
    tmp = idx
time, data = cut(time, data, max_acc)
print("Adapter",time[-1])
runtime["agnews"].append(time[-1])

col = "depth-2"
time = []
multiple = 10
w = 16
d = 3
data = raw_data.iloc[:,column_name.tolist().index(col)].dropna()
round_idx = np.array(list(range(0,len(data)))) * multiple
tmp = -1 * multiple # 记录最后一个访问的idx
for idx in round_idx:
    time = sum_duration(d, w, idx, tmp, time, "Adapter", "agnews")
    tmp = idx
time, data = cut(time, data, max_acc)
print("Adapter optimal",time[-1])
runtime["agnews"].append(time[-1])

col = "depth-2"
time = []
multiple = 10
w = 16
d = 3
data = raw_data.iloc[:,column_name.tolist().index(col)].dropna()
round_idx = np.array(list(range(0,len(data)))) * multiple
tmp = -1 * multiple # 记录最后一个访问的idx
for idx in round_idx:
    time = sum_duration(d, w, idx, tmp, time, "Cache", "agnews")
    tmp = idx
time, data = cut(time, data, max_acc)
print("Adapter optimal cached",time[-1])
runtime["agnews"].append(time[-1])

# %%
# semeval
root_path = "."

Origin = os.path.join(root_path,"Baseline/semeval_2010_task8_niid_label_clients=100_alpha=100_lr=0.1_freeze=_quantize=False_adapter=False_length=64.txt")
Quant = os.path.join(root_path,"Baseline/semeval_2010_task8_niid_label_clients=100_alpha=100_lr=0.1_freeze=_quantize=True_adapter=False_length=64.txt")
Q_Freeze = os.path.join(root_path,"Baseline/semeval_2010_task8_niid_label_clients=100_alpha=100_lr=0.1_freeze=e,0,1,2,3,4,5_quantize=True_adapter=False_MSL=64_workers15_rounds10.txt")
Freeze = os.path.join(root_path,"Baseline/semeval_2010_task8_niid_label_clients=100_alpha=100_lr=0.1_freeze=e,0,1,2,3,4,5_quantize=False_adapter=False_MSL=64_workers15_rounds10.txt")
def load_baseline(depth, file):
    baseline = []
    for line in open(file,"r"):
        baseline.append(float(line))
    baseline_len = len(baseline)
    baseline_drm = [[depth]*baseline_len, (np.array(range(0, baseline_len))*5).tolist(), baseline]
    return baseline_drm
baseline_origin_drm = load_baseline(12, Origin)
baseline_quant = load_baseline(12, Quant)
def load_baseline(depth, file):
    baseline = []
    for line in open(file,"r"):
        baseline.append(float(line))
    baseline_len = len(baseline)
    baseline_drm = [[depth]*baseline_len, (np.array(range(0, baseline_len))*10).tolist(), baseline]
    return baseline_drm
baseline_q_freeze_drm = load_baseline(6, Q_Freeze)
baseline_freeze_drm = load_baseline(6, Freeze)

max_acc = 0.8
max_acc = max_acc * target_acc


time = []
tmp = -1 # 记录最后一个访问的idx
data = baseline_origin_drm
y = [float(i) for i in baseline_origin_drm[2]]
for idx in data[1]:
    id = data[1].index(idx)
    d = data[0][id]
    w = 0
    time = sum_duration(d, w, idx, tmp, time, "BERT", "semeval")
    tmp = idx

time, y = cut(time, y, max_acc)
print("BERT",time[-1])
runtime["semeval"].append(time[-1])


data_path = "./Baseline/semeval-adapter.csv"
raw_data = pd.read_csv(data_path,index_col=0)
column_name = raw_data.columns.values

col = "depth-12"
time = []
multiple = 10
w = 32
d = 12
data = raw_data.iloc[:,column_name.tolist().index(col)].dropna()
round_idx = np.array(list(range(0,len(data)))) * multiple
tmp = -1 * multiple # 记录最后一个访问的idx
for idx in round_idx:
    time = sum_duration(d, w, idx, tmp, time, "Adapter", "semeval")
    tmp = idx
time, data = cut(time, data, max_acc)
print("Adapter",time[-1])
runtime["semeval"].append(time[-1])

col = "depth-12"
time = []
multiple = 10
w = 8
d = 10
data = raw_data.iloc[:,column_name.tolist().index(col)].dropna()
round_idx = np.array(list(range(0,len(data)))) * multiple
tmp = -1 * multiple # 记录最后一个访问的idx
for idx in round_idx:
    time = sum_duration(d, w, idx, tmp, time, "Adapter", "semeval")
    tmp = idx
time, data = cut(time, data, max_acc)
print("Adapter optimal",time[-1])
runtime["semeval"].append(time[-1])

col = "depth-12"
time = []
multiple = 10
w = 8
d = 10
data = raw_data.iloc[:,column_name.tolist().index(col)].dropna()
round_idx = np.array(list(range(0,len(data)))) * multiple
tmp = -1 * multiple # 记录最后一个访问的idx
for idx in round_idx:
    time = sum_duration(d, w, idx, tmp, time, "Cache", "semeval")
    tmp = idx
time, data = cut(time, data, max_acc)
print("Adapter optimal cached",time[-1])
runtime["semeval"].append(time[-1])

# %%
# onto
root_path = "."


Origin = os.path.join(root_path,"Baseline/onto_niid_label_clients=30_alpha=1.0_lr=0.1_freeze=_quantize=False_adapter=False_MSL=256_workers15_rounds10.txt")
Quant = os.path.join(root_path,"Baseline/onto_niid_label_clients=30_alpha=1.0_lr=0.1_freeze=_quantize=True_adapter=False_MSL=256_workers15_rounds10.txt")
Q_Freeze = os.path.join(root_path,"Baseline/onto_niid_label_clients=30_alpha=1.0_lr=0.1_freeze=e,0,1,2,3,4,5_quantize=False_adapter=False_MSL=256_workers15_rounds10.txt")
Freeze = os.path.join(root_path,"Baseline/onto_niid_label_clients=30_alpha=1.0_lr=0.1_freeze=e,0,1,2,3,4,5_quantize=True_adapter=False_MSL=256_workers15_rounds10.txt")
def load_baseline(depth, file):
    baseline = []
    for line in open(file,"r"):
        baseline.append(float(line))
    baseline_len = len(baseline)
    baseline_drm = [[depth]*baseline_len, (np.array(range(0, baseline_len))*10).tolist(), baseline]
    return baseline_drm
baseline_origin_drm = load_baseline(12, Origin)
baseline_quant = load_baseline(12, Quant)
# baseline_q_freeze_drm = load_baseline(6, Q_Freeze)
# 对应table2，此处应该至少是8
baseline_q_freeze_drm = load_baseline(6, Q_Freeze)
baseline_freeze_drm = load_baseline(6, Freeze)

max_acc = 0.75
max_acc = max_acc * target_acc


time = []
tmp = -1 # 记录最后一个访问的idx
data = baseline_origin_drm
y = [float(i) for i in baseline_origin_drm[2]]
for idx in data[1]:
    id = data[1].index(idx)
    d = data[0][id]
    w = 0
    time = sum_duration(d, w, idx, tmp, time, "BERT", "onto")
    tmp = idx

time, y = cut(time, y, max_acc)
print("BERT",time[-1])
runtime["onto"].append(time[-1])


data_path = "./Baseline/onto-adapter.csv"
raw_data = pd.read_csv(data_path,index_col=0)
column_name = raw_data.columns.values

col = "depth-12-width-32"
time = []
multiple = 10
w = 32
d = 12
data = raw_data.iloc[:,column_name.tolist().index(col)].dropna()
round_idx = np.array(list(range(0,len(data)))) * multiple
tmp = -1 * multiple # 记录最后一个访问的idx
for idx in round_idx:
    time = sum_duration(d, w, idx, tmp, time, "Adapter", "onto")
    tmp = idx
time, data = cut(time, data, max_acc)
print("Adapter",time[-1])
runtime["onto"].append(time[-1])

col = "depth-12-width-32"
time = []
multiple = 10
w = 32
d = 12
data = raw_data.iloc[:,column_name.tolist().index(col)].dropna()
round_idx = np.array(list(range(0,len(data)))) * multiple
tmp = -1 * multiple # 记录最后一个访问的idx
for idx in round_idx:
    time = sum_duration(d, w, idx, tmp, time, "Adapter", "onto")
    tmp = idx
time, data = cut(time, data, max_acc)
print("Adapter optimal",time[-1])
runtime["onto"].append(time[-1])

col = "depth-12-width-32"
time = []
multiple = 10
w = 32
d = 12
data = raw_data.iloc[:,column_name.tolist().index(col)].dropna()
round_idx = np.array(list(range(0,len(data)))) * multiple
tmp = -1 * multiple # 记录最后一个访问的idx
for idx in round_idx:
    time = sum_duration(d, w, idx, tmp, time, "Cache", "onto")
    tmp = idx
time, data = cut(time, data, max_acc)
print("Adapter optimal cached",time[-1])
runtime["onto"].append(time[-1])

# %%
runtime

# %%


# %%



