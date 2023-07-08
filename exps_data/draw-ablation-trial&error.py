# %%
import numpy as np
import os
import re
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

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
        latency_tx2_cached = np.array([0.02, 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.90, 0.99, 1.08]) # msl = 256
    else:
        latency_tx2_baseline = np.array([0.5325, 0.612, 0.696, 0.791, 0.883, 0.9713, 1.064, 1.156, 1.2465, 1.33, 1.419, 1.51, 1.7]) / 4
        latency_tx2_cached = np.array([0.02, 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.90, 0.99, 1.08]) / 4# msl = 64
    comm_bert = np.array([0.6, 7.7, 14.8, 21.9, 29.0, 36.0, 43.1, 50.2, 57.3, 64.4, 71.5, 78.6, 109.5]) * 32 / 8 # 这里面没有仅仅freeze embedding的数据
    adapter_para = 0.0125 * width / 8
    comm_adapter =np.array([0.02 + i * adapter_para for i in range(0,13)]) * 4
    
    if type == "BERT" or type == "Freeze":
        latency = latency_tx2_baseline
        comm = comm_bert * 2
    if type == "Q-Freeze": # 使用误差补偿的方法; Quantize Freeze; TODO:混合精度量化
        latency = latency_tx2_baseline
        comm = comm_bert * 2 / 8  # INT4 量化
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


# %%
# 20news
# 20news
root_path = "."
log_root_path = "nice_results/20news-Trail-0-60"
log_file_name = "fednlp_tc_deep_0.log"
file = os.path.join(root_path, log_root_path, log_file_name)


# depth, width, round, metric
dwrm = [[0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3], [8, 8, 8, 16, 16, 16, 24, 32, 32, 32, 40, 48, 48, 48, 56, 64, 64, 64, 64, 64], [-1, 79, 88, 180, 272, 364, 453, 539, 625, 711, 794, 875, 956, 1037, 1115, 1191, 1267, 1343, 1393, 1467], [0, '0.4681359532660648', '0.602761550716941', '0.7422995220392989', '0.7559745087626129', '0.7636749867233139', '0.7684545937334042', '0.7729686670207117', '0.7804036112586299', '0.7838555496548062', '0.7892989909718534', '0.7934147636749868', '0.7954062665958577', '0.7966011683483802', '0.7980616038236856', '0.8005841741901222', '0.8029739776951673', '0.8016463090812533', '0.8020446096654275', '0.8031067445565587']]


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

# from matplotlib.pyplot import MultipleLocator
# x_major_locator=MultipleLocator(2)
# y_major_locator=MultipleLocator(0.2)
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)

max_acc = 0.8
target_acc = 0.99
max_acc = max_acc * target_acc

fig, ax = plt.subplots(figsize=(15,8))
ft = 60
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.xlabel("Elapsed Training Time (hrs)", fontsize=ft)
plt.ylabel("Relative Accuracy", fontsize=ft)
plt.grid(color = 'k', axis="y", linestyle = '--', linewidth = 0.5, alpha=0.2)
plt.xlim(-0.03,7)

from matplotlib.pyplot import MultipleLocator
x_major_locator=MultipleLocator(2)
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)


data_path = "./Baseline/20news_bert.csv"
raw_data = pd.read_csv(data_path,index_col=0)
column_name = raw_data.columns.values
multiple = 20
x = []
y = []

for col in column_name:
    time = []
    d = int(col.split("-")[1])
    w = int(col.split("-")[3])
    # if d == 2 and w == 16:
    #     continue
    data = raw_data.iloc[:,column_name.tolist().index(col)].dropna()
    round_idx = np.array(list(range(0,len(data)))) * multiple
    tmp = -1 * multiple # 记录最后一个访问的idx
    for idx in round_idx:
        time = sum_duration(d, w, idx, tmp, time, "Dyna-A-Freeze")
        tmp = idx
    time, data = cut(time, data, max_acc)
    for t in time:
        x.append(t)
    for d in data:
        y.append(d)
    # plt.scatter(np.array(time), data,s=100,c='blue',alpha=0.5) 
    # plt.plot(np.array(time), data, label = col)
    # print(col,time[-1])

re_x = np.array(range(0,82))/10
# print(re_x)

for i in range(len(x)):
    for j in range(len(re_x)):
        if re_x[j] > x[i]:
            x[i] = re_x[j]-0.1
            break
        
data = np.column_stack((x, y))  # (630,2)
x = list(set(x))
x = sorted(x)
# print(x)
df = pd.DataFrame(data, columns=['time (h)', 'accuracy'])
y_group = df.groupby("time (h)")
y_mean = y_group.mean()["accuracy"]
y_max = y_group.max()["accuracy"]
y_min = y_group.min()["accuracy"]

plt.plot(x, np.array(y_mean)/max_acc, color='darkblue',  linewidth = 10, label='Fixed',linestyle = (0, (1, 1)))
# plt.plot(time, sin_waves, 'b-', label='ideal')
plt.fill_between(x, np.array(y_min)/max_acc, np.array(y_max)/max_acc, color='c', alpha=0.2)


time = []
data = new_dwrm
tmp = -1 # 记录最后一个访问的idx
y = [float(i) for i in data[3]]
for idx in data[2]:
    id = data[2].index(idx)
    d = data[0][id]
    w = data[1][id]
    time = sum_duration(d, w, idx, tmp, time, "Dyna-A-Freeze")
    tmp = idx

time, y = cut(time, y, max_acc)
data = np.column_stack((np.array(time), y))  # (630,2)
df = pd.DataFrame(data, columns=['time (h)', 'accuracy'])


# sns.lineplot(x='time (h)', y='accuracy',data=df, ci = 100, color="darkred", linewidth = 10, label = "Ours")
plt.plot(np.array(time), np.array(y)/max_acc, label = sys, color = "darkred", linewidth = 10)
print("Dyna-A-Freeze",time[-1])

plt.legend(fontsize=ft,ncol = 1,frameon=False,loc="lower right")
plt.savefig('../figs/eval-ablation-trial-20news.pdf', bbox_inches="tight")


# %%
# semeval
root_path = "."
log_root_path = "nice_results/semeval_2010_task8-Trail-1-90"
log_file_name = "semeval_2010_task8-depth-0-freq-90.log"
file = os.path.join(root_path, log_root_path, log_file_name)
ft = 60 # font size

# 
dwrm = [[1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12], [8, 16, 16, 16, 16, 16, 16, 16, 16, 24, 32, 32, 32, 32, 32, 40, 48, 48, 48, 56, 56, 56, 56], [-1, 206, 249, 434, 619, 804, 989, 1174, 1359, 1535, 1703, 1871, 1964, 2088, 2253, 2410, 2561, 2712, 2863, 3008, 3153, 3298, 3443],
[0, '0.2528524107471476', '0.3257269046742731', '0.6352594773647405', '0.6838424733161576', '0.7143908722856092', '0.7364740522635259', '0.7534044902465955', '0.7603974972396025', '0.7684946632315053', '0.7762237762237763', '0.7813765182186235', '0.7854251012145749', '0.7894736842105263', '0.7994111152005889', '0.805668016194332', '0.805668016194332', '0.8100846521899153', '0.8100846521899153', '0.8141332351858668', '0.8185498711814502', '0.8181818181818182', '0.819654030180346']]


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

ft = 60


# from matplotlib.pyplot import MultipleLocator
# x_major_locator=MultipleLocator(2)
# y_major_locator=MultipleLocator(0.2)
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)

max_acc = 0.8
target_acc = 1
max_acc = max_acc * target_acc

fig, ax = plt.subplots(figsize=(15,8))
ft = 60
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.xlabel("Elapsed Training Time (hrs)", fontsize=ft)
# plt.ylabel("Accuracy", fontsize=ft)
plt.grid(color = 'k', axis="y", linestyle = '--', linewidth = 0.5, alpha=0.2)
plt.xlim(0,12)


from matplotlib.pyplot import MultipleLocator
x_major_locator=MultipleLocator(2)
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)


data_path = "./Baseline/semeval-plug.csv"
raw_data = pd.read_csv(data_path,index_col=0)
column_name = raw_data.columns.values
multiple = 20
x = []
y = []

for col in column_name:
    if col == "Depth-0-Width-128 - Evaluation Accuracy":
        multiple = 15
    time = []
    d = int(col.split("-")[1])
    w = int(col.split("-")[3])
    data = raw_data.iloc[:,column_name.tolist().index(col)].dropna().tolist()
    round_idx = np.array(list(range(0,len(data)))) * multiple
    tmp = -1 * multiple # 记录最后一个访问的idx
    for idx in round_idx:
        time = sum_duration(d, w, idx, tmp, time, "Dyna-A-Freeze","semeval")
        tmp = idx
    # print(data)
    time, data = cut(time, data, 0.8)
    for i in range(len(data)):
        if data[i] <= max_acc:
            x.append(time[i])
            y.append(data[i])
        else:
            break
    # plt.scatter(np.array(time), data,s=100,c='blue',alpha=0.5) 
    # plt.plot(np.array(time), data, label = col)
    print(col,time[-1])

re_x = np.array(range(0,200))/ 10

# 将离散点聚合到固定点位上
for i in range(len(x)):
    for j in range(len(re_x)):
        if re_x[j] > x[i]:
            x[i] = re_x[j]-0.1
            break


data = np.column_stack((x, y))  # (630,2)

df = pd.DataFrame(data, columns=['time (h)', 'accuracy'])

x = list(set(x))
x = sorted(x)

y_group = df.groupby("time (h)")
y_mean = y_group.mean()["accuracy"]
y_max = y_group.max()["accuracy"]
y_min = y_group.min()["accuracy"]
y_var = y_group.var()["accuracy"]
y_std = y_group.std()["accuracy"]
plt.plot(x, np.array(y_mean)/max_acc, color='darkblue',  linewidth = 10, label='Fixed',linestyle = (0, (1, 1)))
# plt.plot(time, sin_waves, 'b-', label='ideal')
plt.fill_between(x, np.array(y_min)/max_acc, np.array(y_max)/max_acc, color='c', alpha=0.2)



time = []
data = new_dwrm
tmp = -1 # 记录最后一个访问的idx
y = [float(i) for i in data[3]]
for idx in data[2]:
    id = data[2].index(idx)
    d = data[0][id]
    w = data[1][id]
    time = sum_duration(d, w, idx, tmp, time, "Dyna-A-Freeze","semeval")
    tmp = idx
    
time = np.array(time)
time, y = cut(time, y, max_acc)

plt.plot(time, np.array(y)/max_acc, color='darkred', linewidth = 10, label = sys)

print("Dyna-A-Freeze",time[-1])

plt.legend(fontsize=ft,ncol = 1,frameon=False,loc="lower right")
plt.savefig('../figs/eval-ablation-trial-semeval.pdf', bbox_inches="tight")

# %%



