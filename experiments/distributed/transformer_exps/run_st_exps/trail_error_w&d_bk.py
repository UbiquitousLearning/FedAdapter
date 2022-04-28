import argparse
from cProfile import run
import logging
import os
from time import sleep
import numpy as np
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    return parser.parse_args()


def wait_for_the_training_process(type, args):
    args.type = type
    pipe_path = "./tmp/{args.dataset}-fedml-{args.type}-{args.depth}-{args.time_threshold}".format(args=args)
    pipe_fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(pipe_fd) as pipe:
        while True:
            message = pipe.read()
            if message:
                print("Received: '%s'\n" % message)
                print("Training is finished. Start the next training with...")
                if os.path.exists(pipe_path):
                    os.remove(pipe_path)
                return
            sleep(3)
            # print("Daemon is alive. Waiting for the training result.")


def get_acc(args, type = "shallow"):
    eval_file_path = "./tmp/{args.dataset}_fedavg_output_".format(args=args) + type + "/eval_results.txt"
    file_fd = os.open(eval_file_path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(file_fd) as file:
        while True:
            message = file.read()
            if message:
                print("Newest Performance: \n%s\n" % message)
                idx_acc = message.split().index("f1_score")
                print("Newest f1_score is: %s\n" % message.split()[idx_acc + 2])
                # os.remove(eval_file_path)
                acc = message.split()[idx_acc + 2]
                return acc
            sleep(3)

def remove_space(s):
    s_no_space = ''.join(s.split())
    return s_no_space

def run(type, args):
    args.type = type
    if type == "shallow":
        args.hp = hp_shallow
    if type == "deep":
        args.hp = hp_deep
    if type == "wide":
        args.hp = hp_wide
    
    # os.system("perl -p -i -e 's/pipe_path = .*/pipe_path = \".\/tmp\/{args.dataset}-fedml-{args.type}-{args.depth}-{args.time_threshold}\"/g' /home/cdq/FedNLP/FedML/fedml_api/distributed/fedavg/utils.py".format(args=args)) # pipe tmp name
    print('nohup sh run_seq_tagging_trial.sh '
                '{args.hp} '
                '> ./results/BERT/{args.dataset}-Trail-{args.depth}-{args.time_threshold}/fednlp_tc_{args.type}_{args.run_id}.log 2>&1 &'.format(args=args))
    os.system('nohup sh run_seq_tagging_trial.sh '
                '{args.hp} '
                '> ./results/BERT/{args.dataset}-Trail-{args.depth}-{args.time_threshold}/fednlp_tc_{args.type}_{args.run_id}.log 2>&1 &'.format(args=args))
    
    sleep(20) # 防止tmp文件相互干扰

def remove_cache_model(args):
    os.system("rm -rf .\/tmp\/{args.dataset}_fedavg_output_wide-{args.depth}-{args.time_threshold}".format(args=args))
    os.system("rm -rf .\/tmp\/{args.dataset}_fedavg_output_shallow-{args.depth}-{args.time_threshold}".format(args=args))
    os.system("rm -rf .\/tmp\/{args.dataset}_fedavg_output_deep-{args.depth}-{args.time_threshold}".format(args=args))

def set_hp(delta_round, freeze_layers, args):
        hp = 'FedAvg "niid_label_clients=30_alpha=1.0" 0.1 1 0.5 ' + str(delta_round) + ' 5 ' + remove_space(str([freeze_layers[2]]).replace(',','.')+','+str([-1]).replace(',','.')+','+str(freeze_layers[2]))+','+str(freeze_layers[3][-1]).replace(',','.')+" "+str(args.depth)+" "+str(args.time_threshold) +" "+str(args.type)# linux 读取输入的时候以，为分隔符，需要替换掉
        return hp

def inherit_model(winner_type, args): # from the winner
    args.type = winner_type
    for t in ["shallow", "deep", "wide"]:
        if t == args.type:
            pass
        else:
            args.t = t
            os.system("rm -rf .\/tmp\/{args.dataset}_fedavg_output_{args.t}-{args.depth}-{args.time_threshold}".format(args=args))
            os.system("cp -r .\/tmp\/{args.dataset}_fedavg_output_{args.type}-{args.depth}-{args.time_threshold} .\/tmp\/{args.dataset}_fedavg_output_{args.t}-{args.depth}-{args.time_threshold}".format(args=args))

def kill_process():
    # kill -9 $(ps -ef|grep "fedavg_main_st"| awk '{print $2}')
    os.system("kill -9 $(ps -ef|grep \"fedavg_main_st\"| awk '{print $2}')")

def skil_trial(depth): # use different trial freq to distinguish
    if depth == 1:
        return depth + 1
    else:
        return depth

# logging.basicConfig(level=logging.INFO,
#                     format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
#                     datefmt='%Y-%m-%d,%H:%M:%S')


parser = argparse.ArgumentParser()
args = add_args(parser)

args.dataset = "onto"
args.round = -1 
args.depth = 1
args.width = 8
args.time_threshold = 60
args.max_round = 3000
args.expand = 3 # time_thereshold的膨胀系数，actual time_threshold is time_threshold * (expand * depth + 1). 越深的层应该更稳定，可以减少trial频率

filename = "results/{args.dataset}-depth-{args.depth}-freq-{args.time_threshold}.log".format(args=args)
f=open(filename,"w+")

print("Running args is %s" % str(args),file=f,flush=True)


round = args.round
depth = args.depth 
width = args.width

latency_tx2_cached = np.array([0.02, 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.90, 0.99, 1.08])
bw = 1 # both for upload and download bandwidth
batch_num = 20 # per round
# overhead per round

comp = latency_tx2_cached * batch_num

# freeze_layers[0] is depth list
# freeze_layers[1] is round list
# freeze_layers[2] is depth_shallow or depth_deep
freeze_layers = [[depth],[round],depth,[width]] 
# freeze_layers = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [-1, 105, 211, 317, 423, 529, 635, 741, 847, 953, 1059, 1165, 1271, 1377, 1483, 1589, 1695, 1801, 1907, 2013, 2119, 2225, 2331, 2437, 2543, 2649, 2755, 2861, 2967], 0, [8, 16, 24, 32, 32, 40, 48, 48, 48, 48, 48, 56, 56, 56, 56, 56, 56, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]]

time_threshold = args.time_threshold # trail_freq. Unit: S

expand = args.expand

# remove_cache_model(args)

run_id = 0
# metric = [0, '0.4675081813931744', '0.5148785425101214', '0.5361279646993933', '0.548293464117706', '0.5567368839427663', '0.5619572229666021', '0.5679416579848191', '0.5715295305401311', '0.5744852644327816', '0.577152083964073', '0.5795848620395034', '0.5832243789600723', '0.5836266720305743', '0.5840164965045517', '0.5853879440474993', '0.5866666666666666', '0.5870012547051442', '0.5878988561107767', '0.5880995384306642', '0.5887', '0.5896226415094339', '0.5900356228990016', '0.5903777655144735', '0.5910295514775739', '0.5915929424701354', '0.5927338418305901', '0.593615098885654', '0.5939053165319543']
metric = [0]

step = 3 #每次depth的增加数量 

while freeze_layers[1][-1] < args.max_round: # max_round = 1000
    print("Begin Trail", run_id, ":\n",file=f,flush=True)
    width = freeze_layers[3][-1]
    model_size_32 = np.array([0.02 + i*0.05*width/32 for i in range(0,13)]) * 4
    comm = model_size_32 * 2 / bw
    print(freeze_layers,file=f,flush=True)
    print(metric,file=f,flush=True)
    # os.system("perl -p -i -e 's/Trail[0-9]*/Trail{args.depth}{args.time_threshold}/g' /home/cdq/FedNLP/experiments/distributed/transformer_exps/run_tc_exps/fedavg_main_tc.py".format(args=args)) # wandb name
    depth_shallow = freeze_layers[0][-1]
    depth_deep = depth_shallow + step
    # depth_deep = skil_trial(depth_deep)

    time_threshold = args.time_threshold * (expand * freeze_layers[0][-1] + 1) # depth = freeze_layers[0][-1]

    delta_round_shallow = int(time_threshold // (comm[depth_shallow] + comp[depth_shallow]))
    delta_round_deep = int(time_threshold // (comm[depth_deep] + comp[depth_deep]))

    model_size_32_wide = np.array([0.02 + i*0.05*(width+8)/32 for i in range(0,13)]) * 4
    comm_wide = model_size_32_wide * 2 / bw
    # print(comm,file=f,flush=True)
    # print(comm_wide,file=f,flush=True)
    delta_round_wide = int(time_threshold // (comm_wide[depth_shallow] + comp[depth_shallow]))

    print("Current time_threshold is: ", time_threshold,file=f,flush=True)
    print("delta_round_shallow is: ", delta_round_shallow,file=f,flush=True)
    print("delta_round_deep is: ", delta_round_deep,file=f,flush=True)
    print("delta_round_wide is: ", delta_round_wide,file=f,flush=True)

    current_round = freeze_layers[1][-1]
    round_shallow = current_round + delta_round_shallow
    round_deep = current_round + delta_round_deep
    round_wide = current_round + delta_round_wide

    

    freeze_layers[2] = depth_shallow
    args.type = "shallow"
    hp_shallow = set_hp(delta_round_shallow, freeze_layers, args)


    freeze_layers[2] = depth_deep
    args.type = "deep"
    hp_deep = set_hp(delta_round_deep, freeze_layers, args)

    freeze_layers[2] = depth_shallow
    args.type = "wide"
    freeze_layers[3][-1] = freeze_layers[3][-1] + 8
    hp_wide = set_hp(delta_round_wide, freeze_layers, args)
    freeze_layers[3][-1] = freeze_layers[3][-1] - 8
    
    os.system("mkdir ./tmp/; \
    touch ./tmp/{args.dataset}-fedml-shallow-{args.depth}-{args.time_threshold}; \
    touch ./tmp/{args.dataset}-fedml-wide-{args.depth}-{args.time_threshold}; \
    touch ./tmp/{args.dataset}-fedml-deep-{args.depth}-{args.time_threshold}; \
    mkdir -p ./results/BERT/{args.dataset}-Trail-{args.depth}-{args.time_threshold}".format(args=args))
    
    args.run_id = run_id

    run("shallow", args)
    run("deep", args)
    run("wide", args)

    wait_for_the_training_process("deep", args)
    wait_for_the_training_process("wide", args)
    wait_for_the_training_process("shallow", args)

    acc_shallow = get_acc(args, "shallow-{args.depth}-{args.time_threshold}".format(args=args)) # os.read
    acc_deep = get_acc(args, "deep-{args.depth}-{args.time_threshold}".format(args=args)) # os.read
    acc_wide = get_acc(args, "wide-{args.depth}-{args.time_threshold}".format(args=args)) # os.read

    print("acc_shallow is ", str(acc_shallow),file=f,flush=True)
    print("acc_deep is ", str(acc_deep),file=f,flush=True)
    print("acc_wide is ", str(acc_wide),file=f,flush=True)

    acc_winner = max(acc_shallow, acc_deep, acc_wide)
    print("acc_winner is ", acc_winner, file=f,flush=True)
    if acc_winner == acc_shallow:
        print("winner is shallow",file=f,flush=True)
        round = round_shallow
        depth = depth_shallow
        inherit_model("shallow", args)
    elif acc_winner == acc_wide:
        print("winner is wide",file=f,flush=True)
        round = round_wide
        depth = depth_shallow
        width = width + 8
        inherit_model("wide", args)
    elif acc_winner == acc_deep:
        print("winner is deep",file=f,flush=True)
        round = round_deep
        depth = depth_deep
        inherit_model("deep", args)

    freeze_layers[0].append(depth)
    freeze_layers[1].append(round-1) # 最后一轮会被停掉，所以虽然跑了 round 轮，但是其实本地的模型存的是round-1轮的evaluation结果
    freeze_layers[3].append(width)
    metric.append(acc_winner)
# %%
    
    print("End Trail", run_id, ".\n",file=f,flush=True)
    
    sleep(5)
    run_id += 1
