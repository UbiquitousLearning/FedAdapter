# %%
import argparse
from cProfile import run
import logging
import os
from time import sleep
import numpy as np
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# %%
def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    return parser.parse_args()


def wait_for_the_training_process_shallow(args):
    pipe_path = "./tmp/fedml-shallow-{args.depth}-{args.time_threshold}".format(args=args)
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

def wait_for_the_training_process_deep(args):
    pipe_path = "./tmp/fedml-deep-{args.depth}-{args.time_threshold}".format(args=args)
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


def get_acc(type = "shallow"):
    eval_file_path = "./tmp/fedavg_onto_output_" + type + "/eval_results.txt"
    # eval_file_path = "/tmp/fedavg_20news_output/eval_results.txt"
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

# kill -9 $(ps -ef|grep "fedavg"| awk '{print $2}')
# os.system("kill -9 $(ps -ef|grep \"fedavg\"| awk '{print $2}')")

# logging.basicConfig(level=logging.INFO,
#                     format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
#                     datefmt='%Y-%m-%d,%H:%M:%S')

parser = argparse.ArgumentParser()
args = add_args(parser)

args.round = 0 
args.depth = 0
args.time_threshold = 60
args.max_round = 2000
args.expand = 1 # time_thereshold的膨胀系数，actual time_threshold is time_threshold * (expand * depth + 1). 越深的层应该更稳定，可以减少trial频率

print("Running args is %s" % str(args))

# %%
model_size_32 = np.array([0.02 + i*0.05 for i in range(0,13)]) * 4
latency_tx2_cached = np.array([0.02, 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.90, 0.99, 1.08])
bw = 1 # both for upload and download bandwidth
batch_num = 100 # per round
# overhead per round
comm = model_size_32 * 2 / bw
comp = latency_tx2_cached * batch_num

round = args.round
depth = args.depth 
# freeze_layers[0] is depth list
# freeze_layers[1] is round list
# freeze_layers[2] is depth_shallow or depth_deep
freeze_layers = [[depth],[round],depth] 
# freeze_layers = [[0, 0, 0, 0], [0, 27, 54, 81], 1]

time_threshold = args.time_threshold # trail_freq. Unit: S

expand = args.expand



run_id = 0
while freeze_layers[1][-1] < args.max_round: # max_round = 1000
    print("Begin Trail", run_id, ":\n")
    print(freeze_layers)
    os.system("perl -p -i -e 's/Trail[0-9]*/Trail{args.depth}{args.time_threshold}/g' /home/cdq/FedNLP/experiments/distributed/transformer_exps/run_tc_exps/fedavg_main_tc.py".format(args=args)) # wandb name
    depth_shallow = freeze_layers[0][-1]
    depth_deep = depth_shallow + 1

    time_threshold = args.time_threshold * (expand * freeze_layers[0][-1] + 1) # depth = freeze_layers[0][-1]

    delta_round_shallow = int(time_threshold // (comm[depth_shallow] + comp[depth_shallow]))
    delta_round_deep = int(time_threshold // (comm[depth_deep] + comp[depth_deep]))
    print("Current time_threshold is: ", time_threshold)
    print("delta_round_shallow is: ", delta_round_shallow)
    print("delta_round_deep is: ", delta_round_deep)
    current_round = freeze_layers[1][-1]
    round_shallow = current_round + delta_round_shallow
    round_deep = current_round + delta_round_deep

    freeze_layers[2] = depth_shallow
    hp_shallow = 'FedAvg "uniform" 0.1 1 0.5 ' + str(delta_round_shallow) + ' 5 ' + remove_space(str([freeze_layers[0][-1]]).replace(',','.')+','+str([freeze_layers[1][-1]]).replace(',','.')+','+str(freeze_layers[2]))+" "+str(args.depth)+" "+str(args.time_threshold) # linux 读取输入的时候以，为分隔符，需要替换掉

    freeze_layers[2] = depth_deep
    hp_deep = 'FedAvg "uniform" 0.1 1 0.5 ' + str(delta_round_deep) + ' 5 ' + remove_space(str([freeze_layers[0][-1]]).replace(',','.')+','+str([freeze_layers[1][-1]]).replace(',','.')+','+str(freeze_layers[2]))+" "+str(args.depth)+" "+str(args.time_threshold)

    os.system("mkdir ./tmp/; touch ./tmp/fedml-shallow-{args.depth}-{args.time_threshold}; touch ./tmp/fedml-deep-{args.depth}-{args.time_threshold}; mkdir ./results/BERT/Trail-{args.depth}-{args.time_threshold}; mkdir ./results/BERT/Trail-{args.depth}-{args.time_threshold}/size-32".format(args=args))
    
    args.run_id = run_id
    args.hp = hp_shallow
    os.system("perl -p -i -e 's/pipe_path = .*/pipe_path = \".\/tmp\/fedml-shallow-{args.depth}-{args.time_threshold}\"/g' /home/cdq/FedNLP/FedML/fedml_api/distributed/fedavg/utils.py".format(args=args)) # pipe tmp name
    print('nohup sh run_seq_tagging_shallow.sh '
                '{args.hp} '
                '> ./results/BERT/Trail-{args.depth}-{args.time_threshold}/size-32/fednlp_tc_shallow_{args.run_id}.log 2>&1 &'.format(args=args))
    os.system('nohup sh run_seq_tagging_shallow.sh '
                '{args.hp} '
                '> ./results/BERT/Trail-{args.depth}-{args.time_threshold}/size-32/fednlp_tc_shallow_{args.run_id}.log 2>&1 &'.format(args=args))

    sleep(20) # 防止tmp文件互相干扰

    args.hp = hp_deep
    os.system("perl -p -i -e 's/pipe_path = .*/pipe_path = \".\/tmp\/fedml-deep-{args.depth}-{args.time_threshold}\"/g' /home/cdq/FedNLP/FedML/fedml_api/distributed/fedavg/utils.py".format(args=args)) # pipe tmp name
    print('nohup sh run_seq_tagging_deep.sh '
                '{args.hp} '
                '> ./results/BERT/Trail-{args.depth}-{args.time_threshold}/size-32/fednlp_tc_deep_{args.run_id}.log 2>&1 &'.format(args=args))
    os.system('nohup sh run_seq_tagging_deep.sh '
                '{args.hp} '
                '> ./results/BERT/Trail-{args.depth}-{args.time_threshold}/size-32/fednlp_tc_deep_{args.run_id}.log 2>&1 &'.format(args=args))

    wait_for_the_training_process_deep(args)
    wait_for_the_training_process_shallow(args)

    acc_shallow = get_acc("shallow-{args.depth}-{args.time_threshold}".format(args=args)) # os.read
    acc_deep = get_acc("deep-{args.depth}-{args.time_threshold}".format(args=args)) # os.read

    if acc_shallow >= acc_deep:
        round = round_shallow
        depth = depth_shallow
    else:
        round = round_deep
        depth = depth_deep

    freeze_layers[0].append(depth)
    freeze_layers[1].append(round)
    print("acc_shallow is ", str(acc_shallow))
    print("acc_deep is ", str(acc_deep))
    print("End Trail", run_id, ".\n")
    
    sleep(5)
    run_id += 1
