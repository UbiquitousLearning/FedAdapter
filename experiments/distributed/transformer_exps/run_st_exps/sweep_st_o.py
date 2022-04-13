import argparse
import logging
import os
from time import sleep


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    return parser.parse_args()


def wait_for_the_training_process(args):
    pipe_path = "./tmp/fedml-{args.width}".format(args=args)
    if not os.path.exists(os.path.dirname(pipe_path)):
        try:
            os.makedirs(os.path.dirname(pipe_path))
        except OSError as exc:  # Guard against race condition
            print(exc)
    if not os.path.exists(pipe_path):
        open(pipe_path, 'w').close()
    pipe_fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(pipe_fd) as pipe:
        while True:
            message = pipe.read()
            if message:
                print("Received: '%s'" % message)
                print("Training is finished. Start the next training with...")
                os.remove(pipe_path)
                return
            sleep(3)
            # print("Daemon is alive. Waiting for the training result.")


# customize the log format
logging.basicConfig(level=logging.INFO,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

parser = argparse.ArgumentParser()
args = add_args(parser)

os.system("kill $(ps aux | grep \"fedavg_main_st.py\" | grep -v grep | awk '{print $2}')")

hps = [
    # running
    # 'FedOPT "uniform" 5e-5 1 0.5 20',

    # to do
    'FedOPT "niid_label_clients=30_alpha=0.1" 5e-5 1 0.5 20',
    'FedOPT "niid_label_clients=30_alpha=0.01" 5e-5 1 0.5 20',

    'FedProx "uniform" 1e-1 1 0.5 20',
    'FedProx "niid_label_clients=30_alpha=0.1" 1e-1 1 0.5 20',
    'FedProx "niid_label_clients=30_alpha=0.01" 1e-1 1 0.5 20',

    'FedAvg "uniform" 1e-1 1 0.5 20',
    'FedAvg "niid_label_clients=30_alpha=0.1" 1e-1 1 0.5 20',
    'FedAvg "niid_label_clients=30_alpha=0.01" 1e-1 1 0.5 20',

    # 'FedProx "niid_label_clients=30_alpha=0.1" 1e-1 1 1 30',
    # 'FedProx "niid_label_clients=30_alpha=0.1" 1e-1 1 0.1 30',
    # 'FedProx "niid_label_clients=30_alpha=0.1" 1e-1 1 0.01 30',
]

hps_p = [
    'FedAvg "niid_label_clients=30_alpha=0.1" 1e-1 1 0.5 400 10',
    'FedAvg "niid_label_clients=30_alpha=0.1" 1e-1 1 0.5 400 8',
    'FedAvg "niid_label_clients=30_alpha=0.1" 1e-1 1 0.5 400 4',
    'FedAvg "niid_label_clients=30_alpha=0.1" 1e-1 1 0.5 400 2',
    'FedAvg "niid_label_clients=30_alpha=0.1" 1e-1 1 0.5 400 1',
]

run_id = 0
width = "ost"
for hp in hps_p[::-1]:
    args.hp = hp
    args.run_id = run_id
    args.width = width
    args.clients = hp.split()[-1]

    logging.info("hp = %s" % args.hp)
    os.system("perl -p -i -e 's/pipe_path = .*/pipe_path = \".\/tmp\/fedml-{args.width}\"/g' /home/cdq/FedNLP/FedML/fedml_api/distributed/fedavg/utils.py".format(args=args)) # pipe tmp name
    os.system("mkdir ./tmp/; touch ./tmp/fedml")
    os.system('nohup sh run_seq_tagging_origin.sh '
              '{args.hp} '
              '> ./results/fednlp_st_{args.clients}.log 2>&1 &'.format(args=args))

    wait_for_the_training_process(args)

    logging.info("cleaning the training...")

    # kill $(ps aux | grep fedavg_main_st.py | grep -v grep | awk '{print $2}')
    os.system("kill $(ps aux | grep \"fedavg_main_st.py\" | grep -v grep | awk '{print $2}')")

    sleep(5)
    run_id += 1
