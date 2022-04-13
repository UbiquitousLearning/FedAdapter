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

os.system("kill $(ps aux | grep \"fedavg_main_tc.py\" | grep -v grep | awk '{print $2}')")

# sh run_text_classification.sh FedAvg "niid_label_clients=100_alpha=5.0" 5e-5 0.1 30
# sh run_text_classification.sh FedAvg "niid_label_clients=100_alpha=10.0" 5e-5 0.1 30
# sh run_text_classification.sh FedAvg "niid_label_clients=100_alpha=1.0" 5e-5 0.1 30
# sh run_text_classification.sh FedAvg "uniform" 5e-5 0.1 30
# sh run_text_classification.sh FedAvg "niid_quantity_clients=100_beta=5.0" 5e-5 0.1 30

# sh run_text_classification.sh FedProx "niid_label_clients=100_alpha=5.0" 5e-5 0.1 30
# sh run_text_classification.sh FedProx "niid_label_clients=100_alpha=10.0" 5e-5 0.1 30
# sh run_text_classification.sh FedProx "niid_label_clients=100_alpha=1.0" 5e-5 0.1 30
# sh run_text_classification.sh FedProx "uniform" 5e-5 0.1 30
# sh run_text_classification.sh FedProx "niid_quantity_clients=100_beta=5.0" 5e-5 0.1 30

# sh run_text_classification.sh FedOPT "niid_label_clients=100_alpha=5.0" 5e-5 0.1 30
# sh run_text_classification.sh FedOPT "niid_label_clients=100_alpha=10.0" 5e-5 0.1 30
# sh run_text_classification.sh FedOPT "niid_label_clients=100_alpha=1.0" 5e-5 0.1 30
# sh run_text_classification.sh FedOPT "uniform" 5e-5 0.1 30
# sh run_text_classification.sh FedOPT "niid_quantity_clients=100_beta=5.0" 5e-5 0.1 30

hps = [
    # 'FedAvg "niid_label_clients=100_alpha=5.0" 5e-5 0.1 25', # finished by Zihang
    # 'FedAvg "niid_label_clients=100_alpha=10.0" 5e-5 0.1 25',
    # 'FedAvg "niid_label_clients=100_alpha=1.0" 5e-5 0.1 25',
    # 'FedProx "niid_label_clients=100_alpha=5.0" 5e-5 0.1 25',
    # 'FedProx "niid_label_clients=100_alpha=10.0" 5e-5 0.1 25',
    # 'FedProx "niid_label_clients=100_alpha=1.0" 5e-5 0.1 25',
    # 'FedOPT "niid_label_clients=100_alpha=5.0" 5e-5 0.1 25',
    # 'FedOPT "niid_label_clients=100_alpha=10.0" 5e-5 0.1 25',
    # 'FedOPT "niid_label_clients=100_alpha=1.0" 5e-5 0.1 25',
    # 'FedAvg "uniform" 5e-5 0.1 25',
    # 'FedProx "uniform" 5e-5 0.1 25',
    # 'FedOPT "uniform" 5e-5 0.1 25',
    # 'FedAvg "niid_quantity_clients=100_beta=5.0" 5e-5 0.1 25',
    # 'FedOPT "niid_quantity_clients=100_beta=5.0" 5e-5 0.1 25', # finished by Chaoyang
    # 'FedAvg "uniform" 0.1 1 400 10 e,0,1,2,3,4,5',
    # 'FedAvg "uniform" 0.1 1 400 10 e,0,1,2,3,4,5,6,7,8,9,10,11',
    # 'FedAvg "uniform" 0.1 1 400 5 e,0,1,2,3,4,5,6,7,8,9,10',
    # 'FedAvg "uniform" 0.1 1 400 5 e,0,1,2,3,4,5,6,7,8,9',
    'FedAvg "uniform" 0.1 1 400 5 e,0,1,2,3,4,5,6,7,8',
    # 'FedAvg "uniform" 0.1 1 400 5 e,0,1,2,3,4,5,6,7',
    'FedAvg "uniform" 0.1 1 400 5 e,0,1,2,3,4,5,6',
    # 'FedAvg "uniform" 0.1 1 400 5 e,0,1,2,3,4,5',
    'FedAvg "uniform" 0.1 1 400 5 e,0,1,2,3,4',
    # 'FedAvg "uniform" 0.1 1 400 5 e,0,1,2,3',
    'FedAvg "uniform" 0.1 1 400 5 e,0,1,2',
    # 'FedAvg "uniform" 0.1 1 400 5 e,0,1',
    'FedAvg "uniform" 0.1 1 400 5 e,0',
    'FedAvg "uniform" 0.1 1 400 5 e'
]

run_id = 0
width = 32
for hp in hps:
    args.width = width
    args.hp = hp
    args.run_id = run_id
    
    logging.info("hp = %s" % args.hp)
    logging.info("Width (adapter size) is %s.", str(width))
    os.system("perl -p -i -e 's/pipe_path = .*/pipe_path = \".\/tmp\/fedml-{args.width}\"/g' /home/cdq/FedNLP/FedML/fedml_api/distributed/fedavg/utils.py".format(args=args)) # pipe tmp name
    logging.info("perl -p -i -e 's/pipe_path = .*/pipe_path = \".\/tmp\/fedml-{args.width}\"/g' /home/cdq/FedNLP/FedML/fedml_api/distributed/fedavg/utils.py".format(args=args))

    
    os.system("mkdir ./tmp/; touch ./tmp/fedml-{args.width}; mkdir ./results/BERT/size-{args.width}-freeze".format(args=args))
    os.system('nohup sh run_text_classification_freeze.sh '
              '{args.hp} '
              '> ./results/BERT/size-{args.width}-freeze/fednlp_tc_{args.run_id}.log 2>&1 &'.format(args=args))

    wait_for_the_training_process(args)

    logging.info("cleaning the training...")
    os.system("kill $(ps aux | grep \"fedavg_main_tc.py\" | grep -v grep | awk '{print $2}')")

    sleep(5)
    run_id += 1
