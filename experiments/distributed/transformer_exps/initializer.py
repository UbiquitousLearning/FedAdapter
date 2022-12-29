import random

import numpy as np
import torch
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    BertForQuestionAnswering,
    DistilBertConfig,
    DistilBertTokenizer,
    DistilBertForQuestionAnswering,
    BartConfig, 
    BartForConditionalGeneration, 
    BartTokenizer,
)
import logging

from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_FedAvg_distributed
from FedML.fedml_api.distributed.fedopt.FedOptAPI import FedML_FedOpt_distributed
from FedML.fedml_api.distributed.fedprox.FedProxAPI import FedML_FedProx_distributed
from model.transformer.bert_model import BertForSequenceClassification
from model.transformer.distilbert_model import DistilBertForSequenceClassification
from transformers import AutoModelWithHeads, AutoModel
from transformers.adapters.composition import Stack, Parallel


def get_fl_algorithm_initializer(alg_name):
    if alg_name == "FedAvg":
        fl_algorithm = FedML_FedAvg_distributed
    elif alg_name == "FedOPT":
        fl_algorithm = FedML_FedOpt_distributed
    elif alg_name == "FedProx":
        fl_algorithm = FedML_FedProx_distributed
    else:
        raise Exception("please do sanity check for this algorithm.")

    return fl_algorithm

def create_model_o(args, formulation="classification"):
    # create model, tokenizer, and model config (HuggingFace style)
    MODEL_CLASSES = {
        "classification": {
            "bert": (BertConfig, AutoModelWithHeads, BertTokenizer),
            "distilbert": (DistilBertConfig, AutoModelWithHeads, DistilBertTokenizer), 
            # AutoModelWithHeads for adapter, DistilBertForSequenceClassification for origin
            # "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
            # "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
        },
        "seq_tagging": {
            "bert": (BertConfig, AutoModelWithHeads, BertTokenizer),
            "distilbert": (DistilBertConfig, AutoModelWithHeads, DistilBertTokenizer),
            # AutoModelWithHeads for adapter, DistilBertForTokenClassification for origin
        },
        "span_extraction": {
            "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
        },
        "seq2seq": {
            "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),
            # BartForConditionalGeneration
        }
    }
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[formulation][
        args.model_type]
    import os

    inherit = True # trial&error
    if args.evaluate_during_training_steps == 300: # setup
        inherit = False
    
    if args.evaluate_during_training_steps == 200: # baseline
        inherit = True

    # if False:
    if os.path.exists(os.path.join(args.output_dir, "pytorch_model.bin")) and inherit == True:
        logging.info("There is trained models(adapters), loaded!")
        config = config_class.from_pretrained(args.output_dir)
        model = model_class.from_pretrained(args.output_dir)
        adapter_list = []
        for i in range(int(64/8)):
            adapter_list.append(str(i))
        # model.set_active_adapters(adapter_list)
        model.train_adapter(adapter_list)
        if formulation != "seq2seq":
            tokenizer = tokenizer_class.from_pretrained(
                args.model_name, do_lower_case=args.do_lower_case, local_files_only=True)
        else:
            tokenizer = [None, None]
            tokenizer[0] = tokenizer_class.from_pretrained(args.model_name, local_files_only=True)
            tokenizer[1]= tokenizer[0]
    else:
        config = config_class.from_pretrained(args.model_name, **args.config)
        model = model_class.from_pretrained(args.model_name, config=config)
        width = 64
        u_adapter_size = 8 # 单位宽度的adapter
        rf = int(768 / u_adapter_size)

        adapter_num = int(width / u_adapter_size)
        
        adapter_config = {'original_ln_before':True, 'original_ln_after':True, 'residual_before_ln':True, 'adapter_residual_before_ln':False, 'ln_before':False, 'ln_after':False, 'mh_adapter':False, 'output_adapter':True, 'non_linearity':'relu', 'reduction_factor':rf, 'inv_adapter':None, 'inv_adapter_reduction_factor':None, 'cross_adapter':False, 'leave_out':[]} # [0,1,2,3,4,5,6,7,8,9,10,11]
        if args.dataset == "20news":
            num_labels = 20
        if args.dataset == "onto":
            num_labels = 37
        if args.dataset == "agnews":
            num_labels = 4
        if args.dataset == "semeval_2010_task8":
            num_labels = 19
        adapter_list = []
        for i in range(adapter_num):
            model.add_adapter(str(i),config=adapter_config)
            adapter_list.append(str(i)) 

        model.add_classification_head("0", num_labels=num_labels, layers=1)
        
        
        # model.set_active_adapters(adapter_list)
        model.train_adapter(adapter_list)
        if formulation != "seq2seq":
            tokenizer = tokenizer_class.from_pretrained(
                args.model_name, do_lower_case=args.do_lower_case, local_files_only=False)
        else:
            tokenizer = [None, None]
            tokenizer[0] = tokenizer_class.from_pretrained(args.model_name)
            tokenizer[1]= tokenizer[0]
    logging.info(model)
        
    return config, model, tokenizer

def create_model(args, formulation="classification"):
    # create model, tokenizer, and model config (HuggingFace style)
    MODEL_CLASSES = {
        "classification": {
            "bert": (BertConfig, BertForSequenceClassification , BertTokenizer),
            "distilbert": (DistilBertConfig, AutoModelWithHeads, DistilBertTokenizer), 
            # AutoModelWithHeads for adapter, DistilBertForSequenceClassification for origin
            # "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
            # "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
        },
        "seq_tagging": {
            "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
            "distilbert": (DistilBertConfig, AutoModelWithHeads, DistilBertTokenizer),
            # AutoModelWithHeads for adapter, DistilBertForTokenClassification for origin
        },
        "span_extraction": {
            "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
        },
        "seq2seq": {
            "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),
            # BartForConditionalGeneration
        }
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES[formulation][
        args.model_type]
    # config = config_class.from_pretrained(
    #     args.model_name, num_labels=args.num_labels, **args.config)
    
    inherit = False
    # if self.args.evaluate_during_training_steps != 200 and self.args.evaluate_during_training_steps != 300:
    #     inherit = True

    import os
    if os.path.exists(os.path.join(args.output_dir, "pytorch_model.bin")) and inherit == True:
        logging.info("There is trained models(adapters), loaded!")
        config = config_class.from_pretrained(args.output_dir)
        model = model_class.from_pretrained(args.output_dir)
        adapter_list = []
        for i in range(int(64/8)):
            adapter_list.append(str(i))
        model.set_active_adapters(adapter_list)
        model.train_adapter(adapter_list)
        if formulation != "seq2seq":
            tokenizer = tokenizer_class.from_pretrained(
                args.model_name, do_lower_case=args.do_lower_case, local_files_only=True)
        else:
            tokenizer = [None, None]
            tokenizer[0] = tokenizer_class.from_pretrained(args.model_name, local_files_only=True)
            tokenizer[1]= tokenizer[0]
    else:
        config = config_class.from_pretrained(args.model_name, **args.config)
        model = model_class.from_pretrained(args.model_name, config=config)
        width = 64
        u_adapter_size = 8 # 单位宽度的adapter
        rf = int(768 / u_adapter_size)

        adapter_num = int(width / u_adapter_size)

        adapter_config = {'original_ln_before':True, 'original_ln_after':True, 'residual_before_ln':True, 'adapter_residual_before_ln':False, 'ln_before':False, 'ln_after':False, 'mh_adapter':False, 'output_adapter':True, 'non_linearity':'relu', 'reduction_factor':rf, 'inv_adapter':None, 'inv_adapter_reduction_factor':None, 'cross_adapter':False, 'leave_out':[]} # [0,1,2,3,4,5,6,7,8,9,10,11]

        adapter_list = []
        for i in range(adapter_num):
            model.add_adapter(str(i),config=adapter_config)
            adapter_list.append(str(i))

        # model.set_active_adapters(adapter_list)
        model.train_adapter(adapter_list)
        if formulation != "seq2seq":
            tokenizer = tokenizer_class.from_pretrained(
                args.model_name, do_lower_case=args.do_lower_case, local_files_only=True)
        else:
            tokenizer = [None, None]
            tokenizer[0] = tokenizer_class.from_pretrained(args.model_name)
            tokenizer[1]= tokenizer[0]

    



    # model.add_adapter("0",config=adapter_config)
    # model.add_adapter("0",config=adapter_config)
    # model.add_adapter("0",config=adapter_config)
    # model.add_adapter("0",config=adapter_config)
    # model.add_adapter("0",config=adapter_config)
    # model.add_adapter("0",config=adapter_config)
    # model.train_adapter("a0")

    # Dynamic Adapter
    # adapter_config = {'original_ln_before':True, 'original_ln_after':True, 'residual_before_ln':True, 'adapter_residual_before_ln':False, 'ln_before':False, 'ln_after':False, 'mh_adapter':False, 'output_adapter':True, 'non_linearity':'relu', 'reduction_factor':16, 'inv_adapter':None, 'inv_adapter_reduction_factor':None, 'cross_adapter':False, 'leave_out':[]} # [0,1,2,3,4,5,6,7,8,9,10,11]
    # import os
    # if os.path.exists(os.path.join(args.output_dir, "adapter_config.json")):
    #     logging.info("There is trained adapters, loaded!")
    #     model.load_adapter(args.output_dir)
    # else:
    #     model.add_adapter("0",config=adapter_config)

    # model.train_adapter("0")

    # Variable width adapter
    

    # model.add_adapter("a1",config=adapter_config)
    # model.add_classification_head(
    #     "a1",
    #     num_labels=20, # 20 for TC; ? for ST.
    #     layers=1
    # )

    # model.add_adapter("a2",config=adapter_config)
    # model.add_classification_head(
    #     "a2",
    #     num_labels=20, # 20 for TC; ? for ST.
    #     layers=1
    # )
    # model.add_adapter("a3",config=adapter_config)
    # model.add_adapter("a4",config=adapter_config)
    # cd ~/FedNLP/experiments/distributed/transformer_exps/run_tc_exps && conda activate fednlp && sh run_text_classification_freeze.sh FedAvg "uniform" 0.1 1 400 5 e,0,1,2,3,4,5,6
    # model.set_active_adapters(Parallel("a0", "a1", "a2"))
    # model.train_adapter(Parallel("a0", "a1", "a2"))

    # model.set_active_adapters(Stack("a0", "a1", "a2", "a3", "a4"))
    # model.train_adapter(Stack("a0", "a1", "a2", "a3", "a4"))

    # Adapter Fusion
    # from transformers.adapters.composition import Fuse
    # model.load_adapter("nli/multinli@ukp", load_as="multinli", with_head=False)
    # model.load_adapter("sts/qqp@ukp", with_head=False)
    # model.load_adapter("nli/qnli@ukp", with_head=False)
    # # Add a fusion layer for all loaded adapters
    # model.add_adapter_fusion(Fuse("multinli", "qqp", "qnli"))
    # model.set_active_adapters(Fuse("multinli", "qqp", "qnli"))

    # # model.add_classification_head("cb", num_labels=20,layers=1)

    # adapter_setup = Fuse("multinli", "qqp", "qnli")
    # model.train_adapter_fusion(adapter_setup)   

    # Vanilla Adapter
    # model.add_adapter("rotten_tomatoes")

    # model.add_classification_head(
    #     "rotten_tomatoes",
    #     num_labels=20,
    #     layers=1
    # )
    # model.train_adapter("rotten_tomatoes")

    # logging.info(model)
    
    
    logging.info(model)
    return config, model, tokenizer


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_federated_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # # PipeTransformer related
    parser.add_argument("--run_id", type=int, default=0)

    parser.add_argument("--is_debug_mode", default=0, type=int,
                        help="is_debug_mode")

    # Data related
    # TODO: list all dataset names:
    parser.add_argument('--dataset', type=str, default='agnews', metavar='N',
                        help='dataset used for training')

    parser.add_argument(
        '--data_file_path', type=str,
        default='/home/bill/fednlp_data/data_files/agnews_data.h5',
        help='data h5 file path')

    parser.add_argument(
        '--partition_file_path', type=str,
        default='/home/bill/fednlp_data/partition_files/agnews_partition.h5',
        help='partition h5 file path')

    parser.add_argument('--partition_method', type=str, default='uniform',
                        help='partition method')

    # Model related
    parser.add_argument('--model_type', type=str, default='bert', metavar='N',
                        help='transformer model type')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', metavar='N',
                        help='transformer model name')
    parser.add_argument('--do_lower_case', type=bool, default=True, metavar='N',
                        help='transformer model name')

    # Learning related
    parser.add_argument('--train_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--eval_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for evaluation (default: 8)')

    parser.add_argument('--max_seq_length', type=int, default=128, metavar='N',
                        help='maximum sequence length (default: 128)')

    parser.add_argument('--n_gpu', type=int, default=1, metavar='EP',
                        help='how many gpus will be used ')

    parser.add_argument('--fp16', default=False, action="store_true",
                        help='if enable fp16 for training')
    parser.add_argument('--manual_seed', type=int, default=42, metavar='N',
                        help='random seed')

    # IO related
    parser.add_argument('--output_dir', type=str, default="/tmp/", metavar='N',
                        help='path to save the trained results and ckpts')

    # Federated Learning related
    parser.add_argument('--fl_algorithm', type=str, default="FedAvg",
                        help='Algorithm list: FedAvg; FedOPT; FedProx ')

    parser.add_argument('--backend', type=str, default="MPI",
                        help='Backend for Server and Client')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=1,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--client_num_in_total', type=int, default=-1, metavar='NN',
                        help='number of clients in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int,
                        default=4, metavar='NN', help='number of workers')

    parser.add_argument('--epochs', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, metavar='EP',
                        help='how many steps for accumulate the loss.')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='Optimizer used on the client. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate on the client (default: 0.001)')

    parser.add_argument('--weight_decay', type=float, default=0, metavar='N',
                        help='L2 penalty')

    parser.add_argument('--server_optimizer', type=str, default='sgd',
                        help='Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--server_lr', type=float, default=0.1,
                        help='server learning rate (default: 0.001)')

    parser.add_argument('--server_momentum', type=float, default=0,
                        help='server momentum (default: 0)')

    parser.add_argument('--fedprox_mu', type=float, default=1,
                        help='server momentum (default: 1)')
    parser.add_argument(
        '--evaluate_during_training_steps', type=int, default=100, metavar='EP',
        help='the frequency of the evaluation during training')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    # GPU device management
    parser.add_argument('--gpu_mapping_file', type=str, default="gpu_mapping.yaml",
                        help='the gpu utilization file for servers and clients. If there is no \
                    gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str,
                        default="mapping_default",
                        help='the key in gpu utilization file')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    
    # cached related
    parser.add_argument('--reprocess_input_data',  action='store_true',
                        help='whether generate features')
    
    # freeze related
    parser.add_argument('--freeze_layers', type=str, default='', metavar='N',
                        help='freeze which layers')

    # trial & error
    parser.add_argument('--depth', type=int, default=0,
                        help='start-up depth')

    parser.add_argument('--width', type=int, default=8,
                        help='start-up width (adapter size)')
    
    parser.add_argument('--freq', type=int, default=60,
                        help='trial frequency (s)')

    parser.add_argument('--type', type=str, default='shallow', metavar='N',
                        help='the dirction of trial.')

    


    return parser
