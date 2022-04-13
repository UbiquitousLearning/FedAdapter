#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import copy
import logging
import math
import os
# from thop import profile
from torchstat import stat
from ptflops import get_model_complexity_info

import random    
import numpy as np
import sklearn
import torch
import wandb
from training.utils.text_classification_utils import *
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.profiler import profile, record_function, ProfilerActivity

import torchprof

class TextClassificationTrainer:
    def __init__(self, args, device, model, train_dl=None, test_dl=None):
        self.args = args
        self.device = device

        # set data
        self.num_labels = args.num_labels
        self.set_data(train_dl, test_dl)

        # model
        self.model = model

        # training results
        self.results = {}
        self.best_accuracy = 0.0

        # freeze
        self.freeze_layers = args.freeze_layers.split(",") if args.freeze_layers else []

    def set_data(self, train_dl=None, test_dl=None):
        # Used for fedtrainer
        self.train_dl = train_dl
        self.test_dl = test_dl

    def train_model(self, device=None, rounds=0):
        if not device:
            device = self.device
        logging.info("train_model self.device: " + str(device) + "; round" + str(rounds))
        self.model.to(device)

        # 10% Round
        # if rounds%10 == 0:
        #     self.args.epochs = 10
        # else:
        #     self.args.epochs = 1

        
        # # 10% Client
        # s = random.randint(0,9)
        # logging.info(s)
        # if s == 0:
        #     self.args.epochs = 10
        # else:
        #     self.args.epochs = 1

        # build optimizer and scheduler
        iteration_in_total = len(
            self.train_dl) // self.args.gradient_accumulation_steps * self.args.epochs
        optimizer, scheduler = self.build_optimizer(self.model, iteration_in_total,rounds)

        # training result
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        if self.args.fl_algorithm == "FedProx":
            global_model = copy.deepcopy(self.model)


        batch_chosen = random.sample(range(0, 29),8)
        # logging.info("Batch chosen is %s", str(batch_chosen))

        for epoch in range(0, self.args.epochs):
            for batch_idx, batch in enumerate(self.train_dl):
                # continue
                # logging.info("batch_idx %s", str(batch_idx))  
                
                                   
                # if batch_idx not in batch_chosen:
                #     continue


                # # epoch = 0.5
                # if rounds%2 == 0:
                #     if batch_idx >14:
                #         break
                # if rounds%2 == 1: 
                #     if batch_idx <= 14:
                #         continue
                
                self.model.train()

                # logging.info("load data")
                batch = tuple(t for t in batch)

                # dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                x = batch[1].to(device)

                # logging.info("x shape is %s", str(np.shape(x)))
                labels = batch[4].to(device)
                # logging.info("forward computing")
                # (loss), logits, (hidden_states), (attentions)

                # # `profile_memory` was added in PyTorch 1.6, this will output a runtime warning if unsupported.
                # with torchprof.Profile(self.model, use_cuda=False, profile_memory=False) as prof:
                #     output = self.model(x)
                # output = self.model(x)
                # logging.info(np.shape(x))
                # flops, params = get_model_complexity_info(self.model, (4,256,),as_strings=True,print_per_layer_stat=True)
 
                # print("|flops: %s |params: %s" % (flops,params))

                # flops, params = profile(self.model, inputs=(x, ), verbose=False)
                # logging.info("flops %f, params %f", flops, params)
                # equivalent to `print(prof)` and `print(prof.display())`
                # logging.info(prof.display(show_events=False))
                # logging.info("start forwarding")
                # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                #     # with record_function("model_inference"):
                #     output = self.model(x)
                # logging.info("forwarding done")
                # ogging.info(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
                output = self.model(x)
                # Stack
                # logits = output[0] 
                # Parallel
                logits = (output[0][0] + output[1][0])/2
                loss_fct = CrossEntropyLoss()
                # logging.info(self.num_labels)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if self.args.fl_algorithm == "FedProx":
                    fed_prox_reg = 0.0
                    mu = self.args.fedprox_mu
                    for (p, g_p) in zip(self.model.parameters(),
                                        global_model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((p - g_p.data)) ** 2)
                    loss += fed_prox_reg

                # model outputs are always tuple in pytorch-transformers (see doc)
                # loss = outputs[0]
                # logging.info(loss)
                current_loss = loss.item()
                logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
                                                                           len(self.train_dl), current_loss))

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                # logging.info("backward computing")

                # with torchprof.Profile(self.model, use_cuda=True, profile_memory=True) as prof:
                #     loss.backward()
                # logging.info(prof.display(show_events=False))
                # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                #     with record_function("backward"):
                #         loss.backward()
                # logging.info(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=50))
                loss.backward()
                # logging.info("backward computing done!")
                tr_loss += loss.item()
                # logging.info(global_step)
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.evaluate_during_training and (self.args.evaluate_during_training_steps > 0
                                                               and global_step % self.args.evaluate_during_training_steps == 0):
                        results, _, _ = self.eval_model(epoch, global_step)
                        logging.info(results)

                if self.args.is_debug_mode == 1 and global_step > 3:
                    break
        # results, _, _ = self.eval_model(self.args.epochs-1, global_step)

        # global_step = global_step + 1
        # return 0
        if global_step == 0:
            global_step = global_step + 1
            logging.info("Bug round")
            for batch_idx, batch in enumerate(self.train_dl):
                logging.info(batch_idx)
                logging.info(batch)

        return global_step, tr_loss / global_step

    def eval_model(self, epoch=0, global_step=0, device=None):
        if not device:
            device = self.device

        results = {}

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(self.test_dl)
        test_sample_len = len(self.test_dl.dataset)
        preds = np.empty((test_sample_len, self.num_labels))

        out_label_ids = np.empty(test_sample_len)
        self.model.to(device)
        self.model.eval()
        # logging.info("saving model to ~/model_cache")
        # torch.save(self.model.state_dict(),"/home/cdq/model_cache/model_cache")
        logging.info("len(test_dl) = %d, n_batches = %d" % (len(self.test_dl), n_batches))
        for i, batch in enumerate(self.test_dl):
            with torch.no_grad():
                batch = tuple(t.to(device) for t in batch)
                # sample_index_list = batch[0].cpu().numpy()
                # if i == len(self.test_dl) - 1:
                #     logging.info(batch)
                x = batch[1]
                labels = batch[4]

                output = self.model(x)
                logits = output[0]
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()
                # logging.info("test. batch index = %d, loss = %s" % (i, str(eval_loss)))

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else test_sample_len
            # logging.info("batch index = %d, start_index = %d, end_index = %d" % (i, start_index, end_index))
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps

        model_outputs = preds
        preds = np.argmax(preds, axis=1)
        # logging.info("preds = " + str(preds))
        # logging.info("out_label_ids = " + str(out_label_ids))
        result, wrong = self.compute_metrics(preds, out_label_ids, self.test_dl.examples)
        result["eval_loss"] = eval_loss
        results.update(result)

        os.makedirs(self.args.output_dir, exist_ok=True)
        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        if result["acc"] > self.best_accuracy:
            self.best_accuracy = result["acc"]
        logging.info("best_accuracy = %f" % self.best_accuracy)
        wandb.log(result)

        wandb.log({"Evaluation Accuracy (best)": self.best_accuracy})
        wandb.log({"Evaluation Accuracy": result["acc"]})
        wandb.log({"Evaluation Loss": result["eval_loss"]})

        self.results.update(result)
        logging.info(self.results)

        return result, model_outputs, wrong

    def compute_metrics(self, preds, labels, eval_examples=None):
        assert len(preds) == len(labels)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )

    def build_optimizer(self, model, iteration_in_total,rounds):
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        self.args.warmup_steps = warmup_steps if self.args.warmup_steps == 0 else self.args.warmup_steps
        logging.info("warmup steps = %d" % self.args.warmup_steps)
        # freeze exps only apply for distilbert
        if self.args.model_type == "distilbert" or self.args.model_type == "bert":
            self.freeze_model_parameters(model,rounds)
        if self.args.fl_algorithm == "FedOPT" or self.args.fl_algorithm == "":
            optimizer = AdamW(model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        else:
            optimizer = SGD(model.parameters(), lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=iteration_in_total
        )
        return optimizer, scheduler
    
    def freeze_model_parameters(self, model,rounds):
        # origin
        modules = list()
        # self.freeze_layers = [] 
        logging.info("freeze layers: %s" % str(self.freeze_layers))
        for layer_idx in self.freeze_layers:
            if layer_idx == "e":
                modules.append(model.bert.embeddings)
            elif layer_idx == "h":
                modules.append(model.heads)
            else:
                modules.append(model.bert.encoder.layer[int(layer_idx)])

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        logging.info(get_parameter_number(model))


        # # adaptive adapters
        # # per device adaptive (random)
        # logging.info("before random")
        # random.seed(a=None, version=2) # without this line, it will produce the same s in one round.
        # s = random.randint(0,9) # 1% for strong, 99% for weak
        # logging.info(s)

        # modules = list()
        # layers_pool = ['e', '0', '1', '2', '3', '4', '5'] 
        # layers_pool_freeze_none = layers_pool[:-6]
        # layers_pool_freeze_0 = layers_pool[:-5]
        # layers_pool_freeze_0_1 = layers_pool[:-4]
        # layers_pool_freeze_0_2 = layers_pool[:-3]
        # layers_pool_freeze_0_3 = layers_pool[:-2]
        # layers_pool_freeze_0_4 = layers_pool[:-1]
        # layers_pool_freeze_0_5 = layers_pool[:]

        # layers_seperate_pool_0 = ['5'] 
        # layers_seperate_pool_1 = ['e', '0', '1', '2', '3', '4'] 

        # # activate all grads
        # self.freeze_layers = ['e', '0', '1', '2', '3', '4', '5'] 
        # logging.info("activate all grads")
        # for layer_idx in self.freeze_layers:
        #     if layer_idx == "e":
        #         modules.append(model.distilbert.embeddings)
        #     else:
        #         modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = True
        # model.train_adapter("rotten_tomatoes") # enable adapters


        # modules = list()
        
        # # Train continuous layers.
        # if s == 0: 
        #     self.freeze_layers = layers_pool_freeze_0_1
        # else:
        #     self.freeze_layers = layers_pool_freeze_0_4

        # # They only train their own layers instead of continuous layers.
        # # if s == 0: 
        # #     self.freeze_layers = layers_seperate_pool_0
        # # else:
        # #     self.freeze_layers = layers_seperate_pool_1

        # logging.info("freeze layers: %s" % str(self.freeze_layers))
        # for layer_idx in self.freeze_layers:
        #     if layer_idx == "e":
        #         modules.append(model.distilbert.embeddings)
        #     else:
        #         modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = False

        # logging.info(get_parameter_number(model))


        # per round adaptive
        # modules = list()
        # L1 = [0,1,2,3,4,5,6,7,8]
        # L2 = []
        # L3 = [9]
        
        # # activate all grads
        # self.freeze_layers = ['e', '0', '1', '2', '3', '4', '5'] 
        # logging.info("activate all grads")
        # for layer_idx in self.freeze_layers:
        #     if layer_idx == "e":
        #         modules.append(model.distilbert.embeddings)
        #     else:
        #         modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = True
        # model.train_adapter("rotten_tomatoes") # enable adapters

        # modules = list()

        # if rounds < 140: 
        #     self.freeze_layers = ['e', '0']   # ['e', '0', '1', '2', '3', '4', '5'] 
        #     logging.info("freeze layers: %s" % str(self.freeze_layers))
        #     for layer_idx in self.freeze_layers:
        #         if layer_idx == "e":
        #             modules.append(model.distilbert.embeddings)
        #         else:
        #             modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        #     for module in modules:
        #         for param in module.parameters():
        #             param.requires_grad = False
        # elif rounds < 180: 
        #     self.freeze_layers = ['e', '0', '1'] 
        #     logging.info("freeze layers: %s" % str(self.freeze_layers))
        #     for layer_idx in self.freeze_layers:
        #         if layer_idx == "e":
        #             modules.append(model.distilbert.embeddings)
        #         else:
        #             modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        #     for module in modules:
        #         for param in module.parameters():
        #             param.requires_grad = False
        # elif rounds < 220: 
        #     self.freeze_layers = ['e', '0', '1', '2']
        #     logging.info("freeze layers: %s" % str(self.freeze_layers))
        #     for layer_idx in self.freeze_layers:
        #         if layer_idx == "e":
        #             modules.append(model.distilbert.embeddings)
        #         else:
        #             modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        #     for module in modules:
        #         for param in module.parameters():
        #             param.requires_grad = False
        # elif rounds < 580: 
        #     self.freeze_layers = ['e', '0', '1', '2','3']
        #     logging.info("freeze layers: %s" % str(self.freeze_layers))
        #     for layer_idx in self.freeze_layers:
        #         if layer_idx == "e":
        #             modules.append(model.distilbert.embeddings)
        #         else:
        #             modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        #     for module in modules:
        #         for param in module.parameters():
        #             param.requires_grad = False
        # else: 
        #     self.freeze_layers = ['e', '0', '1', '2', '3','4']
        #     logging.info("freeze layers: %s" % str(self.freeze_layers))
        #     for layer_idx in self.freeze_layers:
        #         if layer_idx == "e":
        #             modules.append(model.distilbert.embeddings)
        #         else:
        #             modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        #     for module in modules:
        #         for param in module.parameters():
        #             param.requires_grad = False
        # elif rounds < 2000: 
        #     self.freeze_layers = ['e', '0', '1', '2', '3', '4']
        #     logging.info("freeze layers: %s" % str(self.freeze_layers))
        #     for layer_idx in self.freeze_layers:
        #         if layer_idx == "e":
        #             modules.append(model.distilbert.embeddings)
        #         else:
        #             modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        #     for module in modules:
        #         for param in module.parameters():
        #             param.requires_grad = False
        # else: 
        #     self.freeze_layers = ['e', '0', '1', '2', '3', '4', '5']
        #     logging.info("freeze layers: %s" % str(self.freeze_layers))
        #     for layer_idx in self.freeze_layers:
        #         if layer_idx == "e":
        #             modules.append(model.distilbert.embeddings)
        #         else:
        #             modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        #     for module in modules:
        #         for param in module.parameters():
        #             param.requires_grad = False
        
        # logging.info(get_parameter_number(model))

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

