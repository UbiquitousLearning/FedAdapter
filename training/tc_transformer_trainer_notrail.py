#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function
from ast import Raise

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


        # batch_chosen = random.sample(range(0, 29),8)
        # logging.info("Batch chosen is %s", str(batch_chosen))

        for epoch in range(0, self.args.epochs):
            for batch_idx, batch in enumerate(self.train_dl):
                # logging.info("batch_idx %s", str(batch_idx))                
                # if batch_idx not in batch_chosen:
                #     continue

                self.model.train()

                batch = tuple(t for t in batch)

                x = batch[1].to(device)

                labels = batch[4].to(device)

                output = self.model(x)
                
                # Stack
                logits = output[0] 
                # Parallel
                # logits = (output[0][0] + output[1][0] + output[2][0])/3

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

                loss.backward()
                tr_loss += loss.item()
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
                # logits = (output[0][0] + output[1][0] + output[2][0])/3
                
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

            # Remove Adapters
            # adapter_idx = dynamic_adapter_idx(rounds)
            # if adapter_idx != 12:
            #     dynamic_add_adapter(model,adapter_idx)

            # Freeze Adapters
            # self.freeze_layers = dynamic_freeze_single(self.freeze_layers, rounds)
            # self.freeze_model_parameters(model,rounds)
            logging.info(get_parameter_number(model))

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
        for param in model.parameters(): # activate all gradients
            param.requires_grad = True
        model.train_adapter("test")

        modules = list()
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

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def dynamic_add_adapter(model, adapter_idx):
    leave_out = [0,1,2,3,4,5,6,7,8,9,10,11]
    leave_out.remove(adapter_idx)
    adapter_config = {'original_ln_before':True, 'original_ln_after':True, 'residual_before_ln':True, 'adapter_residual_before_ln':False, 'ln_before':False, 'ln_after':False, 'mh_adapter':False, 'output_adapter':True, 'non_linearity':'relu', 'reduction_factor':16, 'inv_adapter':None, 'inv_adapter_reduction_factor':None, 'cross_adapter':False, 'leave_out':leave_out} # [0,1,2,3,4,5,6,7,8,9,10,11]
    model.add_adapter(str(adapter_idx), config=adapter_config)
    model.train_adapter(str(adapter_idx))
    logging.info(model)
    logging.info(get_parameter_number(model))
    return model

def dynamic_freeze(freeze_layers,rounds):
    if rounds < 30:
        freeze_layers = [0,1,2,3,4,5,6,7,8,9,10]
    elif rounds < 60:
        freeze_layers = [0,1,2,3,4,5,6,7,8,9]
    elif rounds < 90:
        freeze_layers = [0,1,2,3,4,5,6,7,8]
    elif rounds < 120:
        freeze_layers = [0,1,2,3,4,5,6,7]
    elif rounds < 150:
        freeze_layers = [0,1,2,3,4,5,6]
    elif rounds < 180:
        freeze_layers = [0,1,2,3,4,5]
    elif rounds < 210:
        freeze_layers = [0,1,2,3,4]
    elif rounds < 240:
        freeze_layers = [0,1,2,3]
    elif rounds < 270:
        freeze_layers = [0,1,2]
    elif rounds < 300:
        freeze_layers = [0,1]
    elif rounds < 330:
        freeze_layers = [0]
    else:
        freeze_layers = []
    return freeze_layers

def dynamic_freeze_single(freeze_layers,rounds):
    if rounds < 30:
        freeze_layers = [0,1,2,3,4,5,6,7,8,9,10]
    elif rounds < 60:
        freeze_layers = [0,1,2,3,4,5,6,7,8,9,11]
    elif rounds < 90:
        freeze_layers = [0,1,2,3,4,5,6,7,8,10,11]
    elif rounds < 120:
        freeze_layers = [0,1,2,3,4,5,6,7,9,10,11]
    elif rounds < 150:
        freeze_layers = [0,1,2,3,4,5,6,8,9,10,11]
    elif rounds < 180:
        freeze_layers = [0,1,2,3,4,5,7,8,9,10,11]
    elif rounds < 210:
        freeze_layers = [0,1,2,3,4,6,7,8,9,10,11]
    elif rounds < 240:
        freeze_layers = [0,1,2,3,5,6,7,8,9,10,11]
    elif rounds < 270:
        freeze_layers = [0,1,2,4,5,6,7,8,9,10,11]
    elif rounds < 300:
        freeze_layers = [0,1,3,4,5,6,7,8,9,10,11]
    elif rounds < 330:
        freeze_layers = [0,2,3,4,5,6,7,8,9,10,11]
    else:
        freeze_layers = [1,2,3,4,5,6,7,8,9,10,11]
    return freeze_layers

def dynamic_adapter_idx(rounds):
    idx = 11
    if rounds == 30:
        idx = 11
    elif rounds == 60:
        idx = 10
    elif rounds == 90:
        idx = 9
    elif rounds == 120:
        idx = 8
    elif rounds == 150:
        idx = 7
    elif rounds == 180:
        idx = 6
    elif rounds == 210:
        idx = 5
    elif rounds == 240:
        idx = 4
    elif rounds == 270:
        idx = 3
    elif rounds == 300:
        idx = 2
    elif rounds == 330:
        idx = 1
    elif rounds == 360:
        idx = 0
    
    return idx