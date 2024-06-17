# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement

import re
import os
import sys
import logging
import pickle
import numpy as np
from tqdm import tqdm
import torch
from data_utils import *
from eval_utils import *


sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    n_gold = len(gold_pt)
    n_pred = len(pred_pt)

    for i in range(len(pred_pt)):
        if pred_pt[i] == gold_pt[i]:
            n_tp += 1

    #print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

    return scores


def compute_scores(pred_seqs, gold_seqs, verbose=True):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = gold_seqs[i]
        pred_list = pred_seqs[i]
        # if verbose and i < 10:

        #     print("gold ", gold_seqs[i])
        #     print("pred ", pred_seqs[i])
        #     print()

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    scores = compute_f1_scores(all_preds, all_labels)

    return scores, all_labels, all_preds

def add_space_before_punctuation(text):
    text = re.sub(r'([.!?])', r' \1', text) 
    text = re.sub(r"(\w)'s", r"\1 ' s", text) 
    return text
    
def evaluate(args, data_loader, model, tokenizer):
    
    if args.stage == 'sft':
        result_path_dir = os.path.join(args.output_dir, args.task, args.dataset)
        if not os.path.exists(result_path_dir):
            os.makedirs(result_path_dir)
        result_path = os.path.join(result_path_dir, "sft_test.txt")        
    else:
        result_path = os.path.join(args.output_dir, "dpo_test.txt")
    
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()

    outputs, targets = [], []
    
    for batch in tqdm(data_loader):
        outs = model.generate(input_ids=batch['source_ids'].to(device), 
                                    attention_mask=batch['source_mask'].to(device), 
                                    max_length=args.max_seq_length, num_beams=1, early_stopping=True)
        
        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
        
        outputs.extend(dec)
        targets.extend(target)

    output_sents = [add_space_before_punctuation(sent) for sent in outputs]

    # load original quad
    with open(f'../data/original/{args.task}/{args.dataset}/test.txt', 'r', encoding='UTF-8') as f:
        lines = f.readlines()

    quads = [str(eval(lines[i].split('####')[1].lower())) for i in range(len(lines))]

    with open(result_path, 'w') as f:
        for sent, quad in zip(output_sents, quads):
            f.write(f'{sent}####{quad}\n')
    
    _, all_labels, all_preds = compute_scores(output_sents, targets)

    return _