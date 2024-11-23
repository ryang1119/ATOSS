from typing import Dict, Optional

import torch, json, os, argparse
from peft import LoraConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader
import pandas as pd
from trl import DPOTrainer
from data_utils import *
from eval_utils import *
#from process import *

import random
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict
import json

def load_json_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_dict(data)

def init_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", default='asqp', type=str, help="The name of the task, selected from: [asqp, acos]")
    parser.add_argument("--dataset", default='rest15', type=str, required=True, help="The name of the dataset, selected from: [rest15, rest16, laptop16]")   
    parser.add_argument("--model_name_or_path", default='t5-base', type=str, help="Path to pre-trained model or shortcut name")

    parser.add_argument("--beta", type=float, default=0.1, help="the beta parameter for DPO loss")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="optimizer learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="the number of gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=512, help="max length of each sample")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="max length of each sample's prompt")
    parser.add_argument("--max_target_length", type=int, default=512, help="Only used for encoder decoder model. Max target of each sample's prompt")
    parser.add_argument("--num_train_epochs",type=int, default=1, help="the number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=16, help="batch size per device for training/evaluation")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="batch size per device for training/evaluation")
    parser.add_argument("--max_seq_length", type=int, default=150, help="batch size per device for training/evaluation")

    parser.add_argument("--seed", type=int, default=25, help="random seed")
    parser.add_argument("--n_gpu", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--epoch_result", type=int, default=1, help="the number of epochs to save the model")
    parser.add_argument("--output_dir", type=str, default='', help="the output directory")
    parser.add_argument("--train_data", type=str, default='train', help="the output directory")

    parser.add_argument("--report_to", type=str, default='wandb', help="The list of integrations to report the results and logs to.")
    
    parser.add_argument("--ignore_bias_buffers", type=bool, default=False, help="fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See")
    parser.add_argument("--gradient_checkpointing", type=bool, default=False, help="Whether to use gradient checkpointing or no")
    parser.add_argument("--base_model", default='mvp', type=str,
                        help="The name of the base model, selected from: [para, mvp]")
    parser.add_argument("--domain", default='rest', type=str, help="The name of the domain, selected from: [rest, laptop]")
    parser.add_argument("--lowercase", default=True, type=bool, help="Whether to lowercase the input text")

    # wandb parameters
    args = parser.parse_args()
    
    args.stage = 'dpo'
    args.data_path = f'../data/original/{args.task}/{args.dataset}/'
    args.saved_model_path = f"../outputs/sft/best_model"

    output_dir = f'../outputs/dpo/{args.base_model}/{args.task}/{args.dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.output_dir = output_dir
    
    return args

args = init_args()

print("\n", "=" * 30, f"DPO stage: Task : {args.task} Dataset: {args.dataset}", "=" * 30, "\n")
print('\n', "="*30, f"Base Model: {args.base_model} Domain: {args.domain}", "="*30, '\n')

# 1. load a pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(args.saved_model_path)

if args.ignore_bias_buffers:
    # torch distributed hack
    model._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    ]

model_ref = AutoModelForSeq2SeqLM.from_pretrained(args.saved_model_path)
tokenizer = AutoTokenizer.from_pretrained(args.saved_model_path, lecacy=False, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load train dataset
train_data = load_json_dataset(f'../data/dpo/{args.base_model}/{args.domain}.json')

# initialize training arguments:
training_args = TrainingArguments(
    per_device_train_batch_size=args.train_batch_size,
    remove_unused_columns=False,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    logging_strategy="epoch",
    logging_first_step=True,
    num_train_epochs=args.num_train_epochs,
    output_dir=f'{args.output_dir}',
    optim="adamw_hf",
    adam_epsilon=1e-8,
    warmup_steps=150,
    bf16=True,
    gradient_checkpointing=args.gradient_checkpointing,
    save_total_limit=1,
    seed=args.seed,
)

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=args.beta,
    train_dataset=train_data,
    tokenizer=tokenizer,
    max_length=args.max_length,
    max_target_length=args.max_target_length,
    max_prompt_length=args.max_prompt_length,
)

dpo_trainer.train()
model_save_path = args.output_dir
dpo_trainer.save_model(model_save_path)

print("\n****** Conduct Evaluating with the last state ******")

test_dataset = ABSADataset(tokenizer=tokenizer, task_name=args.task, data_type='test', args=args, max_len=args.max_seq_length)
test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=4)
    
# compute the performance scores
scores = evaluate(args, test_loader, model, tokenizer)
print("Finish evaluating the model!")
