import argparse
import os
import sys
import logging
import pickle
from functools import partial
import time
from tqdm import tqdm
from collections import Counter
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor
from datetime import datetime

from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup

from data_utils import *
from eval_utils import *

torch.set_float32_matmul_precision('high')

def init_args():
    parser = argparse.ArgumentParser(description="Configuration for model training and inference")
    parser.add_argument("--task", default='asqp', type=str, help="The name of the task which inference is conducted on, selected from: [asqp, acos, tasd]")
    parser.add_argument("--dataset", default='rest15', type=str, required=True, help="The name of the dataset which inference is conducted on, selected from: [rest15, rest16, laptop16]")     
    parser.add_argument("--train", default='train', type=str, help="Training data split")
    parser.add_argument("--dev", default='dev', type=str, help="Development data split")
    parser.add_argument("--eval_data_split", default='train', type=str, help="Data split used for evaluation")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str, help="Pre-trained model path or shortcut name")
    parser.add_argument("--load_ckpt_name", type=str, help="Pretrained model checkpoint file to load")
    parser.add_argument("--do_train", action='store_true', help="Whether to perform training")
    parser.add_argument("--do_inference", action='store_true', help="Whether to perform inference")
    parser.add_argument("--max_seq_length", default=200, type=int, help="Maximum input sequence length")
    parser.add_argument("--n_gpu", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Training batch size")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", default=6e-5, type=float, help="Learning rate")
    parser.add_argument("--num_train_epochs", default=50, type=int, help="Number of training epochs")
    parser.add_argument("--seed", default=25, type=int, help="Random seed")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay rate")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer")
    parser.add_argument("--warmup_steps", default=0.0, type=float, help="Number of warmup steps")
    parser.add_argument("--num_path", default=1, type=int, help="Number of paths for something specific")
    parser.add_argument("--beam_size", default=1, type=int, help="Beam size for beam search")
    parser.add_argument("--save_top_k", default=1, type=int, help="Number of top models to save")
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int, help="Validation checks per number of epochs")
    parser.add_argument("--load_path_cache", action='store_true', help="Whether to load path cache")
    parser.add_argument("--lowercase", default=True, type=bool, help="Whether to lowercase the input text")

    args = parser.parse_args()

    args.stage = 'sft'
    args.output_dir =  f'../outputs/sft/'
    args.data_path = f'../data/sft/{args.task}/{args.dataset}'
    
    if not os.path.exists(f'../outputs'):
        os.mkdir(f'../outputs')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return args

class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """

    def __init__(self, config, tfm_model, tokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=['tfm_model'])
        self.config = config
        self.model = tfm_model
        self.tokenizer = tokenizer

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(input_ids=batch["source_ids"],
                       attention_mask=batch["source_mask"],
                       labels=lm_labels,
                       decoder_attention_mask=batch['target_mask'])

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        # get f1
        outs = self.model.generate(input_ids=batch['source_ids'],
                                   attention_mask=batch['source_mask'],
                                   max_length=self.config.max_seq_length,
                                   return_dict_in_generate=True,
                                   output_scores=True,
                                   num_beams=1)

        dec = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in outs.sequences
        ]
        target = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]
        scores, _, _ = compute_scores(dec, target, verbose=False)
        f1 = torch.tensor(scores['f1'], dtype=torch.float32)

        # get loss
        loss = self._step(batch)

        if stage:
            self.log(f"{stage}_loss",
                     loss,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)
            self.log(f"{stage}_f1",
                     f1,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.config.learning_rate,
                          eps=self.config.adam_epsilon)
        scheduler = {
            "scheduler":
            get_linear_schedule_with_warmup(optimizer,
                                            **self.config.lr_scheduler_init),
            "interval":
            "step",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        print("load training data.")
        train_dataset = ABSADataset(tokenizer=self.tokenizer,
                            task_name=self.config.task,
                            data_type='train',
                            args=self.config,
                            max_len=self.config.max_seq_length)
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=2)

        return dataloader

    def val_dataloader(self):
        val_dataset = ABSADataset(tokenizer=self.tokenizer,
                            task_name=self.config.task,
                            data_type='dev',
                            args=self.config,
                            max_len=self.config.max_seq_length)
        return DataLoader(val_dataset,
                          batch_size=self.config.eval_batch_size,
                          num_workers=2)

    @staticmethod
    def rindex(_list, _value):
        return len(_list) - _list[::-1].index(_value) - 1

args = init_args()

print("\n", "=" * 30, f"SFT stage", "=" * 30, "\n")
tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

print(f"Here is an example (from the dev set):")
dataset = ABSADataset(tokenizer=tokenizer,
                    task_name=args.task,
                    data_type='dev',
                    args=args,
                    max_len=args.max_seq_length)

if args.do_train:
    print("\n****** Conduct Training ******")

    # initialize the T5 model
    tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model = T5FineTuner(args, tfm_model, tokenizer)
    
    train_loader = model.train_dataloader()
    
    # config optimizer
    t_total = ((len(train_loader.dataset) //
                (args.train_batch_size * max(1, args.n_gpu))) //
            args.gradient_accumulation_steps *
            float(args.num_train_epochs))
    
    args.lr_scheduler_init = {
        "num_warmup_steps": args.warmup_steps,
        "num_training_steps": t_total
    }
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename='{epoch}-{val_f1:.2f}-{val_loss:.2f}',
        monitor='val_f1',
        mode='max',
        save_top_k=args.save_top_k,
        save_last=False)
    
    early_stop_callback = EarlyStopping(monitor="val_f1",
                                        min_delta=0.00,
                                        patience=20,
                                        verbose=True,
                                        mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # prepare for trainer
    train_params = dict(
        accelerator="gpu",
        devices=[0],
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        max_epochs=args.num_train_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=[
            checkpoint_callback, early_stop_callback,
            TQDMProgressBar(refresh_rate=10), lr_monitor
        ],
    )
    
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    
    best_checkpoint_path = checkpoint_callback.best_model_path
    check_point = torch.load(best_checkpoint_path)
    model.load_state_dict(check_point["state_dict"])

    # save the final model
    model.model.save_pretrained(f"{args.output_dir}/best_model")
    tokenizer.save_pretrained(f"{args.output_dir}/best_model")
    
    print("Finish training and saving the model!")

# do inference
if args.do_inference:

    print("\n****** Conduct Evaluating with the last state ******")

    model = T5ForConditionalGeneration.from_pretrained(f"{args.output_dir}/best_model")
    tokenizer = T5Tokenizer.from_pretrained(f"{args.output_dir}/best_model")
    print()
    
    test_dataset = ABSADataset(tokenizer=tokenizer, task_name=args.task, data_type='test', args=args, max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=4)
    
    scores = evaluate(args, test_loader, model, tokenizer)
    print("Finish evaluating the model!")
    
