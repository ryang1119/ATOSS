#!/bin/sh

TASK="asqp"
DATASET="rest15"

python SFT.py --task $TASK \
            --dataset $DATASET \
            --model_name_or_path t5-base \
            --n_gpu 1 \
            --do_train \
            --do_inference \
            --train_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 6e-5 \
            --num_train_epochs 1 \
            --max_seq_length 180 \
            --check_val_every_n_epoch 1 \
            --save_top_k 1