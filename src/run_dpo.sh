#!/bin/sh

TASK="acos"
DATASET="rest16"
DOMAIN="rest"
BASE_MODEL="mvp"

python DPO.py --task $TASK \
            --dataset $DATASET \
            --domain $DOMAIN \
            --base_model $BASE_MODEL \
            --model_name_or_path t5-base \
            --n_gpu 1 \
            --train_batch_size 8 \
            --eval_batch_size 8 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --num_train_epochs 1 \
            --max_seq_length 200 