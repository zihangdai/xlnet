#!/bin/bash

#### local path
INIT_CKPT_DIR=models/xlnet_cased_L-24_H-1024_A-16
OUTPUT_DIR=data
MODEL_DIR=experiment/extract_features

#### Use 1 GPU, with 8 seqlen-64 samples

python extract_features.py \
    --input_file=data/corpus.txt \
    --init_checkpoint=${INIT_CKPT_DIR}/xlnet_model.ckpt \
    --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
    --use_tpu=False \
    --num_core_per_host=1 \
    --output_file=${OUTPUT_DIR}/output.json \
    --model_dir=${MODEL_DIR} \
    --num_hosts=1 \
    --max_seq_length=64 \
    --eval_batch_size=8 \
    --predict_batch_size=8 \
    --model_config_path=${INIT_CKPT_DIR}/xlnet_config.json \
    --summary_type=mean \
    $@
