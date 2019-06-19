#!/bin/bash

#### local path
SQUAD_DIR=data/squad
INIT_CKPT_DIR=xlnet_cased_L-24_H-1024_A-16

#### google storage path
GS_ROOT=
GS_INIT_CKPT_DIR=${GS_ROOT}/${INIT_CKPT_DIR}
GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/squad
GS_MODEL_DIR=${GS_ROOT}/experiment/squad

# TPU name in google cloud
TPU_NAME=

python run_squad.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --model_config_path=${INIT_CKPT_DIR}/xlnet_config.json \
  --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
  --output_dir=${GS_PROC_DATA_DIR} \
  --init_checkpoint=${GS_INIT_CKPT_DIR}/xlnet_model.ckpt \
  --model_dir=${GS_MODEL_DIR} \
  --train_file=${SQUAD_DIR}/train-v2.0.json \
  --predict_file=${SQUAD_DIR}/dev-v2.0.json \
  --uncased=False \
  --max_seq_length=512 \
  --do_train=True \
  --train_batch_size=48 \
  --do_predict=True \
  --predict_batch_size=32 \
  --learning_rate=3e-5 \
  --adam_epsilon=1e-6 \
  --iterations=1000 \
  --save_steps=1000 \
  --train_steps=8000 \
  --warmup_steps=1000 \
  $@
