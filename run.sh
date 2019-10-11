#!/bin/bash
PYTHONPATH=`pwd`
echo $PYTHONPATH
NAME=ptb
# Data
DATA_ROOT=./data/${NAME}/

# Model
N_LAYER=6
D_MODEL=200
D_EMBED=200
N_HEAD=4
D_HEAD=32
D_INNER=512

# Training
TGT_LEN=128
MEM_LEN=128

BSZ=24

# Testing
TEST_TGT_LEN=80
TEST_MEM_LEN=2100
TEST_CLAMP_LEN=820

TEST_BSZ=8

if [[ $1 == 'train_data' ]]; then
    python tfxl/data_utils.py \
        --data_dir=${DATA_ROOT}/ \
        --dataset=${NAME} \
        --tgt_len=${TGT_LEN} \
        --per_host_train_bsz=${BSZ} \
        --per_host_valid_bsz=${BSZ} \
        ${@:2}
elif [[ $1 == 'test_data' ]]; then
    python tfxl/data_utils.py \
        --data_dir=${DATA_ROOT}/ \
        --dataset=${NAME} \
        --tgt_len=${TEST_TGT_LEN} \
        --per_host_test_bsz=${TEST_BSZ} \
        ${@:2}
elif [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python tfxl/train_gpu.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --model_dir=EXP-${NAME} \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.1 \
        --dropatt=0.0 \
        --learning_rate=0.00025 \
        --warmup_steps=0 \
        --train_steps=400000 \
        --tgt_len=${TGT_LEN} \
        --mem_len=${MEM_LEN} \
        --train_batch_size=${BSZ} \
        --iterations=200 \
        --save_steps=4000 \
        --do_train=True \
        --do_eval=False \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python tfxl/train_gpu.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --model_dir=EXP-${NAME} \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --eval_batch_size=${TEST_BSZ} \
        --do_train=False \
        --do_eval=True \
        --eval_split=test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
