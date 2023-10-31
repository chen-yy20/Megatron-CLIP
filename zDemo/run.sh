#!/bin/bash

set -x

# if [ "$#" -ne 8 ]
# then
#     echo "usage:" $0 "exp_name model_name t p d gbs mbs nodelist"
#     exit 1
# fi

export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)

# export EXP_NAME=$1
export EXP_NAME='GPT_MEGATRON'
export MODEL_NAME='GPT-760M'    
export TENSOR_PARALLEL_SIZE='1'
export PIPELINE_PARALLEL_SIZE='2'
export DATA_PARALLEL_SIZE='4'
export GLOBAL_BATCH_SIZE='64'
export MICRO_BATCH_SIZE='32'
export NODELIST='nico3'

export NUM_LAYERS=-1
export HIDDEN_SIZE=-1
export NUM_ATTN_HEADS=-1

if [ ${MODEL_NAME} == "GPT-760M" ];then
    export NUM_LAYERS=24
    export HIDDEN_SIZE=1536
    export NUM_ATTN_HEADS=16
fi

if [ ${MODEL_NAME} == "GPT-1.3B" ];then
    export NUM_LAYERS=24
    export HIDDEN_SIZE=2048
    export NUM_ATTN_HEADS=16
fi

if [ ${NUM_LAYERS} == -1 ];then
    echo "model name not found."
    exit -1
fi

mkdir -p ./logs
mkdir -p ./logs/${EXP_NAME}

LOG_DIR=$(pwd)/logs/${EXP_NAME}

LOG_PREFIX=${MODEL_NAME}\_t$TENSOR_PARALLEL_SIZE\_p$PIPELINE_PARALLEL_SIZE\_d$DATA_PARALLEL_SIZE\_gbs$GLOBAL_BATCH_SIZE\_mbs$MICRO_BATCH_SIZE\_$(date -Iseconds)
LOG_NAME=${LOG_PREFIX}.log

export PROFILER_LOG_PATH=${LOG_DIR}/${LOG_PREFIX}.prof

mkdir -p $PROFILER_LOG_PATH

NNODES=$(scontrol show hostnames ${NODELIST} | wc -l)

srun \
    --exclusive \
    -p V100\
    -K \
    --time 30:00 \
	-N $NNODES \
    -w $NODELIST \
	--export=ALL \
	--ntasks-per-node=4 \
	--gres=gpu:4 \
	zDemo/pretrain.sh \
	| tee ${LOG_DIR}/${LOG_NAME}
