#!/bin/bash

set -x
if [ "$#" -ne 12 ]
then
    echo "usage:" $0 "exp_name model_name t p d gbs mbs nodelist total recompute_{granularity method num_layers}"
    exit 1
fi

export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)

export EXP_NAME=$1
export MODEL_NAME=$2
export TENSOR_PARALLEL_SIZE=$3
export DATA_PARALLEL_SIZE=$4
export PIPELINE_PARALLEL_SIZE=$5
export GLOBAL_BATCH_SIZE=$6
export MICRO_BATCH_SIZE=$7
export NODELIST=$8
export GPUS_PER_NODE=$9
export RECOMPUTE_GRANULARITY=${10}
export RECOMPUTE_METHOD=${11}
export RECOMPUTE_NUM_LAYERS=${12}

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

if [ ${RECOMPUTE_GRANULARITY} == "selective" ];then
    export EXPERIMENT_METHOD="selective"
fi

if [ ${RECOMPUTE_GRANULARITY} == "none" ];then
    export EXPERIMENT_METHOD="none"
fi

if [ ${RECOMPUTE_GRANULARITY} == "full" ];then
    export EXPERIMENT_METHOD="${RECOMPUTE_GRANULARITY}_${RECOMPUTE_METHOD}_${RECOMPUTE_NUM_LAYERS}"
fi

mkdir -p ./logs
mkdir -p ./logs/${EXP_NAME}

LOG_DIR=$(pwd)/logs/${EXP_NAME}

LOG_PREFIX=${MODEL_NAME}\_t$TENSOR_PARALLEL_SIZE\_d$DATA_PARALLEL_SIZE\_p$PIPELINE_PARALLEL_SIZE\_gbs$GLOBAL_BATCH_SIZE\_mbs$MICRO_BATCH_SIZE\_WAY$EXPERIMENT_METHOD
LOG_NAME=${LOG_PREFIX}.log

NNODES=$(scontrol show hostnames ${NODELIST} | wc -l)

if [ ${RECOMPUTE_GRANULARITY} != "full" ];then
    nohup srun \
    --exclusive=user \
    -p Big \
    -N $NNODES \
    -K \
    -w $NODELIST \
    --time 20:00 \
    --job-name=mmpret \
	--ntasks-per-node=$GPUS_PER_NODE \
    --gres=gpu:$GPUS_PER_NODE \
    --export=ALL \
	bash pretrain.sh  > ${LOG_DIR}/${LOG_NAME} 2>&1
fi

if [ ${RECOMPUTE_GRANULARITY} == "full" ];then
    nohup srun \
    --exclusive=user \
    -p Big \
    -N $NNODES \
    -K \
    -w $NODELIST \
    --time 20:00 \
    --job-name=mmpret \
	--ntasks-per-node=$GPUS_PER_NODE \
    --gres=gpu:$GPUS_PER_NODE \
    --export=ALL \
	bash pretrain_full.sh  > ${LOG_DIR}/${LOG_NAME} 2>&1
fi
