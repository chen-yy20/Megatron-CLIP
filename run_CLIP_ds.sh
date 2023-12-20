#!/bin/bash

set -x

export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)

export EXP_NAME='MEGA-CLIP-DS'
export MODEL_NAME='CLIP'    

# export TENSOR_PARALLEL_SIZE='2'
# export PIPELINE_PARALLEL_SIZE='1'
# export DATA_PARALLEL_SIZE='2'
export GPUS_PER_NODE='2'
export NNODES='1'
export NODELIST='nico[2]'

# export GLOBAL_BATCH_SIZE='64'
# export MICRO_BATCH_SIZE='4'

mkdir -p ./logs
mkdir -p ./logs/${EXP_NAME}

LOG_DIR=$(pwd)/logs/${EXP_NAME}

LOG_PREFIX=${MODEL_NAME}\_t$TENSOR_PARALLEL_SIZE\_p$PIPELINE_PARALLEL_SIZE\_d$DATA_PARALLEL_SIZE\_gbs$GLOBAL_BATCH_SIZE\_mbs$MICRO_BATCH_SIZE\_$(date -Iseconds)
LOG_NAME=${LOG_PREFIX}.log

export PROFILER_LOG_PATH=${LOG_DIR}/${LOG_PREFIX}.prof

mkdir -p $PROFILER_LOG_PATH

NNODES=$(scontrol show hostnames ${NODELIST} | wc -l)

srun \
    --exclusive=user \
    -p Big \
    -K \
	-N $NNODES \
    -w $NODELIST \
    --time 20:00 \
    --job-name=ds-CLIP \
	--ntasks-per-node=$GPUS_PER_NODE \
    --gres=gpu:$GPUS_PER_NODE \
    --export=ALL \
        bash ./zPretrain/pretrain_clip_ds.sh
