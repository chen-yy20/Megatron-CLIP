#!/bin/bash

set -x

export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)

export EXP_NAME='PureDP'
export MODEL_NAME='DP_CLIP'    

export GPUS_PER_NODE='8'
export NNODES='2'
export NODELIST='nico[1,2]'

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

# only allow to use DP for CLIP baseline
export GLOBAL_BATCH_SIZE=1024
export MICRO_BATCHES=64
export DATA_PARALLEL_SIZE=$(($GPUS_PER_NODE*$NNODES))
export MICRO_BATCH_SIZE=$(expr $GLOBAL_BATCH_SIZE / $(($DATA_PARALLEL_SIZE*$MICRO_BATCHES)))

LOG_DIR=${EXP_NAME}\_d$DATA_PARALLEL_SIZE\_gbs$GLOBAL_BATCH_SIZE\_mbs$MICRO_BATCH_SIZE
LOG_NAME=${MODEL_NAME}\_d$DATA_PARALLEL_SIZE\_gbs$GLOBAL_BATCH_SIZE\_mbs$MICRO_BATCH_SIZE\_$(date -Iseconds).log

mkdir -p ./logs
mkdir -p ./logs/${LOG_DIR}

# export PROFILER_LOG_PATH=${LOG_DIR}/${LOG_PREFIX}.prof

# mkdir -p $PROFILER_LOG_PATH

NNODES=$(scontrol show hostnames ${NODELIST} | wc -l)

srun \
    --exclusive=user \
    -p Big \
    -K \
	-N $NNODES \
    -w $NODELIST \
    --time 20:00 \
    --job-name=megaCLIP \
	--ntasks-per-node=$GPUS_PER_NODE \
    --gres=gpu:$GPUS_PER_NODE \
    --export=ALL \
        bash ./zPretrain/pretrain_clip_bl.sh > ./logs/${LOG_DIR}/${LOG_NAME} 2>&1
