#!/bin/bash

set -x

export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)

# export EXP_NAME=$1
# export MODEL_NAME=$2
# export TENSOR_PARALLEL_SIZE=$3
# export DATA_PARALLEL_SIZE=$4
# export PIPELINE_PARALLEL_SIZE=$5
# export GLOBAL_BATCH_SIZE=$6
# export MICRO_BATCH_SIZE=$7
# export NODELIST=$8
# export GPUS_PER_NODE=$9
export EXP_NAME='MEGATRON-CLIP'
export MODEL_NAME='CLIP'    

# export TENSOR_PARALLEL_SIZE='2'
# export PIPELINE_PARALLEL_SIZE='1'
# export DATA_PARALLEL_SIZE='2'
export GPUS_PER_NODE='8'
# export NNODES=$(( $DATA_PARALLEL_SIZE * $TENSOR_PARALLEL_SIZE * $PIPELINE_PARALLEL_SIZE / $GPUS_PER_NODE))
export NNODES='2'
export NODELIST='nico[1-2]'


export TENSOR_MODEL_PARALLEL=$1
export PIPELINE_MODEL_PARALLEL=$2
export DATA_PARALLEL=$3
export MICRO_BATCHES=$4
export VISION_L=$5
export TEXT_L=$6
export LOG_DIR=$7
export LOG_NAME=$8

NNODES=$(scontrol show hostnames ${NODELIST} | wc -l)
# nohup
srun \
    --exclusive=user \
    -p Big \
    -K \
	-N $NNODES \
    -w $NODELIST \
    --time 20:00 \
    --job-name=MegaCLIP \
	--ntasks-per-node=$GPUS_PER_NODE \
    --gres=gpu:$GPUS_PER_NODE \
    --export=ALL \
    bash zPretrain/pretrain_clip_bl_3d.sh \
    # > $LOG_DIR/$LOG_NAME 2>&1
