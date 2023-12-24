#!/bin/bash

set -x

export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)

export EXP_NAME='MEGATRON-CLIP'
export MODEL_NAME='CLIP'    

export GPUS_PER_NODE='8'
# export NNODES=$(( $DATA_PARALLEL_SIZE * $TENSOR_PARALLEL_SIZE * $PIPELINE_PARALLEL_SIZE / $GPUS_PER_NODE))
export NNODES='4'
export NODELIST='nico[1-4]'

# export GLOBAL_BATCH_SIZE='64'
# export MICRO_BATCH_SIZE='4'

# export EXTRA_WORLD_SIZE=$1
# export TENSOR_MODEL_PARALLEL=$2
# export PIPELINE_MODEL_PARALLEL=$3
# export XTENSOR_MODEL_PARALLEL=$4
# export XPIPELINE_MODEL_PARALLEL=$5
# export MICRO_BATCHES=$6
# export VISION_L=$7
# export TEXT_L=$8
# export LOG_DIR=$9
# export LOG_NAME=${10}
# echo $LOG_DIR
# echo $LOG_NAME

NNODES=$(scontrol show hostnames ${NODELIST} | wc -l)
#nohup 
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
    bash ./zPretrain/pretrain_clip_e.sh \
    # > $LOG_DIR/$LOG_NAME 2>&1
