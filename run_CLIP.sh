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

# export GLOBAL_BATCH_SIZE='64'
# export MICRO_BATCH_SIZE='4'

export EXTRA_WORLD_SIZE=$1
export TENSOR_MODEL_PARALLEL=$2
export PIPELINE_MODEL_PARALLEL=$3
export XTENSOR_MODEL_PARALLEL=$4
export XPIPELINE_MODEL_PARALLEL=$5
export MICRO_BATCHES=$6
export VISION_L=$7
export TEXT_L=$8
export LOG_DIR=$9
export LOG_NAME=${10}
echo $LOG_DIR
echo $LOG_NAME

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
