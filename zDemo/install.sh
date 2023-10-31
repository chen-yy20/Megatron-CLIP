#!/bin/bash

a=$(echo $HOSTNAME | cut -c12-16)

JOB_NAME='GPT_MEGATRON'
GPUS_PER_NODE=1
partition='V100'
NNODE=1
NODELIST='nico4'

srun --exclusive=user \
    --partition=${partition} \
    -N $NNODE \
    -w $NODELIST \
    -K \
    --time 30:00 \
    --job-name=${JOB_NAME} \
    --gres=gpu:v132p:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --export=ALL \
    bash zDemo/build.sh
