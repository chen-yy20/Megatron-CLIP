#!/bin/bash
set -x
# if [ "$#" -ne 8 ]
# then
#     echo "usage:" $0 "exp_name model_name t p d gbs mbs nodelist"
#     exit 1
# fi

export NUM_WORKER=1
export NODELIST="nico4"
export NUM_PROC=4

srun --exclusive \
    -p V100 \
    -N $NUM_WORKER \
    -w $NODELIST \
    -K \
    --time 30:00 \
    --ntasks-per-node=$NUM_PROC \
    --gres=gpu:$NUM_PROC \
    --export=ALL \
    bash train.sh
