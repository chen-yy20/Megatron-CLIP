#!/bin/bash
set -x
# if [ "$#" -ne 8 ]
# then
#     echo "usage:" $0 "exp_name model_name t p d gbs mbs nodelist"
#     exit 1
# fi

export NNODE=1
export NODELIST="nico4"
export NTASK_PER_NODE=1
export PYTHONENV="/home/zanzong/workspace/deepspeed-env/bin/activate"
export TIMESTAMP=$(date "+%Y-%m-%d-%H-%M-%S")

srun --exclusive=user \
    -p V100 \
    -N $NNODE \
    -w $NODELIST \
    -K \
    --time 30:00 \
    --ntasks-per-node=$NTASK_PER_NODE \
    --gres=gpu:$NTASK_PER_NODE \
    --export=ALL \
    bash train.sh