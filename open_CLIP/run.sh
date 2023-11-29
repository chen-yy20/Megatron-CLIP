#!/bin/bash
set -x
# if [ "$#" -ne 8 ]
# then
#     echo "usage:" $0 "exp_name model_name t p d gbs mbs nodelist"
#     exit 1
# fi

export NNODE=1
export NODELIST="nico3"
export NTASK_PER_NODE=4
export PYTHONENV="/home/chen-yy20/mega_env/bin/activate"
export TIMESTAMP=$(date "+%Y-%m-%d-%H-%M-%S")

srun --exclusive=user \
    -p V100 \
    -N $NNODE \
    -w $NODELIST \
    -K \
    --time 30:00 \
    --job-name="clip" \
    --ntasks-per-node=$NTASK_PER_NODE \
    --gres=gpu:v132p:$NTASK_PER_NODE \
    --export=ALL \
    bash train.sh
