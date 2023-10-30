#!/bin/bash
set -x
# if [ "$#" -ne 8 ]
# then
#     echo "usage:" $0 "exp_name model_name t p d gbs mbs nodelist"
#     exit 1
# fi

export NNODE=1
export NODELIST="nico3"
export NTASK_PER_NODE=2
export PYTHONENV="/home/chen-yy20/venv/bin/activate"
export TIMESTAMP=$(date "+%Y-%m-%d-%H-%M-%S")

# --gres=gpu:[v116p,v132p]:$NTASK_PER_NODE # for allocate specific 32GB/16GB cards
# --gres=gpu:$NTASK_PER_NODE # no specific
# --exclusive=user 选项确保资源是独占的。
# -p V100 选项指定了要使用的分区。
# --time 30:00 选项设置了作业的最大运行时间。
# --job-name="clip" 选项设置了作业的名称。
# --ntasks-per-node=$NTASK_PER_NODE 选项指定了每个节点的任务数。
# --gres=gpu:v132p:$NTASK_PER_NODE 选项指定了要使用的 GPU 类型和数量, v132p 指的是只使用32G内存的V100
# -p partition 分区 V100 long big V132等等

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
