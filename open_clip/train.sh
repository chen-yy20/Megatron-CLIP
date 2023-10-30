#!/bin/bash
# 开启调试模式，将要执行的命令输出出来 bash -x train.sh
set -x

eval "$(source $PYTHONENV)"
export MASTER_PORT=12802
# 这行代码使用 scontrol 命令来获取当前作业的信息，然后使用 tr 命令将等号替换为空格，使用 grep 命令过滤出 BatchHost，最后使用 awk 命令获取 BatchHost 的值，并将其赋值给 MASTER_ADDR 环境变量，用于指定主节点的地址。
export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
# 这两行代码分别计算当前进程所在节点的排名和本地排名，并将它们赋值给 NODE_RANK 和 LOCAL_RANK 环境变量。
export NODE_RANK=$(expr $SLURM_PROCID / $NNODE)
export LOCAL_RANK=$NODE_RANK
# 这两行代码分别计算集群中的进程总数和当前进程的排名，并将它们赋值给 WORLD_SIZE 和 RANK 环境变量。
export WORLD_SIZE=$(($NTASK_PER_NODE*$NNODE))
export RANK=$SLURM_PROCID

cd /home/chen-yy20/flexpipe/models/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

exec python -u src/training/main.py \
    --deepspeed_config=ds_configs/ds_config.json \
    --save-frequency 1 \
    --train-data="/mnt/zoltan/zanzong/CC3M/cc3m/{00000..00331}.tar" \
    --train-num-samples 3000000 \
    --batch-size 16 \
    --epochs=1 \
    --model ViT-H-16 \
    --name "ViT-H-16-"$TIMESTAMP \
    --seed 0 \
    --force-patch-dropout 0. \
    --gather-with-grad
