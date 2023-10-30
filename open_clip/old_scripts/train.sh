#!/bin/bash
set -x

# eval "$(/path/to/conda/bin/conda shell.bash hook)" # init conda
# conda activate open_clip
# export CUDA_VISIBLE_DEVICES=0,1,2,3 #,4,5,6,7,8
export MASTER_PORT=12802
export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NUM_WORKER)
export WORLD_SIZE=$(($NUM_PROC*$NUM_WORKER))
export RANK=$SLURM_PROCID

cd /home/zanzong/workspace/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

exec python -u src/training/main.py \
    --save-frequency 1 \
    --report-to tensorboard \
    --train-data="/mnt/zoltan/zanzong/CC3M/cc3m/{00000..00331}.tar" \
    --train-num-samples 3000000 \
    --warmup 2000 \
    --batch-size=16 \
    --epochs=32 \
    --model ViT-L-16 \
    --name "ViT-L-16-Vanilla" \
    --seed 0 \
    --local-loss \
    --gather-with-grad \
    --force-patch-dropout 0.


# --model ViT-B-32 \
# --name "ViT-B-32-Vanilla" \