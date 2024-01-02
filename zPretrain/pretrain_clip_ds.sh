#!/bin/bash
set -ex

export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NNODES)
export RANK=$SLURM_PROCID
export LOCAL_RANK=$(expr $SLURM_PROCID % $GPUS_PER_NODE)
export CUDA_DEVICE_MAX_CONNECTIONS=1 # for async gradient all reduce

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "fp16": {
    "enabled": false,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

  exec python -u ./pretrain_CLIP_ds.py \
    --transformer-impl local \
    --global-batch-size $GLOBAL_BATCH_SIZE \
	  --tensor-model-parallel-size 1 \
	  --pipeline-model-parallel-size 1 \
    --micro-batch-size $MICRO_BATCH_SIZE \
	  --v-num-layers 28 \
    --v-hidden-size 1792 \
    --v-num-attention-heads 8 \
    --v-seq-length 264 \
    --num-layers 18 \
	  --hidden-size 1280 \
    --num-attention-heads 20 \
    --seq-length 77 \
    --max-position-embeddings 1024 \
    --train-samples $TRAIN_SAMPLES \
	  --lr-decay-samples 4882800 \
    --lr 0.0001 \
    --min-lr 0.00001 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 0 \
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
    --clip-grad 1.0 \
    --weight-decay 0.01 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.002 \
    --tokenizer-type CLIPTokenizer \
    --img-h 256 \
    --img-w 256 \
    --v-global-average-pool \
    --v-concat-cls-token \
    $extra_args \
    # --log-timers-to-tensorboard \
    # --log-memory-to-tensorboard \
    # --tensorboard-dir $OUTPUT_DIR \
    

