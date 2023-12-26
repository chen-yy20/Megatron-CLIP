#! /bin/bash
set -x

export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NNODES)
export RANK=$SLURM_PROCID
export CUDA_DEVICE_MAX_CONNECTIONS=1 # for async gradient all reduce

exec python -W ignore \
        ./pretrain_CLIP.py \
        --transformer-impl local \
        --global-batch-size $GLOBAL_BATCH_SIZE \
	--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL \
	--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --extra-world-size $EXTRA_WORLD_SIZE \
        --xtensor-model-parallel-size $XTENSOR_MODEL_PARALLEL \
        --xpipeline-model-parallel-size $XPIPELINE_MODEL_PARALLEL \
        --xmicro-batch-size $XMICRO_BATCH_SIZE \
	--v-num-layers $VISION_L \
        --v-hidden-size 1792 \
        --v-num-attention-heads 8 \
        --v-seq-length 264 \
        --num-layers $TEXT_L \
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
        --clip-grad 0.0 \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.002 \
        --tokenizer-type CLIPTokenizer \
        --img-h 256 \
        --img-w 256 \
        --v-global-average-pool \
        # --tensorboard-profile \
        # --profile-ranks 12 13 14 15 \
        # --profile-dir logs/v12-t4-12_13_14_15
        # --recompute-granularity selective \
        # --train-data-path /mnt/zoltan/zanzong/CC3M/cc3m/{00000..00331}.tar \
        # --num-layers-per-virtual-pipeline-stage 1 \
        # --tensorboard-profile \
        # --tensorboard-dir ./tensorboard \
        # --log-memory-to-tensorboard \
        # --log-timers-to-tensorboard
