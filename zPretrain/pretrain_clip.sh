#! /bin/bash
set -x

export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NNODES)
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export RANK=$SLURM_PROCID
export CUDA_DEVICE_MAX_CONNECTIONS=1 # for async gradient all reduce

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

source ~/workspace/mega-env/bin/activate
# source /opt/spack/share/spack/setup-env.sh;spack load cuda@11.8.0;spack load gcc@10.2.0;spack load nccl@2.10.3
cd /home/zanzong/workspace/Megatron-CLIP


# TRAIN_SAMPLES=$(( $GLOBAL_BATCH_SIZE * 50))
TRAIN_SAMPLES=$(( $GLOBAL_BATCH_SIZE * 2))

PROFILER_LOG_PATH=$PROFILER_LOG_PATH \
exec python -W ignore \
        ./pretrain_CLIP.py \
        --transformer-impl local \
	--tensor-model-parallel-size 2 \
	--pipeline-model-parallel-size 1 \
	--num-layers 12 \
	--hidden-size 512 \
        --num-attention-heads 8 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --micro-batch-size 4 \
        --global-batch-size 64 \
        --train-samples 128 \
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
        --fp16 \
        --extra-world-size 4 \
        --xtensor-model-parallel-size 2 \
        --xpipeline-model-parallel-size 1 \
        --tokenizer-type CLIPTokenizer \
        --img-h 256 \
        --img-w 256 \
        --v-global-average-pool \
        # --recompute-granularity selective \
        # --train-data-path /mnt/zoltan/zanzong/CC3M/cc3m/{00000..00331}.tar \
        # --num-layers-per-virtual-pipeline-stage 1 \
        # --tensorboard-profile \
        # --tensorboard-dir ./tensorboard \
        # --log-memory-to-tensorboard \
        # --log-timers-to-tensorboard
