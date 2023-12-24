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


export GLOBAL_BATCH_SIZE=1024
# export MICRO_BATCHES=128
# export TENSOR_MODEL_PARALLEL=1
# export PIPELINE_MODEL_PARALLEL=4

DATA_PARALLEL_SIZE=$(expr $WORLD_SIZE / $(($TENSOR_MODEL_PARALLEL*$PIPELINE_MODEL_PARALLEL)))
MICRO_BATCH_SIZE=$(expr $GLOBAL_BATCH_SIZE / $(($DATA_PARALLEL_SIZE*$MICRO_BATCHES)))
# TRAIN_SAMPLES=$(( $GLOBAL_BATCH_SIZE * 50))
TRAIN_SAMPLES=$(($GLOBAL_BATCH_SIZE * 5))

exec python -W ignore \
        ./pretrain_CLIP_bl_3d.py \
        --transformer-impl local \
        --global-batch-size $GLOBAL_BATCH_SIZE \
	--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL \
	--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL \
        --micro-batch-size $MICRO_BATCH_SIZE \
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
        --uniform-modility \
        # --timing-log-level 2 \
