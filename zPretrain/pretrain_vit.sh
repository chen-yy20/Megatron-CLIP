#! /bin/bash
set -x



export NNODES=$(( $DATA_PARALLEL_SIZE * $TENSOR_PARALLEL_SIZE * $PIPELINE_PARALLEL_SIZE / $GPUS_PER_NODE))
export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NNODES)
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export RANK=$SLURM_PROCID
export CUDA_DEVICE_MAX_CONNECTIONS=1 # for async gradient all reduce

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

source ~/mega_env/bin/activate
cd /home/chen-yy20/Megatron-LM

# VOCAB_FILE=~/Megatron-LM/zData/wikidataset/gpt2-vocab.json
# MERGE_FILE=~/Megatron-LM/zData/wikidataset/gpt2-merges.txt
# DATA_PATH=~/Megatron-LM/zData/wikidataset/my-bert_text_sentence

VOCAB_FILE=/mnt/zoltan/zanzong/fastmoe-dataset/wikidataset/gpt2-vocab.json
MERGE_FILE=/mnt/zoltan/zanzong/fastmoe-dataset/wikidataset/gpt2-merges.txt
DATA_PATH=/mnt/zoltan/zanzong/fastmoe-dataset/wikidataset/my-bert_text_sentence

# TRAIN_SAMPLES=$(( $GLOBAL_BATCH_SIZE * 50))
TRAIN_SAMPLES=$(( $GLOBAL_BATCH_SIZE * 2))

CLASSIFIER_ARGS="
   	   --tensor-model-parallel-size 1 \
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --patch-dim 4 \
        --seq-length 3136 \
        --max-position-embeddings 3136 \
        --img-h 224 \
        --img-w 224 \
        --mask-factor 1.0 \
        --fp16 \
        --train-iters 750000 \
        --lr-decay-style cosine \
        --micro-batch-size 4 \
        --global-batch-size 1024 \
        --lr 0.0005 \
        --min-lr 0.00001 \
        --attention-dropout 0.0 \
        --weight-decay 0.05 \
        --lr-warmup-iters 12500 \
        --clip-grad 1.0 \
        --no-gradient-accumulation-fusion \
        --num-workers 4 \
        --DDP-impl torch "

DATA_ARGS="
     --tokenizer-type NullTokenizer \
     --vocab-size 0 \
     --data-path $DATA_PATH_TRAIN $DATA_PATH_VAL \
     --no-data-sharding \
     --split 949,50,1 \
"

OUTPUT_ARG="
     --log-interval 32 \
     --save-interval 10000 \
     --eval-interval 2500 \
     --eval-iters 100 \
     --tensorboard-dir ${CHECKPOINT_PATH} \
"

PROFILER_LOG_PATH=$PROFILER_LOG_PATH \
exec python \
        ./pretrain_vision_classify.py \
        $CLASSIFIER_ARGS \
     $DATA_ARGS \
     $OUTPUT_ARGS \
     --save $CHECKPOINT_PATH \
     --load $CHECKPOINT_PATH