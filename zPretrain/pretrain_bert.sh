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


CHECKPOINT_PATH=<Specify path>
VOCAB_FILE=<Specify path to file>/bert-vocab.txt
DATA_PATH=<Specify path and file prefix>_text_sentence

BERT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr 0.0001 \
    --train-iters 2000000 \
    --lr-decay-iters 990000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun pretrain_bert.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH