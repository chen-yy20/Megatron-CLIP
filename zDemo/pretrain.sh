#! /bin/bash
set -x

GPUS_PER_NODE=8

export NNODES=$(( $DATA_PARALLEL_SIZE * $TENSOR_PARALLEL_SIZE * $PIPELINE_PARALLEL_SIZE / $GPUS_PER_NODE))
export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NNODES)
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export RANK=$SLURM_PROCID

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

# source ~/venv/bin/activate
conda activate mega

cd /home/chen-yy20/Megatron-LM

VOCAB_FILE=/mnt/zoltan/zanzong/fastmoe-dataset/wikidataset/gpt2-vocab.json
MERGE_FILE=/mnt/zoltan/zanzong/fastmoe-dataset/wikidataset/gpt2-merges.txt
DATA_PATH=/mnt/zoltan/zanzong/fastmoe-dataset/wikidataset/my-bert_text_sentence

# TRAIN_SAMPLES=$(( $GLOBAL_BATCH_SIZE * 50))
TRAIN_SAMPLES=$(( $GLOBAL_BATCH_SIZE * 15))

PROFILER_LOG_PATH=$PROFILER_LOG_PATH \
exec python3 \
        ./pretrain_gpt.py \
        --vocab-file $VOCAB_FILE \
	--merge-file $MERGE_FILE \
	--tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
	--pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
	--num-layers $NUM_LAYERS \
	--hidden-size $HIDDEN_SIZE \
        --num-attention-heads $NUM_ATTN_HEADS \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --train-samples $TRAIN_SAMPLES \
	--lr-decay-samples 4882800 \
        --lr 0.0001 \
        --min-lr 0.00001 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters -1 \
        --data-path ${DATA_PATH} \
        --split 100,0,0 \
        --clip-grad 1.0 \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.002 \
        --fp16 \
        --DDP-impl local \
	--fmoefy \
	--num-experts 1 \
	--top-k 2 \
        # --log-num-zeros-in-grad \
        # --log-params-norm 
