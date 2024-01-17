#!/bin/bash
set -x
# model config
export MODEL_NAME='MEGA-CLIP'
export VISION_L=56
export TEXT_L=36

# nico config
export GPUS_PER_NODE='4'
export NODELIST='nico[1]'
PARTITION='Big'

export NNODES=$(scontrol show hostnames ${NODELIST} | wc -l)
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Batch size
export GLOBAL_BATCH_SIZE=64
export MICRO_BATCHES=8
export TRAIN_SAMPLES=$(( $GLOBAL_BATCH_SIZE * 10)) 

# STAGE_MBS will be adjusted when using TRAINING_MODE=1
export STAGE_MBS=24 # for Indep-modal

# extra setting
export CHECKPOINT='0'
export LOG='0'
export LOG_LEVEL=0 # [0,1,2]

# Training mode
export TRAINING_MODE='4' # 0:独立模态 1:混合模态 2:纯DP 3:ZeRO 4: 双向混合流水线 5:混合模态+Chimera双向

export extra_args=""

if [ $TRAINING_MODE == '0' ]; then
    export EXP_NAME='Indep'

    export EXTRA_WORLD_SIZE=1

    export TENSOR_MODEL_PARALLEL=1
    export PIPELINE_MODEL_PARALLEL=3

    export XTENSOR_MODEL_PARALLEL=1
    export XPIPELINE_MODEL_PARALLEL=1

    if [ $EXTRA_WORLD_SIZE -ge $WORLD_SIZE ]; then
        echo "Error: EXTRA_WORLD_SIZE must be less than WORLD_SIZE."
        exit 1
    fi

    if [ $(expr $EXTRA_WORLD_SIZE % $(($XTENSOR_MODEL_PARALLEL*$XPIPELINE_MODEL_PARALLEL))) -ne 0 ]; then
        echo "Error: EXTRA_WORLD_SIZE must be divisible by XTENSOR_MODEL_PARALLEL * XPIPELINE_MODEL_PARALLEL."
        exit 1
    fi

    if [ $(expr $(($WORLD_SIZE-$EXTRA_WORLD_SIZE)) % $(($TENSOR_MODEL_PARALLEL*$PIPELINE_MODEL_PARALLEL))) -ne 0 ]; then
        echo "Error: (WORLD_SIZE - EXTRA_WORLD_SIZE) must be divisible by TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL."
        exit 1
    fi

    export DATA_PARALLEL_SIZE=$(expr $(($WORLD_SIZE-$EXTRA_WORLD_SIZE)) / $(($TENSOR_MODEL_PARALLEL*$PIPELINE_MODEL_PARALLEL)))
    export XDATA_PARALLEL_SIZE=$(expr $EXTRA_WORLD_SIZE / $(($XTENSOR_MODEL_PARALLEL*$XPIPELINE_MODEL_PARALLEL)))
    # find suitable STAGE_MBS
    _upper_bound=3
    for ((stage_mbs=$(($GLOBAL_BATCH_SIZE / $MICRO_BATCHES + _upper_bound)) ; stage_mbs>0; stage_mbs--)); do
        if ((stage_mbs % $DATA_PARALLEL_SIZE == 0)) && ((stage_mbs % $XDATA_PARALLEL_SIZE == 0)); then
            export STAGE_MBS=$stage_mbs
            echo "find suitable stage_mbs: $stage_mbs"
            break
        fi
    done
    export MICRO_BATCH_SIZE=$(expr $STAGE_MBS / $DATA_PARALLEL_SIZE)
    export XMICRO_BATCH_SIZE=$(expr $STAGE_MBS / $XDATA_PARALLEL_SIZE)
    export GLOBAL_BATCH_SIZE=$(( $STAGE_MBS * $MICRO_BATCHES))
    LOG_DIR=${EXP_NAME}\_TP$TENSOR_MODEL_PARALLEL\_PP$PIPELINE_MODEL_PARALLEL\_DP$DATA_PARALLEL_SIZE\_XTP$XTENSOR_MODEL_PARALLEL\_XPP$XPIPELINE_MODEL_PARALLEL\_XDP$XDATA_PARALLEL_SIZE\_gbs$GLOBAL_BATCH_SIZE\_mbs$MICRO_BATCH_SIZE\_xmbs$XMICRO_BATCH_SIZE
    LOG_NAME=${MODEL_NAME}\_tL$TEXT_L\_vL$VISION_L\_#mb$MICRO_BATCHES\_$(date -Iseconds).log 

elif [[ $TRAINING_MODE == '1' || $TRAINING_MODE == '5' ]]; then
    export EXP_NAME='Mix'
    export TENSOR_MODEL_PARALLEL=1
    export PIPELINE_MODEL_PARALLEL=4

    if [ $(expr $(($WORLD_SIZE)) % $(($TENSOR_MODEL_PARALLEL*$PIPELINE_MODEL_PARALLEL))) -ne 0 ]; then
        echo "Error: (WORLD_SIZE - EXTRA_WORLD_SIZE) must be divisible by TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL."
        exit 1
    fi
    export DATA_PARALLEL_SIZE=$(expr $WORLD_SIZE / $(($TENSOR_MODEL_PARALLEL*$PIPELINE_MODEL_PARALLEL)))
    export MICRO_BATCH_SIZE=$(expr $GLOBAL_BATCH_SIZE / $(($DATA_PARALLEL_SIZE*$MICRO_BATCHES)))


    LOG_DIR=${EXP_NAME}\_TP$TENSOR_MODEL_PARALLEL\_PP$PIPELINE_MODEL_PARALLEL\_DP$DATA_PARALLEL_SIZE\_gbs$GLOBAL_BATCH_SIZE\_mbs$MICRO_BATCH_SIZE
    LOG_NAME=${MODEL_NAME}\_tL$TEXT_L\_vL$VISION_L\_mbs$MICRO_BATCH_SIZE\_$(date -Iseconds).log
    extra_args=" --uniform-modility ${extra_args}"
    if [ $TRAINING_MODE == '5' ]; then
        extra_args=" --bidirectional-pipeline-size 4 ${extra_args}"
        extra_args=" --bidirectional-pipeline ${extra_args}"
    fi

elif [ $TRAINING_MODE == '2' ]; then
    export EXP_NAME='PureDP'

    export DATA_PARALLEL_SIZE=$(($GPUS_PER_NODE*$NNODES))
    export MICRO_BATCH_SIZE=$(expr $GLOBAL_BATCH_SIZE / $(($DATA_PARALLEL_SIZE*$MICRO_BATCHES)))

    LOG_DIR=${EXP_NAME}\_d$DATA_PARALLEL_SIZE\_gbs$GLOBAL_BATCH_SIZE\_mbs$MICRO_BATCH_SIZE
    LOG_NAME=${MODEL_NAME}\_tL$TEXT_L\_vL$VISION_L\_mbs$MICRO_BATCH_SIZE\_$(date -Iseconds).log
    extra_args=" --pure-dp ${extra_args}"

elif [ $TRAINING_MODE == '3' ]; then
    export EXP_NAME='ZeRO'
    export ZERO_STAGE='1'

    export DATA_PARALLEL_SIZE=$(($GPUS_PER_NODE*$NNODES))
    export MICRO_BATCH_SIZE=$(expr $GLOBAL_BATCH_SIZE / $(($DATA_PARALLEL_SIZE*$MICRO_BATCHES)))
    export DS_CONFIG=ds_config.json

    LOG_DIR=${EXP_NAME}\_d$DATA_PARALLEL_SIZE\_gbs$GLOBAL_BATCH_SIZE\_mbs$MICRO_BATCH_SIZE
    LOG_NAME=${MODEL_NAME}\_tL$TEXT_L\_vL$VISION_L\_mbs$MICRO_BATCH_SIZE\_$(date -Iseconds).log
    
    extra_args=" --deepspeed ${extra_args}"
    extra_args=" --deepspeed_config=$DS_CONFIG ${extra_args}"
    extra_args=" --zero-stage=$ZERO_STAGE ${extra_args}"

elif [ $TRAINING_MODE == '4' ]; then
    export EXP_NAME='Flexpipe'
    export TENSOR_MODEL_PARALLEL=1
    export PIPELINE_MODEL_PARALLEL=4

    if [ $(expr $(($WORLD_SIZE)) % $(($TENSOR_MODEL_PARALLEL*$PIPELINE_MODEL_PARALLEL))) -ne 0 ]; then
        echo "Error: (WORLD_SIZE - EXTRA_WORLD_SIZE) must be divisible by TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL."
        exit 1
    fi
    export DATA_PARALLEL_SIZE=$(expr $WORLD_SIZE / $(($TENSOR_MODEL_PARALLEL*$PIPELINE_MODEL_PARALLEL)))
    export MICRO_BATCH_SIZE=$(expr $GLOBAL_BATCH_SIZE / $(($DATA_PARALLEL_SIZE*$MICRO_BATCHES)))
    LOG_DIR=${EXP_NAME}\_TP$TENSOR_MODEL_PARALLEL\_PP$PIPELINE_MODEL_PARALLEL\_DP$DATA_PARALLEL_SIZE\_gbs$GLOBAL_BATCH_SIZE\_mbs$MICRO_BATCH_SIZE
    LOG_NAME=${MODEL_NAME}\_tL$TEXT_L\_vL$VISION_L\_mbs$MICRO_BATCH_SIZE\_$(date -Iseconds).log
    extra_args=" --bidirectional-pipeline-size 4 ${extra_args}"
    extra_args=" --bidirectional-pipeline ${extra_args}"
fi

if [ $CHECKPOINT == '1' ]; then
    extra_args=" --recompute-activations ${extra_args}"
fi
extra_args=" --timing-log-level $LOG_LEVEL ${extra_args}"
# extra_args=" --v-concat-cls-token ${extra_args}"

export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)


if [ $LOG == '1' ]; then
    mkdir -p ./logs
    mkdir -p ./logs/${LOG_DIR}
    if [ $TRAINING_MODE == '0' ]; then
        srun \
        --exclusive=user \
        -p $PARTITION \
        -K \
        -N $NNODES \
        -w $NODELIST \
        --time 20:00 \
        --job-name=megaCLIP \
        --ntasks-per-node=$GPUS_PER_NODE \
        --gres=gpu:$GPUS_PER_NODE \
        --export=ALL \
        bash ./zPretrain/pretrain_clip_e.sh > ./logs/$LOG_DIR/$LOG_NAME 2>&1

    elif [[ $TRAINING_MODE == '1' || $TRAINING_MODE == '5' ]]; then
        srun \
        --exclusive=user \
        -p $PARTITION \
        -K \
        -N $NNODES \
        -w $NODELIST \
        --time 20:00 \
        --job-name=MegaCLIP \
        --ntasks-per-node=$GPUS_PER_NODE \
        --gres=gpu:$GPUS_PER_NODE \
        --export=ALL \
        bash ./zPretrain/pretrain_clip_bl_3d.sh > ./logs/$LOG_DIR/$LOG_NAME 2>&1

    elif [ $TRAINING_MODE == '2' ]; then
        srun \
        --exclusive=user \
        -p $PARTITION \
        -K \
        -N $NNODES \
        -w $NODELIST \
        --time 20:00 \
        --job-name=MegaCLIP \
        --ntasks-per-node=$GPUS_PER_NODE \
        --gres=gpu:$GPUS_PER_NODE \
        --export=ALL \
        bash zPretrain/pretrain_clip_bl.sh > ./logs/$LOG_DIR/$LOG_NAME 2>&1
    
    elif [ $TRAINING_MODE == '2' ]; then
        srun \
        --exclusive=user \
        -p $PARTITION \
        -K \
        -N $NNODES \
        -w $NODELIST \
        --time 20:00 \
        --job-name=MegaCLIP \
        --ntasks-per-node=$GPUS_PER_NODE \
        --gres=gpu:$GPUS_PER_NODE \
        --export=ALL \
        bash zPretrain/pretrain_clip_ds.sh > ./logs/$LOG_DIR/$LOG_NAME 2>&1
    fi
elif [ $LOG == '0' ]; then
    if [ $TRAINING_MODE == '0' ]; then
        srun \
        --exclusive=user \
        -p $PARTITION \
        -K \
        -N $NNODES \
        -w $NODELIST \
        --time 20:00 \
        --job-name=megaCLIP \
        --ntasks-per-node=$GPUS_PER_NODE \
        --gres=gpu:$GPUS_PER_NODE \
        --export=ALL \
        bash ./zPretrain/pretrain_clip_e.sh

    elif [[ $TRAINING_MODE == '1' || $TRAINING_MODE == '5' ]]; then
        srun \
        --exclusive=user \
        -p $PARTITION \
        -K \
        -N $NNODES \
        -w $NODELIST \
        --time 20:00 \
        --job-name=MegaCLIP \
        --ntasks-per-node=$GPUS_PER_NODE \
        --gres=gpu:$GPUS_PER_NODE \
        --export=ALL \
        bash zPretrain/pretrain_clip_bl_3d.sh

    elif [ $TRAINING_MODE == '2' ]; then
        srun \
        --exclusive=user \
        -p $PARTITION \
        -K \
        -N $NNODES \
        -w $NODELIST \
        --time 20:00 \
        --job-name=MegaCLIP \
        --ntasks-per-node=$GPUS_PER_NODE \
        --gres=gpu:$GPUS_PER_NODE \
        --export=ALL \
        bash zPretrain/pretrain_clip_bl.sh
    
    elif [ $TRAINING_MODE == '3' ]; then
        srun \
        --exclusive=user \
        -p $PARTITION \
        -K \
        -N $NNODES \
        -w $NODELIST \
        --time 20:00 \
        --job-name=MegaCLIP \
        --ntasks-per-node=$GPUS_PER_NODE \
        --gres=gpu:$GPUS_PER_NODE \
        --export=ALL \
        bash zPretrain/pretrain_clip_ds.sh
    elif [ $TRAINING_MODE == '4' ]; then
        srun \
        --exclusive=user \
        -p $PARTITION \
        -K \
        -N $NNODES \
        -w $NODELIST \
        --time 20:00 \
        --job-name=MegaCLIP \
        --ntasks-per-node=$GPUS_PER_NODE \
        --gres=gpu:v132p:$GPUS_PER_NODE \
        --export=ALL \
        bash zPretrain/pretrain_clip_flexpipe.sh
    fi

fi



