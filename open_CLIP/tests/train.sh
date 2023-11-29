#!/bin/bash
set -x

eval "$(source $PYTHONENV)"
export MASTER_PORT=12802
export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NNODE)
export LOCAL_RANK=$NODE_RANK
export WORLD_SIZE=$(($NTASK_PER_NODE*$NNODE))
export RANK=$SLURM_PROCID

cd /home/zanzong/workspace/flexpipe/models/open_clip/tests
export PYTHONPATH="$PYTHONPATH:$PWD/src"

# exec python -u test_async_sv.py $LOCAL_RANK
exec python -u test_interrupt_backward.py $LOCAL_RANK
