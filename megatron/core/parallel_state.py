# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import os
from typing import Optional

import torch
import torch.distributed

from .utils import GlobalMemoryBuffer

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None

# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None

# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None

# Embedding group.
_EMBEDDING_GROUP = None

# Position embedding group.
_POSITION_EMBEDDING_GROUP = None

# Loss calculation group for multimodal contrastive models.
_CONTRASTIVE_LOSS_GROUP = None
_CONTRASTIVE_LOSS_RANK = None

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

_DATA_PARALLEL_GROUP_GLOO = None
# tensor model parallel group and data parallel group combined
# used for fp8 and moe training
_TENSOR_AND_DATA_PARALLEL_GROUP = None

# Expert parallel group that the current rank belongs to.
_TENSOR_AND_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP = None


_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None


# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None


# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None


# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS = None

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None

_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None

# Flag to identify if it is an extra branch
_IS_EXTRA_BRANCH = None
_HAVE_EXTRA_BRANCH = None
_REAL_WORLD_SIZE = None

# Save all valid groups for all process
_ALL_DATA_PARALLEL_GROUPS = {}
_ALL_TENSOR_MODEL_PARALLEL_GROUPS = {}
_ALL_PIPELINE_MODEL_PARALLEL_GROUPS = {}

# def print_rank_0(message):
#     """If distributed is initialized, print only on rank 0."""
#     if torch.distributed.is_initialized():
#         if torch.distributed.get_rank() == 0:
#             print(message, flush=True)
#     else:
#         print(message, flush=True)

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    extra_world_size: Optional[int] = None,
    xtensor_model_parallel_size: Optional[int] = None,
    xpipeline_model_parallel_size: Optional[int] = None, 
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    bidirectional_pipeline=False,
    bidirectional_pipeline_size=1
) -> None:
    """Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    is_multi_branch = True if extra_world_size > 0 else False

    # 对两个world的初始化
        
    global _IS_EXTRA_BRANCH
    global _HAVE_EXTRA_BRANCH
    global _REAL_WORLD_SIZE
    _REAL_WORLD_SIZE = world_size
    rank = torch.distributed.get_rank()
    offset = 0
    real_tensor_model_parallel_size = tensor_model_parallel_size
    real_pipeline_model_parallel_size = pipeline_model_parallel_size
    ori_world_size = world_size - extra_world_size
    data_parallel_size = ori_world_size // (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
    xdata_parallel_size = extra_world_size // (xtensor_model_parallel_size * xpipeline_model_parallel_size * context_parallel_size) if is_multi_branch else 1
    def print_rank_0(msg, extra=False):
        if extra:
            if rank - offset == 0:
                print(msg, flush=True)
        else:
            if rank == 0:
                print(msg, flush=True)
        torch.distributed.barrier()
    
    if is_multi_branch:
        assert extra_world_size < world_size, f'GPUs for extra branch size {extra_world_size} must be less than whole world size {world_size}.'
        _HAVE_EXTRA_BRANCH = True
        if rank >= world_size - extra_world_size:
            _IS_EXTRA_BRANCH = True
            offset = world_size - extra_world_size
            world_size = extra_world_size
            _REAL_WORLD_SIZE = world_size
            # print(f"Rank {rank} is an extra branch. World size {get_real_world_size()}")
            # print("Ready to set new TP PP DP world size...")
            real_tensor_model_parallel_size = xtensor_model_parallel_size
            real_pipeline_model_parallel_size = xpipeline_model_parallel_size
            
        else:
            _IS_EXTRA_BRANCH = False
            offset = 0
            world_size = world_size - extra_world_size
            _REAL_WORLD_SIZE = world_size
            
        print_rank_0(f"World_size: {world_size}| Extra_world_size: {extra_world_size}")
    else:
        _HAVE_EXTRA_BRANCH = False

    if (
        world_size
        % (real_tensor_model_parallel_size * real_pipeline_model_parallel_size * context_parallel_size)
        != 0
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )
    real_data_parallel_size: int = world_size // (
        real_tensor_model_parallel_size * real_pipeline_model_parallel_size * context_parallel_size
    )
    if real_data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({real_data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )
    
    num_tensor_model_parallel_groups: int = ori_world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = ori_world_size // pipeline_model_parallel_size

    x_num_tensor_model_parallel_groups: int = extra_world_size // xtensor_model_parallel_size
    x_num_pipeline_model_parallel_groups: int = extra_world_size // xpipeline_model_parallel_size


    if virtual_pipeline_model_parallel_size is not None:
        if not real_pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    print_rank_0("========================================================")
    # Build the data-parallel groups.
    # FIXME: This is a hacky way to build the data parallel groups.
    group = None
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP

    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
    global _ALL_DATA_PARALLEL_GROUPS
    all_data_parallel_group_ranks_with_cp = []
    all_xdata_parallel_group_ranks_with_cp = []

    ranks = []

    # original world DP set up 
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups

        tp_step = context_parallel_size * tensor_model_parallel_size
        for j in range(tp_step):
            ranks = range(start_rank + j, end_rank, tp_step)
            print_rank_0(f" DP group: {list(ranks)}")
            group = torch.distributed.new_group(ranks)
            # if tuple(ranks) not in _ALL_DATA_PARALLEL_GROUPS:
            #     _ALL_DATA_PARALLEL_GROUPS[tuple(ranks)] = group
            # group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                 
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                # _DATA_PARALLEL_GROUP_GLOO = group_gloo
                _DATA_PARALLEL_GLOBAL_RANKS = ranks
        
        for j in range(tensor_model_parallel_size):
            ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
            group_with_cp = torch.distributed.new_group(ranks_with_cp)
            # group_with_cp_gloo = torch.distributed.new_group(ranks_with_cp, backend="gloo")
            if rank in ranks_with_cp:
                _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                # _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
                _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp
    if is_multi_branch:
    # extra world DP set up 
        for i in range(xpipeline_model_parallel_size):
            start_rank = i * x_num_pipeline_model_parallel_groups + ori_world_size
            end_rank = (i + 1) * x_num_pipeline_model_parallel_groups + ori_world_size

            tp_step = context_parallel_size * xtensor_model_parallel_size
            for j in range(tp_step):
                ranks = range(start_rank + j, end_rank, tp_step)
                print_rank_0(f"xDP group: {list(ranks)}", extra=True)
                group = torch.distributed.new_group(ranks)
                # if tuple(ranks) not in _ALL_DATA_PARALLEL_GROUPS:
                #     _ALL_DATA_PARALLEL_GROUPS[tuple(ranks)] = group
                # group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                    
                if rank in ranks:
                    _DATA_PARALLEL_GROUP = group
                    # _DATA_PARALLEL_GROUP_GLOO = group_gloo
                    _DATA_PARALLEL_GLOBAL_RANKS = ranks
            
            for j in range(xtensor_model_parallel_size):
                ranks_with_cp = range(start_rank + j, end_rank, xtensor_model_parallel_size)
                all_xdata_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
                group_with_cp = torch.distributed.new_group(ranks_with_cp)
                # group_with_cp_gloo = torch.distributed.new_group(ranks_with_cp, backend="gloo")
                if rank in ranks_with_cp:
                    _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                    # _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
                    _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp
        torch.distributed.barrier()

    print_rank_0("========================================================")
    
    # Apply SHARP to DP process groups
    if use_sharp:
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=get_data_parallel_group(with_context_parallel=context_parallel_size > 1),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_SHARP_DISABLE=1` to restrict SHARP application to DP process groups
        os.environ["NCCL_SHARP_DISABLE"] = "1"

    # Build the context-parallel groups.
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                i * num_pipeline_model_parallel_groups
                + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                i * num_pipeline_model_parallel_groups
                + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    _CONTEXT_PARALLEL_GROUP = group
                    _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
    if is_multi_branch:
        for i in range(xpipeline_model_parallel_size):
            for j in range(xdata_parallel_size):
                start_rank = (
                    i * x_num_pipeline_model_parallel_groups
                    + j * xtensor_model_parallel_size * context_parallel_size
                )
                start_rank += ori_world_size
                end_rank = (
                    i * x_num_pipeline_model_parallel_groups
                    + (j + 1) * xtensor_model_parallel_size * context_parallel_size
                )
                end_rank += ori_world_size
                for k in range(xtensor_model_parallel_size):
                    ranks = range(start_rank + k, end_rank, xtensor_model_parallel_size)
                    group = torch.distributed.new_group(ranks)
                    if rank in ranks:
                        _CONTEXT_PARALLEL_GROUP = group
                        _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
    
    # for i in range(max(pipeline_model_parallel_size, xpipeline_model_parallel_size)):
    #     for j in range(max(data_parallel_size, xdata_parallel_size)):
    #         start_rank = (
    #             i * num_pipeline_model_parallel_groups
    #             + j * real_tensor_model_parallel_size * context_parallel_size
    #             + offset
    #         )
    #         end_rank = (
    #             i * num_pipeline_model_parallel_groups
    #             + (j + 1) * real_tensor_model_parallel_size * context_parallel_size
    #             + offset
    #         )
    #         for k in range(max(tensor_model_parallel_size, xtensor_model_parallel_size)):
               
    #             if k < real_tensor_model_parallel_size and j < real_data_parallel_size and i < real_pipeline_model_parallel_size:
    #                 ranks = range(start_rank + k, end_rank, real_tensor_model_parallel_size)
    #             # print_rank_0(f"CP group: {ranks}")
    #             group = torch.distributed.new_group(ranks)
                
    #             if rank in ranks:
    #                 _CONTEXT_PARALLEL_GROUP = group
    #                 _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
    
        
    torch.distributed.barrier()

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for i in range(data_parallel_size * context_parallel_size):
        ranks = [
            data_parallel_group_ranks_with_cp[i]
            for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
        ]
        print_rank_0(f" MP group: {list(ranks)}")
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group
    if is_multi_branch:
        for i in range(xdata_parallel_size * context_parallel_size):
            ranks = [
                data_parallel_group_ranks_with_cp[i]
                for data_parallel_group_ranks_with_cp in all_xdata_parallel_group_ranks_with_cp
            ]
            print_rank_0(f"xMP group: {list(ranks)}", extra=True)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _MODEL_PARALLEL_GROUP = group

    # for i in range(max(data_parallel_size * context_parallel_size, xdata_parallel_size * context_parallel_size)):
    #     if i < real_data_parallel_size * context_parallel_size:
    #         ranks = [
    #             data_parallel_group_ranks_with_cp[i]
    #             for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
    #         ]
    #         print_rank_0(f"Model parallel group: {ranks}")
    #     group = torch.distributed.new_group(ranks)   
    #     if rank in ranks:
    #         _MODEL_PARALLEL_GROUP = group

    print_rank_0("========================================================")

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _ALL_TENSOR_MODEL_PARALLEL_GROUPS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'

    # ori TP group set up
    for i in range(num_tensor_model_parallel_groups):    
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        print_rank_0(f" TP group: {list(ranks)}")
        group = torch.distributed.new_group(ranks)
        # if tuple(ranks) not in _ALL_TENSOR_MODEL_PARALLEL_GROUPS:
        #     _ALL_TENSOR_MODEL_PARALLEL_GROUPS[tuple(ranks)] = group
                
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # extra TP group set up
    if is_multi_branch:
        for i in range(x_num_tensor_model_parallel_groups):    
            ranks = range(i * xtensor_model_parallel_size + ori_world_size, (i + 1) * xtensor_model_parallel_size + ori_world_size)
            print_rank_0(f"xTP group: {list(ranks)}", extra=True)
            group = torch.distributed.new_group(ranks)
            # if tuple(ranks) not in _ALL_TENSOR_MODEL_PARALLEL_GROUPS:
            #     _ALL_TENSOR_MODEL_PARALLEL_GROUPS[tuple(ranks)] = group
                    
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group
    
    print_rank_0("========================================================")

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    global _ALL_PIPELINE_MODEL_PARALLEL_GROUPS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
    loss_rank = list() # the pipeline rank which participate loss calculating
    # ori PP group set up
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, ori_world_size, num_pipeline_model_parallel_groups)
        print_rank_0(f" PP group: {list(ranks)}")
        group = torch.distributed.new_group(ranks)
        # if tuple(ranks) not in _ALL_PIPELINE_MODEL_PARALLEL_GROUPS:
        #     _ALL_PIPELINE_MODEL_PARALLEL_GROUPS[tuple(ranks)] = group
        if bidirectional_pipeline:
            loss_rank.append(list(ranks)[pipeline_model_parallel_size - bidirectional_pipeline_size])
        loss_rank.append(list(ranks)[-1])
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Vision branch doesn't needs embedding.
        _EMBEDDING_GLOBAL_RANKS = []
        _POSITION_EMBEDDING_GLOBAL_RANKS = []
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        # if len(ranks) > 1:
        #     embedding_ranks = [ranks[0], ranks[-1]]
        #     position_embedding_ranks = [ranks[0]]
        #     if pipeline_model_parallel_split_rank is not None:
        #         if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
        #             embedding_ranks = [
        #                 ranks[0],
        #                 ranks[pipeline_model_parallel_split_rank],
        #                 ranks[-1],
        #             ]
        #         if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
        #             position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
        # else:
        #     embedding_ranks = ranks
        #     position_embedding_ranks = ranks

        # group = torch.distributed.new_group(embedding_ranks)
        # if rank in embedding_ranks:
        #     _EMBEDDING_GROUP = group
        # if rank in ranks:
        #     _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        # group = torch.distributed.new_group(position_embedding_ranks)
        # if rank in position_embedding_ranks:
        #     _POSITION_EMBEDDING_GROUP = group
        # if rank in ranks:
        #     _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # extra PP group set up
    if is_multi_branch:
        for i in range(x_num_pipeline_model_parallel_groups):
            ranks = range(i + ori_world_size, ori_world_size + extra_world_size, x_num_pipeline_model_parallel_groups)
            print_rank_0(f"xPP group: {list(ranks)}", extra=True)
            group = torch.distributed.new_group(ranks)
            # if tuple(ranks) not in _ALL_PIPELINE_MODEL_PARALLEL_GROUPS:
            #     _ALL_PIPELINE_MODEL_PARALLEL_GROUPS[tuple(ranks)] = group
            if bidirectional_pipeline:
                loss_rank.append(list(ranks)[pipeline_model_parallel_size - bidirectional_pipeline_size])
            loss_rank.append(list(ranks)[-1])
            if rank in ranks:
                _PIPELINE_MODEL_PARALLEL_GROUP = group
                _PIPELINE_GLOBAL_RANKS = ranks
        
            # Only language models need embedding group to sync grads.
            # Setup embedding group (to exchange gradients between
            # first and last stages).
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                if pipeline_model_parallel_split_rank is not None:
                    if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[pipeline_model_parallel_split_rank],
                            ranks[-1],
                        ]
                    if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks

            group = torch.distributed.new_group(embedding_ranks)
            if rank in embedding_ranks:
                _EMBEDDING_GROUP = group
            if rank in ranks:
                _EMBEDDING_GLOBAL_RANKS = embedding_ranks

            group = torch.distributed.new_group(position_embedding_ranks)
            if rank in position_embedding_ranks:
                _POSITION_EMBEDDING_GROUP = group
            if rank in position_embedding_ranks:
                _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks
    # print(f"rank={rank}, _POSITION_EMBEDDING_GLOBAL_RANKS={_POSITION_EMBEDDING_GLOBAL_RANKS}, _EMBEDDING_GLOBAL_RANKS={_EMBEDDING_GLOBAL_RANKS}", flush=True)
    global _CONTRASTIVE_LOSS_GROUP
    global _CONTRASTIVE_LOSS_RANK
    _CONTRASTIVE_LOSS_GROUP = torch.distributed.new_group(loss_rank)
    _CONTRASTIVE_LOSS_RANK = loss_rank
    print_rank_0(f"Create LOSS group: {loss_rank}")
    print_rank_0("========================================================")

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
        _TENSOR_AND_DATA_PARALLEL_GROUP is None
    ), 'Tensor + data parallel group is already initialized'
    tensor_and_data_group_size_with_cp: int = real_tensor_model_parallel_size * real_data_parallel_size * context_parallel_size
    num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
    former_ranks = []
    for i in range(max(ori_world_size // (tensor_model_parallel_size * data_parallel_size * context_parallel_size), 
                       extra_world_size // (xtensor_model_parallel_size * xdata_parallel_size * context_parallel_size))):
        if i < num_tensor_and_data_groups_with_cp:
            start_rank = i * tensor_and_data_group_size_with_cp + offset
            end_rank = start_rank + tensor_and_data_group_size_with_cp
            ranks = range(start_rank, end_rank)
        # print_rank_0(f"TP + DP group with CP: {list(ranks)}")
        group = torch.distributed.new_group(ranks)
        
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group

        for j in range(context_parallel_size):
            ranks = []
            for k in range(max(data_parallel_size, xdata_parallel_size)):
                if k < real_data_parallel_size and i < num_tensor_and_data_groups_with_cp:
                    start_rank = (
                        i * tensor_and_data_group_size_with_cp
                        + j * real_tensor_model_parallel_size
                        + k * real_tensor_model_parallel_size * context_parallel_size
                        + offset
                    )
                    end_rank = start_rank + real_tensor_model_parallel_size
                    ranks = ranks + list(range(start_rank, end_rank))
                    former_ranks = ranks
                else:
                    ranks = former_ranks
            # print_rank_0(f"TP + DP group: {list(ranks)}")
            group = torch.distributed.new_group(ranks)    
            if rank in ranks:
                _TENSOR_AND_DATA_PARALLEL_GROUP = group


    # # Build the tensor + expert parallel groups
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is None
    ), 'Tensor + expert parallel group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is None
    ), 'Data modulo expert group is already initialized'
    tensor_and_data_group_size: int = real_tensor_model_parallel_size * real_data_parallel_size # 2, 4
    num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size # 4 , 2
    tensor_and_expert_group_size: int = real_tensor_model_parallel_size * expert_model_parallel_size
    num_expert_groups: int = real_data_parallel_size // expert_model_parallel_size
    
    for i in range(max(ori_world_size // (tensor_model_parallel_size * data_parallel_size),
                       extra_world_size // (xtensor_model_parallel_size * xdata_parallel_size))):
        for j in range(max(data_parallel_size // expert_model_parallel_size, xdata_parallel_size // expert_model_parallel_size)):
            if i < num_tensor_and_data_groups and j < num_expert_groups:
                start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size + offset
                end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size + offset
                ranks = range(start_rank, end_rank)
            # print_rank_0(f"Tensor + expert parallel group: {ranks}")
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _TENSOR_AND_EXPERT_PARALLEL_GROUP = group


    for i in range(max(ori_world_size // (tensor_model_parallel_size * data_parallel_size),
                       extra_world_size // (xtensor_model_parallel_size * xdata_parallel_size))):
        if i < num_tensor_and_data_groups:
            start_rank = i * tensor_and_data_group_size + offset
            end_rank = (i + 1) * tensor_and_data_group_size + offset
        for j in range(max(data_parallel_size // expert_model_parallel_size, xdata_parallel_size // expert_model_parallel_size)):
            if i < num_tensor_and_data_groups and j < num_expert_groups:
                ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
            # print_rank_0(f"Data modulo expert parallel group: {ranks}")
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _DATA_MODULO_EXPERT_PARALLEL_GROUP = group



    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP

def is_extra_branch_rank():
    """Check if it is an extra branch"""
    # assert _IS_EXTRA_BRANCH is not None, 'IS EXTRA BRANCH is not initialized'
    if _IS_EXTRA_BRANCH is None:
        return False
    return _IS_EXTRA_BRANCH

def check_extra_branch_rank(rank):
    """Check if the given rank is an extra branch."""
    if not _HAVE_EXTRA_BRANCH:
        return False
    global _REAL_WORLD_SIZE
    if rank >= torch.distributed.get_world_size() - _REAL_WORLD_SIZE:
        return True
    else:
        return False
    
def get_all_groups(pg_type):
    """Enumerate all process groups on the world for the given type of parallel method.
    Return:
        groups (dict): Organized as {`device list`: group object, ...}
    """
    assert pg_type in ["tp", "dp", "pp"], "pg_type must be one of tp, dp or pp."
    if pg_type == "dp":
        return _ALL_DATA_PARALLEL_GROUPS
    elif pg_type == "tp":
        return _ALL_TENSOR_MODEL_PARALLEL_GROUPS
    else:
        return _ALL_PIPELINE_MODEL_PARALLEL_GROUPS

def has_extra_branch():
    """Check if it has extra branch"""
    assert _HAVE_EXTRA_BRANCH is not None, 'HAVE EXTRA BRANCH  is not initialized'
    return _HAVE_EXTRA_BRANCH

def get_real_world_size():
    assert _REAL_WORLD_SIZE is not None, 'REAL WORLD SIZE is not initialized'
    return _REAL_WORLD_SIZE

def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
   
    if check_initialized:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None
    ), 'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
  
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'data parallel group with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
        return _DATA_PARALLEL_GROUP


def get_data_parallel_group_gloo(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
        ), 'data parallel group-gloo with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, 'data parallel group-gloo is not initialized'
        return _DATA_PARALLEL_GROUP_GLOO


def get_context_parallel_group(check_initialized=True):
    """Get the context parallel group the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    if check_initialized:
        assert (
            _CONTEXT_PARALLEL_GLOBAL_RANKS is not None
        ), 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GLOBAL_RANKS


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, 'embedding group is not initialized'
    return _EMBEDDING_GROUP


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _POSITION_EMBEDDING_GROUP is not None, 'position embedding group is not initialized'
    return _POSITION_EMBEDDING_GROUP


def get_amax_reduction_group(with_context_parallel=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_tensor_and_data_parallel_group(with_context_parallel=False):
    """Get the tensor and data parallel group the caller rank belongs to."""

    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_tensor_and_expert_parallel_group():
    if _TENSOR_AND_EXPERT_PARALLEL_GROUP is None:
        print(f"rank: {torch.distributed.get_rank()} has no TE parallel group.", flush=True)
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is not None
    ), 'tensor and expert parallel group is not initialized'
    return _TENSOR_AND_EXPERT_PARALLEL_GROUP


def get_data_modulo_expert_parallel_group():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is not None
    ), 'data modulo expert parallel group is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())

# Add for Deepspeed ZeRO calling
def get_model_parallel_world_size():
    assert get_pipeline_model_parallel_world_size() == 1, "legacy get_model_parallel_world_size is only supported if PP is disabled"
    return get_tensor_model_parallel_world_size()

def get_model_parallel_rank():
    assert get_pipeline_model_parallel_world_size() == 1, "legacy get_model_parallel_rank is only supported if PP is disabled"
    return get_tensor_model_parallel_rank()

def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_split_rank(rank):
    """Set pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank(config=None):
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if config is not None and hasattr(config, "down_or_up"):
        if config.down_or_up == "up":
            mirror_rank = get_pipeline_model_parallel_world_size() - \
                torch.distributed.get_rank(group=get_pipeline_model_parallel_group()) - 1
            if mirror_rank >= config.bidirectional_pipeline_size:
                mirror_rank = None
            return mirror_rank
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_pipeline_first_stage(ignore_virtual=False, config=None):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank(config=config) == 0


def is_pipeline_last_stage(ignore_virtual=False, config=None):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = (
            get_virtual_pipeline_model_parallel_world_size()
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    if config is not None and hasattr(config, "down_or_up"):
        if config.down_or_up == "up":
            return get_pipeline_model_parallel_rank(config=config) == (config.bidirectional_pipeline_size - 1)
    return get_pipeline_model_parallel_rank(config=config) == (get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def get_pipeline_model_parallel_loss_group():
    assert _CONTRASTIVE_LOSS_GROUP is not None, "Contrastive loss group is not initialized"
    return _CONTRASTIVE_LOSS_GROUP

def get_pipeline_model_parallel_loss_rank():
    assert _CONTRASTIVE_LOSS_RANK is not None, "Contrastive loss rank is not initialized"
    return _CONTRASTIVE_LOSS_RANK


def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False

# not adapted for flexpipe
def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False

# not adapted for flexpipe
def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next
    stage executes encoder block for a model with both encoder and
    decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and is_pipeline_stage_after_split(rank + 1)


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_src_rank(rank=None):
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank() if rank is None else rank
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    else:
        assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank(config=None):
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    if config is not None and hasattr(config, "down_or_up"):
        if config.down_or_up == "up":
            return _PIPELINE_GLOBAL_RANKS[-1]
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank(config=None):
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    if config is not None and hasattr(config, "down_or_up"):
        if config.down_or_up == "up":
            return _PIPELINE_GLOBAL_RANKS[get_pipeline_model_parallel_world_size() - 
                                          config.bidirectional_pipeline_size]
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank(config=None):
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    if config is not None and hasattr(config, "down_or_up"):
        if config.down_or_up == "up":
            return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank(config=None):
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    if config is not None and hasattr(config, "down_or_up"):
        if config.down_or_up == "up":
            return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size(with_context_parallel=False):
    """Return world size for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)
        )
    else:
        return 0


def get_data_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)
        )
    else:
        return 0


def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_context_parallel_group())
    else:
        return 0


def get_context_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_context_parallel_group())
    else:
        return 0


def get_expert_model_parallel_world_size():
    """Return my rank for the expert parallel group"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_expert_model_parallel_rank():
    """Return my rank for the expert parallel group"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_data_modulo_expert_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_data_modulo_expert_parallel_group())
    else:
        return 0


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None

def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP_WITH_CP
    _DATA_PARALLEL_GROUP_WITH_CP = None
    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    _CONTEXT_PARALLEL_GLOBAL_RANKS = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    _TENSOR_AND_DATA_PARALLEL_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    _TENSOR_AND_EXPERT_PARALLEL_GROUP = None
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    _DATA_MODULO_EXPERT_PARALLEL_GROUP = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None

  