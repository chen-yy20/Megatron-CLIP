from megatron.core.parallel_state import initialize_model_parallel
from megatron.initialize import initialize_megatron
import torch
import torch.distributed
import os

rank = int(os.getenv('RANK', '0'))
world_size = int(os.getenv("WORLD_SIZE", '1'))
# print(f'rank: {rank}, world_size: {world_size}')

torch.distributed.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        )
torch.distributed.barrier()

# initialize_model_parallel(
#     tensor_model_parallel_size=2,
#     pipeline_model_parallel_size=2,
# )
initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=4,
    extra_world_size= 8,
    xtensor_model_parallel_size=1,
    xpipeline_model_parallel_size=2,
    is_multi_branch=True
)

# step 1 : initialize megatron
# initialize_megatron()

