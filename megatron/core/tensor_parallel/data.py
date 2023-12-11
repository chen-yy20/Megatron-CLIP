# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.distributed

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
)

_MAX_DATA_DIM = 5


def _check_data_types(keys, data, target_dtype):
    """Check that all the keys have the same target data type."""
    for key in keys:
        assert data[key].dtype == target_dtype, (
            '{} has data type {} which '
            'is different than {}'.format(key, data[key].dtype, target_dtype)
        )


def _build_key_size_numel_dictionaries(keys, data):
    """Build the size on rank 0 and broadcast."""
    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]

    # Pack the sizes on rank zero.
    if get_tensor_model_parallel_rank() == 0:
        offset = 0
        for key in keys:
            # print(f"Rank {rank} key {key} data.shape {data.shape}", flush=True)
            assert data[key].dim() < max_dim, 'you should increase MAX_DATA_DIM'
            size = data[key].size()
            for i, s in enumerate(size):
                sizes[i + offset] = s
            offset += max_dim
    # Move to GPU and broadcast.
    sizes_cuda = torch.cuda.LongTensor(sizes)
    # rank = torch.distributed.get_rank()
    # print(f"Rank {rank} sizes_cuda {sizes_cuda} => begin broadcast, group: {get_tensor_model_parallel_group()}", flush=True)
    torch.distributed.broadcast(
        sizes_cuda, get_tensor_model_parallel_src_rank(), group=get_tensor_model_parallel_group()
    )
    # print(f"Rank {rank} sizes_cuda {sizes_cuda} => finish broadcast", flush=True)

    # Move back to cpu and unpack.
    sizes_cpu = sizes_cuda.cpu()
    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel


def broadcast_data(keys, data, datatype):
    """Broadcast data from rank zero of each model parallel group to the
    members of the same model parallel group.

    Arguments:
        keys: list of keys in the data disctionary to be broadcasted
        data: data dictionary of string keys and cpu tensor values.
        datatype: torch data type of all tensors in data associated
                  with keys.
    """
    rank = torch.distributed.get_rank()
    # print(f"Rank {rank} keys {keys} => begin broadcast_data()", flush=True)
    # Build (key, size) and (key, number of elements) dictionaries along
    # with the total number of elements on all ranks.
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(keys, data)

    # Pack on rank zero.
    if get_tensor_model_parallel_rank() == 0:
        # Check that all keys have the same data type.
        _check_data_types(keys, data, datatype)
        # Flatten the data associated with the keys
        """
        对于给定的keys列表，它通过列表解析创建一个张量列表，其中每个张量都是data字典中相应键对应的值。
        .contiguous()的作用是确保张量在内存中是连续存储的，因为在一些情况下，张量可能不是连续存储的
        （例如，通过切片或其他操作导致不连续的存储）。
        接着，通过.view(-1)将每个张量展平成一个一维张量。
        最后，通过torch.cat函数按指定的维度（dim=0表示沿着行的方向）将这些一维张量连接成一个大的一维张量。   
        最终，.cuda()将这个大张量移动到GPU上。
        """
        flatten_data = torch.cat([data[key].contiguous().view(-1) for key in keys], dim=0).cuda()
    else:
        """
        对于非rank 0的情况，直接创建一个空的张量flatten_data，并设置其大小为total_numel，即所有元素的总数，
        设备为当前GPU设备，数据类型为datatype。
        """
        flatten_data = torch.empty(total_numel, device=torch.cuda.current_device(), dtype=datatype)

    # Broadcast, 在此处通过get_tensor_model_parallel_src_rank指定了广播的源rank
    torch.distributed.broadcast(
        flatten_data, get_tensor_model_parallel_src_rank(), group=get_tensor_model_parallel_group()
    )

    # Unpack
    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output
