# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import time
import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import _Reduce_Scatter, ReduceOp, _AlltoAll
from functools import partial
# from typing import Union
from megatron import get_args
from megatron import print_rank_0, print_rank_all
from megatron import get_timers
# # from megatron import get_tokenizer
from megatron.core import parallel_state, tensor_parallel
from megatron.core.parallel_state import is_extra_branch_rank
from megatron.core.enums import ModelType
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args, \
                            clip_vision_transformer_config_from_args, clip_text_transformer_config_from_args
# # from megatron.core.models.gpt.gpt_layer_specs import (
# #     gpt_layer_with_transformer_engine_spec,
# #     gpt_layer_with_transformer_engine_spec_moe
# # )
from megatron.model import CLIP_model


# load dataset
from open_CLIP.src.open_clip.factory import get_tokenizer
from open_CLIP.src.training.data import get_wds_dataset
from megatron.data.vit_dataset import ClassificationTransform
from deepspeed.runtime.utils import see_memory_usage

# Get data

def model_provider(pre_process=True, post_process=True):
    """Builds the model. """
    args = get_args()

    print_rank_0('building CLIP model ...')
    text_config = clip_text_transformer_config_from_args(args)
    vision_config = clip_vision_transformer_config_from_args(args)

    model = CLIP_model.CombinedCLIPModel(
        vision_cfg=vision_config,
        text_cfg=text_config,
        pre_process=pre_process,
        post_process=post_process,
    )
    model.to(torch.cuda.current_device())
    see_memory_usage(f"after building CLIP model", force=True)
    return model


def get_batch(data_iterator):
    """Generate a batch."""
    data = {}
    tokens = {}
    # Broadcast data.
    if data_iterator is not None:
        combine_data = next(data_iterator)
        keys = ['image']
        data['image'] = combine_data[0].to(dtype=torch.float32)
        datatype = torch.float32
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)
        image_tokens = data_b['image'].contiguous()
        
        keys = ['text']
        data['text'] = combine_data[1].to(dtype=torch.int32) # 将数据转换为相同目标类型7
        datatype = torch.int32
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)
        text_tokens = data_b['text'].contiguous()
        tokens['image'] = image_tokens
        tokens['text'] = text_tokens
    else:
        keys = ['image']
        datatype = torch.float32
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)
        image_tokens = data_b['image'].contiguous()
        keys = ['text']
        datatype = torch.int32
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)
        text_tokens = data_b['text'].contiguous()
        tokens['image'] = image_tokens
        tokens['text'] = text_tokens
    return tokens

class GroupAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, tensor):
        # Need contiguous tensors for collectives.
        tensor = tensor.contiguous()
        ctx.group = group
        out_tensor_list = [
            torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))
        ]
        dist.all_gather(out_tensor_list, tensor, group=group)
        # Only tensor generated in local process has grad function,
        # so replace the gathered one to enable backward.
        self_rank = dist.get_rank(group=group)
        out_tensor_list[self_rank] = tensor
        return out_tensor_list

    @staticmethod
    def backward(ctx, *grad_outputs):
        if dist.get_backend(group=ctx.group) is dist.Backend.NCCL:
            # Different from original AllGather, we get rank for given group
            rank = dist.get_rank(ctx.group)
            gx = torch.empty_like(grad_outputs[rank])
            _Reduce_Scatter.apply(ReduceOp.SUM, ctx.group, gx, *grad_outputs)
        else:
            # As many backends doesn't support ReduceScatter, we use AlltoAll with .sum()
            # to emulate the ReduceScatter behavior
            tensor_list = [torch.empty_like(tensor) for tensor in grad_outputs]
            gxs = _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)
            gx = torch.sum(torch.stack(gxs), dim=0)
        return (None, gx)
    
def gather_all_tensors(result: Tensor, group=None):
    """
    Function to gather all tensors from several ddp processes onto a list that
    is broadcasted to all processes. Works on tensors that have the same number
    of dimensions, but where each dimension may differ. In this case tensors are
    padded, gathered and then trimmed to secure equal workload for all processes

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    
    def _simple_gather_all_tensors(result: Tensor, group=None):
        gathered_result = GroupAllGather.apply(group, result)
        return gathered_result

    if group is None:
        group = torch.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    
    # gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    # dist.all_gather(gathered_result, result_padded, group)
    gathered_result = _simple_gather_all_tensors(result_padded, group=group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    # group_rank = dist.get_rank(group=group)
    # gathered_result[group_rank] = result
    
    return gathered_result

def loss_func(output):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """    
    args = get_args()
    # print_rank_all(f"loss_func get output:{output.shape}, requires_grad={output.requires_grad}", False)
    vision_output = output['image']
    text_output = output['text']
    combine_vision_output = gather_all_tensors(vision_output, parallel_state.get_pipeline_model_parallel_loss_group())
    combine_text_output = gather_all_tensors(text_output, parallel_state.get_pipeline_model_parallel_loss_group())
    # print_rank_all(f"gathered tensor={[t.shape for t in combine_output]}", False)
    # print_rank_all(f"gathered tensor req. gradient={[t.requires_grad for t in combine_output]}", False)

    local_dp_src_rank = dist.get_rank() - (dist.get_rank() // args.tensor_model_parallel_size) * args.tensor_model_parallel_size
    image_output = [combine_vision_output[i] for i in range(local_dp_src_rank, len(combine_vision_output), args.tensor_model_parallel_size)]
    text_output = [combine_text_output[i] for i in range(local_dp_src_rank, len(combine_text_output), args.tensor_model_parallel_size)] 
    # print_rank_all(f"loss dp src local rank={local_dp_src_rank}, image len={len(image_output)}, text len={len(text_output)}", False)

    # 连接特征
    image_output = torch.cat(image_output, dim=0)
    text_output = torch.cat(text_output, dim=0)
    assert any([image_output.requires_grad, text_output.requires_grad]), \
        "At least one modal's output shoud has gradients. Check gather_all_tensors() function."
    text_features = text_output.contiguous()
    image_features = image_output.contiguous()
    logits_per_image = image_features @ text_features.T
    logits_per_text = text_features @ image_features.T
    labels = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=torch.cuda.current_device())
    # text_outputs = torch.argmax(logits_per_text, -1)
    # correct = (text_outputs == labels).float()
    # accuracy = torch.mean(correct)
    total_loss = (
            F.cross_entropy(logits_per_text, labels) +
            F.cross_entropy(logits_per_image, labels)
    ) / 2
    # print_rank_all(f"total_loss: {total_loss}", False)
    
    # averaged_loss = average_losses_across_data_parallel_group([total_loss, accuracy]) 
    
    return total_loss, {"loss": total_loss, "accuracy": 0.0}



def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()
    rank = torch.distributed.get_rank()
    # Get the batch.
    # timers('batch-generator', log_level=2).start()
    # print_rank_all("begin get_batch()")
    
    input_pairs = get_batch(data_iterator)
    output_tensor_dict = model(input_pairs)
    return output_tensor_dict, loss_func


class DatasetArguments:
    def __init__(self, batch_size):
        self.train_data = "/mnt/zoltan/zanzong/CC3M/cc3m/{00000..00331}.tar"
        self.train_num_samples = 10000
        self.train_data_upsampling_factors = None
        self.seed = 1234
        self.batch_size = batch_size # img: 16  text: torch.Size([16, 77])
        self.workers = 1
        self.world_size = 1
        self.model = 'RN50'

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    print_rank_0('> building train, validation, and test datasets '
                'for CLIP ...')
    args = get_args()
    img_transform = ClassificationTransform((256, 256))

    self_micro_batch_size = args.xmicro_batch_size if is_extra_branch_rank() else args.micro_batch_size
    dataset_args = DatasetArguments(self_micro_batch_size)
    data = {}
    data['train'] = get_wds_dataset(dataset_args, img_transform, True, tokenizer=get_tokenizer(dataset_args.model))
    train_ds = data['train']
    print_rank_0("> finished creating CLIP datasets ...")

    return train_ds, None, None

if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
