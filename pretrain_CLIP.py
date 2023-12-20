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
import argparse
from open_CLIP.src.open_clip.factory import get_tokenizer
from open_CLIP.src.training.data import get_wds_dataset
from megatron.data.vit_dataset import ClassificationTransform


# Get data

def model_provider(pre_process=True, post_process=True):
    """Builds the model. """
    args = get_args()

    print_rank_0('building CLIP model ...')
    text_config = clip_text_transformer_config_from_args(args)
    vision_config = clip_vision_transformer_config_from_args(args)

    # 目前先作为两个完全独立的模型来训练
    if not is_extra_branch_rank():
        model = CLIP_model.CLIPVisionModel(
            config=vision_config,
            args=args,
            pre_process=pre_process,
            post_process=post_process,
            image_projection=True,
        ).to("cuda")
    else:
        model = CLIP_model.CLIPTextModel(
            config = text_config,
            pre_process = pre_process,
            post_process = post_process,
            add_pooler = True,
            text_projection = True,
        ).to("cuda")

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # Broadcast data.
    if data_iterator is not None:
        combine_data = next(data_iterator)
        data = {}
        if is_extra_branch_rank():
            keys = ['text']
            data['text'] = combine_data[1].to(dtype=torch.int32) # 将数据转换为相同目标类型
        else:
            keys = ['image']
            data['image'] = combine_data[0].to(dtype=torch.float32)
        # keys = ['image', 'text']
        # data['image'] = combine_data[0]
        # data['text'] = combine_data[1].to(dtype = torch.float16)
        # print_rank_0(f"Data type: {data['image'].dtype} {data['text'].dtype}")
        
    else:
        # 非TP的src进程不会得到数据
        keys = ['text'] if is_extra_branch_rank() else ['image']
        data = None
    datatype = torch.int32 if is_extra_branch_rank() else torch.float32
    # print_rank_all(f"keys {keys} iterator {data_iterator} -> begin broadcast_data()")
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    # print_rank_all(f"keys {keys} data_b {data_b.keys()}")
    # Unpack.
    if is_extra_branch_rank():
        tokens = data_b['text'].contiguous()
    else:
        tokens = data_b['image'].contiguous()

    # # Get the masks and postition ids.
    # args.reset_position_ids,
    #         args.reset_attention_mask,
    #         args.eod_mask_loss
    if is_extra_branch_rank():
        eod_token = 49408 # begin: 49408 end: 49408 
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            eod_token,
            reset_attention_mask=True,
            reset_position_ids=True,
            eod_mask_loss=True,
            )

        return tokens, loss_mask, attention_mask, position_ids
    else:
        return tokens, None, None, None

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
        return tuple(out_tensor_list)

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
    
    def _simple_gather_all_tensors(result: Tensor, world_size: int, group=None):
        gathered_result = GroupAllGather.apply(group, result)
        return gathered_result

    if group is None:
        group = torch.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, world_size, group)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, world_size, group)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    dist.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    group_rank = dist.get_rank(group=group)
    gathered_result[group_rank] = result
    
    return gathered_result

def loss_func(output):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """    
    args = get_args()
    # gather特征，剔除冗余
    loss_ranks = parallel_state.get_pipeline_model_parallel_loss_rank()
    combine_output = gather_all_tensors(output, parallel_state.get_pipeline_model_parallel_loss_group())
    # print_rank_all(f"gathered tensor={[t.shape for t in combine_output]}", False)

    image_world_size = args.world_size
    text_world_size = args.extra_world_size
    image_loss_ranks = [r for r in loss_ranks if r < image_world_size]
    text_loss_ranks = [r for r in loss_ranks if r >= image_world_size]
    image_dp_size = len(image_loss_ranks) // args.tensor_model_parallel_size
    text_dp_size = len(text_loss_ranks) // args.xtensor_model_parallel_size
    image_output = [combine_output[i] for i in range(image_dp_size)]
    text_output = [combine_output[i + len(image_loss_ranks)] for i in range(text_dp_size)]
    # print_rank_all(f"selected image output:{list(range(image_dp_size))}", False)
    # print_rank_all(f"selected text output:{list(range(text_dp_size))}", False)
    
    # 连接特征
    image_output = torch.cat(image_output, dim=0)
    text_output = torch.cat(text_output, dim=0)

    # print_rank_0(f"-LOSS-  image_output: {image_output.dtype}  text_output: {text_output.dtype}")
    # image_output.register_hook(lambda grad: print(f"gradient of image out tensor: {grad}", flush=True))
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
    timers('batch-generator', log_level=2).start()
    # print_rank_all("begin get_batch()")

    tokens, loss_mask, attention_mask, position_ids = get_batch(data_iterator)

    # print_rank_all(f"input tokens: {tokens.shape} {tokens[:1,:10]}")
    timers('batch-generator').stop()
    if is_extra_branch_rank():
        # BERT lm
        output_tensor = model(tokens,tokens) # 此处一个是stage间输入，一个是text原始输入
    else:
        # Vit Backbone
        output_tensor = model(tokens)
    # print_rank_all(f"Call forward_step, output shape={output_tensor.shape}, dtype={output_tensor.dtype}", False)
    # def hook(gard):
    #     print(f"grad hook:[forward_step finish]: {gard}", flush=True)
    #     return gard
    # output_tensor.register_hook(hook)
    # print_rank_all(f"final-output: {output_tensor.requires_grad}")
    # print(f"call forward_step final-output: {output_tensor.shape}", flush=True)
    return output_tensor, loss_func


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
    args = get_args()
    img_transform = ClassificationTransform((256, 256))
    self_micro_batch_size = args.xmicro_batch_size if is_extra_branch_rank() else args.micro_batch_size
    # self_dp = args.xdata_parallel_size if is_extra_branch_rank() else args.data_parallel_size
    # mbs_cal = mbs_calculator(args.global_batch_size, self_micro_batch_size, self_dp)
    # if is_extra_branch_rank():
    #     args.xmicro_batch_size = mbs_cal.get()
    #     bs = args.xmicro_batch_size
    # else:
    #     args.micro_batch_size = mbs_cal.get()
    #     bs = args.micro_batch_size
    
    dataset_args = DatasetArguments(self_micro_batch_size)
    data = {}
    data['train'] = get_wds_dataset(dataset_args, img_transform, True, tokenizer=get_tokenizer(dataset_args.model))
    train_ds = data['train']
    
    # data['train'].set_epoch(0)
    # train_ds = data['train'].dataloader

    print_rank_0('> building train, validation, and test datasets '
                 'for CLIP ...')
    print_rank_0("> finished creating CLIP datasets ...")

    return train_ds, None, None

if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
