# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import time
import torch
from torch import Tensor
import torch.nn.functional as F
from functools import partial
# from typing import Union
from megatron import get_args
from megatron import print_rank_0, print_rank_all
from megatron import get_timers
# # from megatron import get_tokenizer
from megatron.core import tensor_parallel, mpu
from megatron.core.enums import ModelType
# from megatron.data.gpt_dataset import GPTDataset, build_train_valid_test_datasets
# import megatron.model
from megatron.core.parallel_state import is_extra_branch_rank
# from megatron.training import pretrain
from megatron.training_ds import pretrain
# from megatron.core.transformer.spec_utils import import_module
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

# deepspeed libraries
import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator
import subprocess
import json

# Get data

def model_provider(pre_process=True, post_process=True) -> CLIP_model.combined_CLIPModel:
    """Builds the model. """
    args = get_args()

    print_rank_0('building CLIP model ...')
    text_config = clip_text_transformer_config_from_args(args)
    vision_config = clip_vision_transformer_config_from_args(args)
    embed_dim = 1024
    # for deepspeed ZeRO-3 
    with deepspeed.zero.Init(
        data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             # config_dict_or_path=args.deepspeed_config,
                             config_dict_or_path='./ds_config.json',
                             enabled=args.zero_stage == 3,
                             mpu=mpu
    ):      
        model = CLIP_model.combined_CLIPModel(
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

    # Broadcast data. TP is not used
    combine_data = next(data_iterator)
    tokens = {}
    tokens['image'] = combine_data[0].to(dtype=torch.float32).to(torch.cuda.current_device())
    tokens['text'] = combine_data[1].to(dtype=torch.int32).to(torch.cuda.current_device())
    keys = ['image', 'text']

    eod_token = 49408 # begin: 49408 end: 49408 
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens['text'],
        eod_token,
        reset_attention_mask=True,
        reset_position_ids=True,
        eod_mask_loss=True,
        )

    return tokens

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out
    
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
        gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_result, result, group)
        return gathered_result

    if group is None:
        group = torch.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)
    torch.distributed.barrier(group=group)

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
    torch.distributed.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    
    return gathered_result

def loss_func(output_tensor_dict):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """    
    args = get_args()
    # gather特征，剔除冗余
    image_output = output_tensor_dict['image']
    text_output = output_tensor_dict['text']
    combined_image_output = gather_all_tensors(image_output)
    combined_text_output = gather_all_tensors(text_output)
    combined_image_output = torch.cat(combined_image_output, dim=0)
    combined_text_output = torch.cat(combined_text_output, dim=0)
    combined_image_output.requires_grad = True
    combined_text_output.requires_grad = True
    # print_rank_all(f"combined_image_output: {combined_image_output.shape} combined_text_output: {combined_text_output.shape}")
    # image_output.register_hook(lambda grad: print(f"gradient of input tensor: {grad}", flush=True))
    text_features = combined_text_output.contiguous()
    image_features = combined_image_output.contiguous()
    logits_per_image = image_features @ text_features.T
    logits_per_text = text_features @ image_features.T
    labels = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=torch.cuda.current_device())
    text_outputs = torch.argmax(logits_per_text, -1)
    correct = (text_outputs == labels).float()
    accuracy = torch.mean(correct)
    total_loss = (
            F.cross_entropy(logits_per_text, labels) +
            F.cross_entropy(logits_per_image, labels)
    ) / 2
    # print_rank_all(f"total_loss: {total_loss}, accuracy: {accuracy}")
    
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

    input_pairs = get_batch(data_iterator)
    # print_rank_all(f"input tokens: {tokens.shape} {tokens[:1,:10]}")
    timers('batch-generator').stop()
    output_tensor_dict = model(input_pairs)
    image_output = output_tensor_dict['image']
    text_output = output_tensor_dict['text']
    # print_rank_all(f"final-output: {output_tensor.requires_grad}")
    # print_rank_all(f"final-output- image:{image_output.shape} text:{text_output.shape}")
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
    args = get_args()
    img_transform = ClassificationTransform((256, 256))

    self_micro_batch_size = args.xmicro_batch_size if is_extra_branch_rank() else args.micro_batch_size
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


def command_exists(cmd):
    result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0

def git_ds_info():
    from deepspeed.env_report import main as ds_report
    ds_report()

    # Write out version/git info
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    print(f'**** Git info for Megatron: git_hash={git_hash} git_branch={git_branch} ****')

if __name__ == "__main__":
    git_ds_info()
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
