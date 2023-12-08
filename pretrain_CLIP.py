# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from torch import Tensor
import torch.nn.functional as F
from functools import partial
# from typing import Union
from megatron import get_args
from megatron import print_rank_0, print_rank_all
from megatron import get_timers
# # from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
# from megatron.data.gpt_dataset import GPTDataset, build_train_valid_test_datasets
# import megatron.model
from megatron.core.parallel_state import is_extra_branch_rank
from megatron.training import pretrain
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


# Get data

def model_provider(pre_process=True, post_process=True) -> CLIP_model.CLIPModel:
    """Builds the model. """
    args = get_args()

    print_rank_0('building CLIP model ...')
    text_config = clip_text_transformer_config_from_args(args)
    vision_config = clip_vision_transformer_config_from_args(args)
    embed_dim = 512

    # 目前先作为两个完全独立的模型来训练
    if not is_extra_branch_rank():
        model = CLIP_model.CLIPVisionModel(
            config=vision_config,
            args=args,
            pre_process=pre_process,
            post_process=post_process,
            image_projection=True,
        )
    else:
        model = CLIP_model.CLIPTextModel(
            config = text_config,
            pre_process = pre_process,
            post_process = post_process,
            add_pooler = True,
            return_embeddings = False,
            text_projection = True,
        )

    return model


def get_batch(data_iterator):
    """Generate a batch."""
    # args = get_args()
    # Items and their type.
    datatype = torch.float16
    # Broadcast data.
    # FIXME: 此处的rank0可能有问题
    if data_iterator is not None:
        combine_data = next(data_iterator)
        data = {}
        if is_extra_branch_rank():
            keys = ['text']
            data['text'] = combine_data[1].to(dtype = torch.float16) # 将数据转换为相同目标类型
        else:
            keys = ['image']
            data['image'] = combine_data[0]
        # keys = ['image', 'text']
        # data['image'] = combine_data[0]
        # data['text'] = combine_data[1].to(dtype = torch.float16)
        # print_rank_0(f"Data type: {data['image'].dtype} {data['text'].dtype}")
        
    else:
        # 非TP的src进程不会得到数据
        keys = ['text'] if is_extra_branch_rank() else ['image']
        data = None
    print_rank_all(f"keys {keys} iterator {data_iterator} -> begin broadcast_data()")
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    print_rank_all(f"keys {keys} data_b {data_b.keys()}")
    # Unpack.
    if is_extra_branch_rank():
        tokens = data_b['text'].long().contiguous()
    else:
        tokens = data_b['image'].long().contiguous()

    # # Get the masks and postition ids.
    # args.reset_position_ids,
    #         args.reset_attention_mask,
    #         args.eod_mask_loss
    if is_extra_branch_rank():
        eod_token = 49407 # begin: 49406 end: 49407 
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

def loss_func(output):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """    
    args = get_args()
    # gather特征，剔除冗余
    world_size = torch.distributed.get_world_size()

    # QUESTION: 据说以下做法是可行的
    # combine_output = [torch.zeros(args.micro_batch_size, args.clip_embeded_dim, device=torch.cuda.current_device()) 
    #                   for i in range(world_size)]
    # torch.distributed.all_gather(combine_output, output)
    # combine_output[torch.distributed.get_rank()] = output
    combine_output = GatherLayer.apply(output)
    print_rank_all(f"combine_output: {combine_output[0].shape} {combine_output[0].requires_grad}")
    

    image_world_size = world_size - args.extra_world_size
    image_tp_group_num = image_world_size // args.tensor_model_parallel_size
    text_world_size = args.extra_world_size
    text_tp_group_num = text_world_size // args.xtensor_model_parallel_size
    
    image_output = [combine_output[i * args.tensor_model_parallel_size] for i in range(image_tp_group_num)]
    text_output = [combine_output[i * args.xtensor_model_parallel_size + image_world_size] for i in range(text_tp_group_num)]

    # 连接特征
    image_output = torch.cat(image_output, dim=0)
    text_output = torch.cat(text_output, dim=0)

    print_rank_0(f"-LOSS-  image_output: {image_output.requires_grad}  text_output: {text_output.requires_grad}")
    
    text_features = text_output.contiguous().float()
    image_features = image_output.contiguous().float()

    labels = torch.arange(text_output.shape[0], dtype=torch.long, device=torch.cuda.current_device())
    text_logits = text_features @ image_features.T
    image_logits = image_features @ text_features.T
    text_outputs = torch.argmax(text_logits, -1)
    correct = (text_outputs == labels).float()
    accuracy = torch.mean(correct)
    total_loss = (
            F.cross_entropy(text_logits, labels) +
            F.cross_entropy(image_logits, labels)
    ) / 2
    print_rank_all(f"total_loss: {total_loss} accuracy: {accuracy}")
    # averaged_loss = average_losses_across_data_parallel_group([total_loss, accuracy]) 
    
    return total_loss, {"loss": total_loss, "accuracy": accuracy}



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
    print_rank_all("begin get_batch()")

    tokens, loss_mask, attention_mask, position_ids = get_batch(data_iterator)

    # print_rank_all(f"input tokens: {tokens.shape} {tokens[:1,:10]}")
    print_rank_all(f"input tokens: {tokens.shape}")
    timers('batch-generator').stop()
    if is_extra_branch_rank():
        # BERT lm
        output_tensor = model(tokens,tokens) # 此处一个是stage间输入，一个是text原始输入
    else:
        # Vit Backbone
        output_tensor = model(tokens)
    print_rank_all(f"final-output: {output_tensor.requires_grad}")
    # print_rank_all(f"final-output: {output_tensor[:1,:10]}")
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

    dataset_args = DatasetArguments(args.micro_batch_size)
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
