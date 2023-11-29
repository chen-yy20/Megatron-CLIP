# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from torch import Tensor
# from functools import partial
# from typing import Union
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
# # from megatron import get_tokenizer
from megatron.core import tensor_parallel
# from megatron.core.enums import ModelType
# from megatron.data.gpt_dataset import GPTDataset, build_train_valid_test_datasets
# import megatron.model
from megatron.core.parallel_state import is_extra_branch_rank
# from megatron.training import pretrain
# from megatron.core.transformer.spec_utils import import_module
from megatron.utils import get_ltor_masks_and_position_ids
# from megatron.utils import average_losses_across_data_parallel_group
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
parser = argparse.ArgumentParser(description='Generate CLIP dataset batch')

dataset_args = parser.parse_args()
dataset_args.train_data = "/mnt/zoltan/zanzong/CC3M/cc3m/{00000..00331}.tar"
dataset_args.train_num_samples = 1000000
dataset_args.train_data_upsampling_factors = None
dataset_args.seed = 1234
dataset_args.batch_size = 16 # img: 16  text: torch.Size([16, 77])
dataset_args.workers = 1
dataset_args.world_size = 1
dataset_args.model = 'RN50'

img_transform = ClassificationTransform((256, 256))

data = {}
data['train'] = get_wds_dataset(dataset_args, img_transform, True, tokenizer=get_tokenizer(dataset_args.model))
print(data)
data['train'].set_epoch(0)
dataloader = data['train'].dataloader
data_iterator = iter(dataloader)

# ========================================================

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
            return_embeddings = True,
            text_projection = True,
        )

    return model


def get_batch(data_iterator):
    """Generate a batch."""
    # args = get_args()
    # Items and their type.
    datatype = torch.int64

    # Broadcast data.
    # FIXME: 此处的rank0可能有问题
    if data_iterator is not None:
        data = next(data_iterator)
        if is_extra_branch_rank():
            keys = ['text']
            data = data[1]
        else:
            keys = ['image']
            data = data[0]
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

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

def loss_func(text_output: Tensor, image_output: Tensor):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """    
    text_logits = text_output.contiguous().float()
    image_logits = image_output.contiguous().float()
#     losses = output_tensor.float()
#     loss_mask = loss_mask.view(-1).float()
#     loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

#     # Check individual rank losses are not NaN prior to DP all-reduce.
#     args = get_args()
#     if args.check_for_nan_in_loss_and_grad:
#         global_rank = torch.distributed.get_rank()
#         assert not loss.isnan(), (
#             f'Rank {global_rank}: found NaN in local forward loss calculation. '
#             f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
#         )

#     # Reduce loss for logging.
#     averaged_loss = average_losses_across_data_parallel_group([loss])

#     return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    if is_extra_branch_rank():
        tokens, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    else:
        tokens, _, _, _ = get_batch(data_iterator)

    timers('batch-generator').stop()
    if is_extra_branch_rank():
        # BERT lm
        output_tensor = model(tokens, position_ids, attention_mask)
    else:
        # Vit Backbone
        output_tensor = model(tokens)

    return output_tensor


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    # args = get_args()

    # print_rank_0('> building train, validation, and test datasets '
    #              'for GPT ...')
    # train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
    #     data_prefix=args.data_path,
    #     splits_string=args.split,
    #     train_valid_test_num_samples=train_val_test_num_samples,
    #     seq_length=args.seq_length,
    #     seed=args.seed,
    #     skip_warmup=(not args.mmap_warmup),
    #     train_data_prefix=args.train_data_path,
    #     valid_data_prefix=args.valid_data_path,
    #     test_data_prefix=args.test_data_path,
    #     data_cache_path=args.data_cache_path)
    # print_rank_0("> finished creating GPT datasets ...")

    # return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
