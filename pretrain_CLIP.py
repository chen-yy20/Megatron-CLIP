# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from torch import Tensor
import torch.nn.functional as F
from functools import partial
# from typing import Union
from megatron import get_args
from megatron import print_rank_0
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
    text_features = text_output.contiguous().float()
    image_features = image_output.contiguous().float()

    labels = torch.arange(text_output.shape[0], dtype=torch.long)
    text_logits = text_features @ image_features.T
    image_logits = image_features @ text_features.T
    text_outputs = torch.argmax(text_logits, -1)
    correct = (text_outputs == labels).float()
    accuracy = torch.mean(correct)
    total_loss = (
            F.cross_entropy(text_logits, labels) +
            F.cross_entropy(image_logits, labels)
    ) / 2
    averaged_loss = average_losses_across_data_parallel_group([total_loss, accuracy])
    return total_loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}



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

    return output_tensor, loss_func

class DatasetArguments:
    def __init__(self):
        self.train_data = "/mnt/zoltan/zanzong/CC3M/cc3m/{00000..00331}.tar"
        self.train_num_samples = 10000
        self.train_data_upsampling_factors = None
        self.seed = 1234
        self.batch_size = 16 # img: 16  text: torch.Size([16, 77])
        self.workers = 1
        self.world_size = 1
        self.model = 'RN50'

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    img_transform = ClassificationTransform((256, 256))

    dataset_args = DatasetArguments()
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
