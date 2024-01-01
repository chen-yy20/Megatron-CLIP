# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import numpy as np
import einops

from megatron import get_args
from megatron.core import tensor_parallel
from megatron.model.enums import AttnMaskType
from megatron.model.language_model import parallel_lm_logits, get_language_model
from .module import MegatronModule
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal, Callable
from megatron.core.enums import ModelType

# Vision Import
from megatron.model.transformer import ParallelTransformer
from megatron.model.vision.vit_backbone import VitBackbone, VitMlpHead, CLIP_VitBackbone, twod_interpolate_position_embeddings_hook
from megatron.model.utils import (
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)

# Transformer Import
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

# Vision Model

class CLIPVisionModel(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self, config, args, 
                 pre_process=True, 
                 post_process=True,
                 image_projection = True,
                 ):
        super().__init__(config=config, share_embeddings_and_output_weights=False)
        args = get_args()
        self.output_tokens = args.v_output_tokens
        self.hidden_size = args.v_hidden_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.model_type = ModelType.encoder_or_decoder
        self.image_projection = image_projection
        if self.post_process:
            if self.image_projection:
                scale = args.v_hidden_size ** -0.5
                self.projection = nn.Parameter(scale * torch.randn(config.hidden_size, args.clip_embeded_dim))

        self.backbone = CLIP_VitBackbone(
            config=config,
            pre_process=self.pre_process,
            post_process=self.post_process,
            single_token_output=self.output_tokens
        )

 
    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.backbone.set_input_tensor(input_tensor)

    def forward(self, input):
        # post process 都在 Vit-backbone里面完成了
        result = self.backbone(input) # len(hidden_states) = 2
        if isinstance(result, tuple):
            pooled, tokens = result
        else:
            pooled = result

        # def hook(gard):
        #     print(f"grad hook:[CLIPVisionModel.pooled]: {gard}", flush=True)
        #     return gard
        # pooled.register_hook(hook)
        return pooled
        

# Text Model



# textConfig = TransformerConfig(tensor_model_parallel_size=1,
#                                context_parallel_size=1,
#                                pipeline_model_parallel_size=1,
#                                sequence_parallel=False,
#                                expert_model_parallel_size=1,
#                                perform_initialization=True,
#                                use_cpu_initialization=False,
#                                fp16=False,
#                                bf16=False,
#                                params_dtype=torch.float32,
#                                gradient_accumulation_fusion=False,
#                                async_tensor_model_parallel_allreduce=False,
#                                num_layers=12,
#                                hidden_size=512,
#                                num_attention_heads=8,
#                                ffn_hidden_size=512,
#                                hidden_dropout=0.1,
#                                attention_dropout=0.1,
#                                fp32_residual_connection=False,
#                                ) 

"""
 Text63M model. 
A 63M-parameter 12layer 512-wide model with 8 attention heads. 
The transformer operates on a lower-cased byte pair encoding (BPE) representation 
of the text with a 49,152 vocab size
"""   
class CLIPTextModel(MegatronModule):

    def __init__(
        self,
        config,
        pre_process = True,
        post_process = True,
        add_pooler = True, # mean pooler
        text_projection = True,
    ):
        super().__init__(config=config, share_embeddings_and_output_weights=False)
        args = get_args()
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_pooler = add_pooler
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.text_projection = text_projection
        
        # CLIP 当中的positional_embedding 是 nn.Parameter(torch.empty(self.num_pos, width)) , 我们这里暂且使用RotaryEmbedding
        # 注意设置 args.position_embedding_type == 'rope'
        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=0, # num_tokentypes: size of the token-type embeddings. 0 value will ignore this embedding
            add_pooler=self.add_pooler,
            encoder_attn_mask_type=AttnMaskType.padding,
            pre_process=self.pre_process,
            post_process=self.post_process)
        
        # self.initialize_word_embeddings()

        if self.post_process and text_projection:
            self.projection = nn.Parameter(torch.empty(config.hidden_size, args.clip_embeded_dim))
            init_method_normal(args.init_method_std)(self.projection) 
       

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def CLIP_position_ids(self, token_ids):
    # Create position ids， it's the same as bert_position_ids
        seq_length = token_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

        return position_ids

    def forward(
        self,
        input_ids,
        # text,   # [batchsize, seq_length]  
        labels = None,
    ) -> Tensor:
        # get positional ids 
        position_ids = self.CLIP_position_ids(input_ids)
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        # 使用全局的attention mask
        # attention_mask = torch.ones(batch_size, seq_length)
        extended_attention_mask = torch.ones(batch_size, 1, seq_length, seq_length, dtype=torch.bool, device=torch.cuda.current_device())
        # Run decoder.
        lm_output = self.language_model( 
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=None,
        )

        if self.post_process:
            output, pooled_output = lm_output
            # FIXME: 是LanguageModel会完成post_process的工作吗？
            # output = output.permute(1, 0, 2) # [s, b, h] -> [b, s, h]
            # output = self.layernorm(output)
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # text = text[:,1:] # begin and end is both 49408
            # output = output[torch.arange(output.shape[0]), text.argmax(dim=-1)]
            if self.text_projection:
                lm_output = pooled_output @ self.projection
        return lm_output

        
    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
        if self.post_process:
            state_dict_[self._lm_head_key] \
                = self.lm_head.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)
        if self.post_process and self.add_binary_head:
            state_dict_[self._binary_head_key] \
                = self.binary_head.state_dict(prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix, keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if self.post_process:
            self.lm_head.load_state_dict(
                state_dict[self._lm_head_key], strict=strict)
        if self.post_process and self.add_binary_head:
            self.binary_head.load_state_dict(
                state_dict[self._binary_head_key], strict=strict)
        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)

# TODO: To be done
class CombinedCLIPModel(MegatronModule):
    def __init__(
            self,
            vision_cfg: TransformerConfig,
            text_cfg: TransformerConfig,
            pre_process=True,
            post_process=True,
    ):
        args = get_args()
         # pretrain function will get the config attribute for all models
        self.config = self.merge_config(text_cfg, vision_cfg)
        super().__init__(config=self.config, share_embeddings_and_output_weights=False)

        self.visual = CLIPVisionModel(
                        config=vision_cfg,
                        args=args,
                        pre_process=pre_process,
                        post_process=post_process,
                        image_projection=True,
                    )

        self.text = CLIPTextModel(
                        config = text_cfg,
                        pre_process=pre_process,
                        post_process=post_process,
                        add_pooler=True,
                        text_projection=True,
                    )
        self.pre_process = pre_process
        self.post_process = post_process
       

    def merge_config(self, text_cfg, vision_cfg, prefix_1='', prefix_2='v_'):
        result_dict = dict()
        text_dict = text_cfg.__dict__
        vision_dict = vision_cfg.__dict__
        merged_dict = {**text_dict, **vision_dict}
        for key, value in merged_dict.items():
            if key in text_dict and key in vision_dict:
                if text_dict[key] == vision_dict[key]:
                    result_dict[key] = value
                else:
                    result_dict[prefix_1+key] = text_dict[key]
                    result_dict[prefix_2+key] = vision_dict[key]
            else:
                result_dict[key] = value
        print(f"result_dict: {result_dict}", flush=True)

        def create_dataclass_from_dict(class_name, input_dict):
            # TODO: 有通信的优化空间？
            # 创建一个元类，用于动态生成类
            class DictDataclassMeta(type):
                def __new__(cls, name, bases, dct):
                    # 将字典的键转换为类的属性
                    for key in input_dict.keys():
                        dct[key] = input_dict[key]
                    return super().__new__(cls, name, bases, dct)

            # 使用元类创建类
            return DictDataclassMeta(class_name, (), {})
        
        merge_TransformerConfig = create_dataclass_from_dict('merge_TransformerConfig', result_dict)
        combined_config = merge_TransformerConfig()  
        return combined_config

    def set_input_tensor(self, input_tensor):
        args = get_args()
        if args.deepspeed or args.pure_dp:
            # no need to set without pipeline 
            self.input_tensor = input_tensor
        else:
            # If not the first stage, forward func will read input from here
            assert len(input_tensor) == 1, \
                    'input_tensor should only be length 1 for stage with both encoder and decoder'
            input_tensor = input_tensor[0]
            self.visual.set_input_tensor(input_tensor['image'])
            self.text.set_input_tensor(input_tensor['text'])
        

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        pass
        
    def forward(self, combine_input):
        image_tokens = combine_input['image']
        text_tokens = combine_input['text']
        # image_token: torch.float32, text_tokens: torch.int32
        image_features = self.visual(image_tokens)
        text_features = self.text(text_tokens, text_tokens)
        from megatron import print_rank_all
        # print_rank_all(f"image_features: {image_features.shape}, text_features: {text_features.shape}")
        return {'image':image_features, 'text':text_features}
