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


# Shall we use this layernorm?
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

# Vision Model

class CLIPVisionModel(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self, config, args, 
                 pre_process=True, 
                 post_process=True,
                 image_projection = False,
                 ):
        super(VitBackbone, self).__init__()  
        args = get_args()
        self.output_tokens = args.v_output_tokens
        self.hidden_size = args.v_hidden_size
        self.pre_process = pre_process
        self.post_process = post_process
        if image_projection:
            scale = args.v_hidden_size ** -0.5
            self.image_projection = nn.Parameter(scale * torch.randn(config.hidden_size, args.clip_embedded_dim))
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
        hidden_states = self.backbone(input)
        if self.post_process:
            hidden_states = hidden_states.permute(1, 0, 2)
            hidden_states = LayerNorm(hidden_states)
            if self.image_projection:
                hidden_states = hidden_states @ self.image_projection
        return hidden_states
        

# Text Model



textConfig = TransformerConfig(tensor_model_parallel_size=1,
                               context_parallel_size=1,
                               pipeline_model_parallel_size=1,
                               sequence_parallel=True,
                               expert_model_parallel_size=1,
                               perform_initialization=True,
                               use_cpu_initialization=False,
                               fp16=False,
                               bf16=False,
                               params_dtype=torch.float32,
                               gradient_accumulation_fusion=False,
                               async_tensor_model_parallel_allreduce=False,
                               num_layers=12,
                               hidden_size=512,
                               num_attention_heads=8,
                               ffn_hidden_size=512,
                               hidden_dropout=0.1,
                               attention_dropout=0.1,
                               fp32_residual_connection=False,
                               ) 
def CLIP_position_ids(token_ids):
    # Create position ids， it's the same as bert_position_ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids

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
        add_pooler = False,
        return_embeddings = True,
        text_projection = False,
    ):
        super().__init__(config=config)
        args = get_args()
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_pooler = add_pooler
        self.return_embeddings = return_embeddings
        if self.return_embeddings:
            assert self.post_process and self.add_pooler

        
        # CLIP 当中的positional_embedding 是 nn.Parameter(torch.empty(self.num_pos, width)) , 我们这里暂且使用RotaryEmbedding
        # 注意设置 args.position_embedding_type == 'rope'
        self.language_model, self._language_model_key = get_language_model(
            config=config,
            add_pooler=self.add_pooler,
            encoder_attn_mask_type= AttnMaskType.padding,
            pre_process=self.pre_process,
            post_process=self.post_process)
        
        self.initialize_word_embeddings()

        if self.post_process and text_projection:
            self.text_projection = nn.Parameter(torch.empty(config.hidden_size, args.clip_embedded_dim))
       

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)


    def forward(
        self,
        input_ids,
        labels = None,
    ) -> Tensor:

        # get positional ids 
        position_ids = CLIP_position_ids(input_ids)
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        # 使用全局的attention mask
        attention_mask = torch.ones(batch_size, seq_length)
        extended_attention_mask = torch.ones(batch_size, 1, seq_length, seq_length, dtype=torch.bool, device=torch.cuda.current_device())
        
        # Run decoder.
        lm_ouput = self.language_model( 
            input_ids,
            position_ids,
            extended_attention_mask,
            token_type_ids=None, # FIXME: token_type_ids， while it can be None
        )

        if self.post_process and self.add_pooler:
            lm_ouput, pooled_output = lm_ouput

            if self.return_embeddings:
                embeddings = torch.transpose(lm_ouput, 0, 1)
                masks = torch.sum(attention_mask, dim=-1)

                output = torch.zeros(
                    size = (embeddings.shape[0], embeddings.shape[2]),
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                )
                for i, (embdding, mask) in enumerate(zip(embeddings, masks)):
                    output[i, :] = torch.mean(embdding[1:mask-1], dim=0)

                return output


        # logits, _ = self.output_layer(hidden_states, weight=output_weight)
        else:
            pooled_output = None

        if self.post_process:
            lm_output = lm_output.permute(1, 0, 2) # [b s h] => [s b h]
            output = LayerNorm(lm_output)
            output = output @ self.text_projection
            return output
        else:
            return lm_ouput

        
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


class CLIPModel(MegatronModule):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            vision_cfg: TransformerConfig,
            text_cfg: TransformerConfig,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = CLIPVisionModel(
                        config=vision_cfg,
                        args=get_args(),
                        pre_process=True,
                        post_process=True,
                        image_projection= True,
                    )

        self.text = CLIPTextModel(
                        config = text_cfg,
                        pre_process=True,
                        post_process=True,
                        add_pooler=False,
                        return_embeddings=False, # 暂时还不知道这个参数的必要性
                        text_projection=True,
                    )
        self.token_embedding = self.text.embedding # FIXME:
        self.transformer = self.text.decoder
        self.vocab_size = self.text.vocab_size
        self.positional_embedding = self.text.rotary_pos_emb
        # FIXME:
        self.ln_final = self.text.decoder.ln_final
        # self.text_projection = text.text_projection
        # self.register_buffer('attn_mask', text.attn_mask, persistent=False)
        # 计算相似度
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        pass

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features
    
    # 为什么这个要提出来算
    def encode_text(self, text, normalize: bool = False):
        x = self.token_embedding(text)
        x = x + self.positional_embedding(text)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] # WHAT?
        # features = self.text(text)
        return F.normalize(x, dim=-1) if normalize else x
        
    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
        return image_features, text_features, self.logit_scale.exp()