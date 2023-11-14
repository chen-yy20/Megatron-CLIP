# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Text63M model. 
A 63M-parameter 12layer 512-wide model with 8 attention heads. 
The transformer operates on a lower-cased byte pair encoding (BPE) representation 
of the text with a 49,152 vocab size"""

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import numpy as np
import einops

from megatron import get_args
from megatron.core import tensor_parallel
from megatron.model.enums import AttnMaskType
from megatron.model.language_model import parallel_lm_logits
from .module import MegatronModule
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal, Callable

# Vision Import
from megatron.model.transformer import ParallelTransformer
from megatron.model.vision.vit_backbone import VitBackbone, VitMlpHead, twod_interpolate_position_embeddings_hook
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

    def __init__(self, config, args, pre_process=True, post_process=True):
        super(VitBackbone, self).__init__()  
        args = get_args()
        self.output_tokens = args.vOutputTokens
        self.hidden_size = args.vHidden
        self.pre_process = pre_process
        self.post_process = post_process
        self.backbone = VitBackbone(
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
        return hidden_states
        

# Text Model

@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False

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


class CLIPTextModel(MegatronModule):

    def __init__(
        self,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        config: TransformerConfig = textConfig,
        pre_process: bool = True,
        post_process: bool = False,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,      
    ) -> None:
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type


        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.kv_channels, rotary_percent, seq_len_interpolation_factor
            )

        # Transformer
        self.decoder = TransformerBlock(
            config=self.config,
            transformer_layer_spec=self.transformer_layer_spec,
            self_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # 应该没有postprocess

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params=None,
    ) -> Tensor:
        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )

        # logits, _ = self.output_layer(hidden_states, weight=output_weight)
        return hidden_states

        # # logits and loss
        # output_weight = None
        # if self.share_embeddings_and_output_weights:
        #     output_weight = self.shared_embedding_or_output_weight()
        # logits, _ = self.output_layer(hidden_states, weight=output_weight)

        # if labels is None:
        #     # [s b h] => [b s h]
        #     return logits.transpose(0, 1).contiguous()

        # loss = self.compute_language_model_loss(labels, logits)

        # return loss

def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    act_layer = nn.GELU
    vision_heads = vision_cfg.width // vision_cfg.head_width
    # norm_layer要如何确定？
    # FIXME: NUM_CLASSES肯定不对，config的细节也要改
    visual = CLIPVisionModel(
        config=vision_cfg,
        num_classes=embed_dim,
        finetune=False,
        pre_process=True,
        post_process=False,
    )

    return visual

def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = nn.GELU
    text = CLIPTextModel(
        transformer_layer_spec,
        vocab_size=text_cfg.vocab_size,
        max_sequence_length=text_cfg.context_length,
        config=textConfig,
        pre_process=True,
        post_process=False,
    )

    return text

class CLIPModel(MegatronModule):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg)
        self.text = _build_text_tower(embed_dim, text_cfg)
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