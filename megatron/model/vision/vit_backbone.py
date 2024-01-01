# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Vision Transformer(VIT) model."""

import math
import einops
import torch
import apex
import torch.nn.functional as F
from megatron import get_args
from megatron.model.transformer import ParallelTransformer
from megatron.model.utils import (
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
    get_norm
)
from megatron.model.module import MegatronModule

CLASS_TOKEN_LENGTH = 8

class VitMlpHead(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, config, hidden_size, num_classes):
        super(VitMlpHead, self).__init__()
        self.config = config
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dense_out = torch.nn.Linear(hidden_size, num_classes)
        torch.nn.init.constant_(self.dense_out.bias, -10)

    def forward(self, hidden_states):
        # hidden_states: [b, 1, h]
        # sequence_index: index of the token to pool.
        dense_in_result = self.dense_in(hidden_states)
        tanh_result = torch.tanh(dense_in_result)
        dense_out_result = self.dense_out(tanh_result)
        return dense_out_result


def isPerfectSquare(x):
    if(x >= 0):
        sr = math.sqrt(x)
        return (int(sr) * int(sr) == x)
    return False


def twod_interpolate_position_embeddings_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):

    args = get_args()
    num_patches_per_dim_h = args.img_h // args.patch_dim
    num_patches_per_dim_w = args.img_w // args.patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    hidden_size = args.hidden_size

    key = prefix + "weight"

    assert key in state_dict
    if key in state_dict:
        input_param = state_dict[key]

        input_seq_len = input_param.shape[0]
        assert(isPerfectSquare(input_seq_len) or isPerfectSquare(input_seq_len - CLASS_TOKEN_LENGTH))
        input_has_class_token = not isPerfectSquare(input_seq_len)
        num_tok_input = input_seq_len - CLASS_TOKEN_LENGTH if input_has_class_token else input_seq_len
        num_tok_output = num_patches
        output_has_class_token = args.class_token_present

        # update input_param and load it to state_dict[key]
        if input_has_class_token:
            input_param_tok = input_param[:CLASS_TOKEN_LENGTH, :]
            input_param_grid = input_param[CLASS_TOKEN_LENGTH:, :]
        else:
            input_param_tok = torch.zeros(CLASS_TOKEN_LENGTH, hidden_size)
            input_param_grid = input_param

        assert input_param.shape[1] == hidden_size

        if num_tok_input != num_tok_output:

            gs_input = int(math.sqrt(num_tok_input))
            gs_new = (num_patches_per_dim_h, num_patches_per_dim_w)

            input_param_grid = input_param_grid.transpose(0, 1).contiguous()
            input_param_grid = input_param_grid.reshape(
                (1, -1, gs_input, gs_input)
            )
            input_param_grid = input_param_grid.float()
            scale_factor = (gs_new[0] / gs_input, gs_new[1] / gs_input)

            input_param_grid = F.interpolate(
                input_param_grid, scale_factor=scale_factor, mode="bilinear"
            )

            input_param_grid = input_param_grid.half()
            input_param_grid = input_param_grid.reshape((-1, num_tok_output))
            input_param_grid = input_param_grid.transpose(0, 1).contiguous()

            assert input_param_grid.shape[1] == hidden_size

        input_param = input_param_grid
        assert (
            input_param.shape[0] == num_tok_output
            and input_param.shape[1] == hidden_size
        )

        if output_has_class_token:
            input_param = torch.cat((input_param_tok, input_param), dim=0)

        state_dict[key] = input_param


class VitBackbone(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self,
                 config,
                 pre_process=True,
                 post_process=True,
                 class_token=True,
                 single_token_output=False,
                 post_layer_norm=True,
                 drop_path_rate=0.0):
        super(VitBackbone, self).__init__(share_embeddings_and_output_weights=False)
        args = get_args()
        self.config = config

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        self.pre_process = pre_process
        self.post_process = post_process
        self.class_token = class_token
        self.post_layer_norm = post_layer_norm
        self.hidden_size = args.hidden_size
        self.patch_dim = args.patch_dim
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.micro_batch_size = args.micro_batch_size
        self.single_token_output = single_token_output
        self.drop_path_rate = drop_path_rate

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w
        self.seq_length = self.num_patches + (CLASS_TOKEN_LENGTH if self.class_token else 0)
        self.flatten_dim = self.patch_dim * self.patch_dim * args.num_channels
        self.input_tensor = None
        self.position_ids = None

        # preprocess中都没有用到megatron，对于CLIP而言，preprocess和posrprocess都是可以直接特定的
        if self.pre_process:
            # cls_token
            if self.class_token:
                self.cls_token = torch.nn.Parameter(
                    torch.randn(1, CLASS_TOKEN_LENGTH, self.hidden_size)
                )
                torch.nn.init.zeros_(self.cls_token)
            self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()

            # Linear encoder
            self.linear_encoder = torch.nn.Linear(
                self.flatten_dim, self.hidden_size
            )

            # embedding
            self.position_embeddings = torch.nn.Embedding(
                self.seq_length, self.hidden_size
            )
            init_method_normal(args.init_method_std)(
                self.position_embeddings.weight
            )

            args.class_token_present = self.class_token
            self.position_embeddings._register_load_state_dict_pre_hook(
                twod_interpolate_position_embeddings_hook
            )

            self.embedding_dropout = torch.nn.Dropout(args.hidden_dropout)

        # Transformer，要搞清楚config有哪些
        self.transformer = ParallelTransformer(
            config,
            model_type=args.model_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=self.post_layer_norm,
            drop_path_rate=self.drop_path_rate
        )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.transformer.set_input_tensor(input_tensor)

    def forward(self, input):

        if self.pre_process:
            rearranged_input = einops.rearrange(
                input,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_dim,
                p2=self.patch_dim,
            )

            assert rearranged_input.dtype == torch.half
            encoder_output = self.linear_encoder(rearranged_input)

            concatenated_tokens = encoder_output
            if self.class_token:
                cls_tokens = self.cls_token.expand(encoder_output.shape[0], -1, -1)
                concatenated_tokens = torch.cat((cls_tokens, encoder_output), dim=1)

            token_embeddings = concatenated_tokens + \
                    self.position_embeddings(self.position_ids[:, :concatenated_tokens.shape[1]])
            # [b, s, h] => [s, b, h]
            token_embeddings = token_embeddings.transpose(0, 1).contiguous()
            hidden_states = self.embedding_dropout(token_embeddings)
        else:
            hidden_states = input

        hidden_states = self.transformer(hidden_states, None)

        if self.post_process:
            # [s b h] => [b s h]
            if self.single_token_output:
                hidden_states = hidden_states[0]
            else:
                hidden_states = hidden_states.transpose(0, 1).contiguous()

        return hidden_states

class AttentionalPooler(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256
    ):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = torch.nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = torch.nn.LayerNorm(d_model)
        self.ln_k = torch.nn.LayerNorm(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

class CLIP_VitBackbone(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self,
                 config,
                 pre_process=True,
                 post_process=True,
                 single_token_output=False,
                 post_layer_norm=True,
                 drop_path_rate=0.0):
        super().__init__(config=config, share_embeddings_and_output_weights=False)
        args = get_args()
        self.config = config

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        self.pre_process = pre_process
        self.post_process = post_process
        self.class_token = args.v_concat_cls_token
        self.post_layer_norm = post_layer_norm
        self.hidden_size = args.v_hidden_size
        self.patch_dim = args.patch_dim
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.micro_batch_size = args.micro_batch_size
        self.single_token_output = single_token_output
        self.output_tokens = args.v_output_tokens
        self.drop_path_rate = drop_path_rate
        self.input_patchnorm = args.v_input_patchnorm
        self.global_average_pool = args.v_global_average_pool
        self.layer_norm = get_norm(config)

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w
        self.seq_length = self.num_patches + (args.v_cls_token_len if self.class_token else 0)
        print(f"seq_length: {self.seq_length}", flush=True)
        assert self.seq_length == args.v_seq_length
        self.flatten_dim = self.patch_dim * self.patch_dim * args.num_channels
        self.input_tensor = None
        self.position_ids = None

        if self.pre_process:
            # cls_token
            if self.class_token:
                self.cls_token = torch.nn.Parameter(
                    torch.randn(1, CLASS_TOKEN_LENGTH, self.hidden_size)
                )
                torch.nn.init.zeros_(self.cls_token)
            self.position_ids = torch.arange(self.seq_length).expand(self.micro_batch_size, -1).cuda()

            # Linear encoder
            self.linear_encoder = torch.nn.Linear(
                self.flatten_dim, self.hidden_size
            )

            # embedding
            self.position_embeddings = torch.nn.Embedding(
                self.seq_length, self.hidden_size
            )
            init_method_normal(args.init_method_std)(
                self.position_embeddings.weight
            )

            self.ln_pre = torch.nn.LayerNorm(self.hidden_size, eps=1e-5)

            args.class_token_present = self.class_token
            self.position_embeddings._register_load_state_dict_pre_hook(
                twod_interpolate_position_embeddings_hook
            )

            self.embedding_dropout = torch.nn.Dropout(args.hidden_dropout)
        
        self.transformer = ParallelTransformer(
            config,
            model_type=args.model_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
            # post_layer_norm=self.post_layer_norm,
            post_norm = False,  # 在外面已经使用了ln
            drop_path_rate=self.drop_path_rate,
            modality="image"
        )
        scale = self.hidden_size ** -0.5
        if self.post_process:
            if args.v_attentional_pool:
                self.attn_pool = AttentionalPooler(self.hidden_size, self.hidden_size, n_head=args.v_attn_pooler_heads, n_queries=args.v_nqueries)
                # What is output dim
                self.ln_post = torch.nn.LayerNorm(self.hidden_size, eps=1e-5)
                self.proj = torch.nn.Parameter(scale * torch.randn(self.hidden_size, args.clip_embeded_dim))
            else:
                self.ln_post = torch.nn.LayerNorm(self.hidden_size, eps=1e-5)
                self.proj = torch.nn.Parameter(scale * torch.randn(self.hidden_size, args.clip_embeded_dim))

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.transformer.set_input_tensor(input_tensor[0])

    def _global_pool(self, x: torch.Tensor):
        if self.global_average_pool:
            # 平均值池化
            return x.mean(dim=1), x
        else:
            '''
            在这种模式下, x 中的每个样本被假设包含两部分：第一个元素是特定的值，而剩余的元素构成其余特征。
            方法返回一个元组，第一个元素是 x 中每个样本的第一个元素，即 x[:, 0]。这个元素通常被视为某种特殊的值。
            第二个元素是 x 中每个样本的其余部分，即 x[:, 1:]。这个部分包含了除第一个元素之外的所有特征。
            第一个元素就是分类结果，第二个元素就是特征向量
            '''
            return x[:, 0], x[:, 1:]
        
    def forward(self, input):
        if self.pre_process:
            rearranged_input = einops.rearrange(
                input,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", # 16 256 hidden_width
                p1=self.patch_dim,
                p2=self.patch_dim,
            )
            # rearranged_input = rearranged_input.to(torch.half)
            # assert rearranged_input.dtype == torch.half
            args = get_args()
            encoder_output = self.linear_encoder(rearranged_input)
            if self.class_token:
                cls_tokens = self.cls_token.expand(encoder_output.shape[0], -1, -1)
                concatenated_tokens = torch.cat((cls_tokens, encoder_output), dim=1)
            else:
                concatenated_tokens = encoder_output
            #print(f"{concatenated_tokens.shape}, {self.position_ids.shape},{self.position_ids[:, :concatenated_tokens.shape[1]].shape}", flush=True)
            token_embeddings = concatenated_tokens + \
                    self.position_embeddings(self.position_ids[:, :concatenated_tokens.shape[1]])
            # Add patch dropout and layernorm
            token_embeddings = self.ln_pre(token_embeddings)

            # [b, s, h] => [s, b, h]
            token_embeddings = token_embeddings.transpose(0, 1).contiguous()
            hidden_states = self.embedding_dropout(token_embeddings)
        else:
            hidden_states = input
        hidden_states = self.transformer(hidden_states=hidden_states, attention_mask=None)
        if self.post_process:
            # [s b h] => [b s h]
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            if self.attn_pool is not None:
                hidden_states = self.attn_pool(hidden_states)
                hidden_states = self.ln_post(hidden_states)
                pooled, tokens = self._global_pool(hidden_states)
            else:
                pooled, tokens = self._global_pool(hidden_states)
                pooled = self.layer_norm(pooled)
            if self.proj is not None:
                pooled = pooled @ self.proj
            # def hook(gard):
            #     print(f"grad hook:[vit_backbone.proj]: {gard}", flush=True)
            #     return gard
            # self.proj.register_hook(hook)
            
            if self.output_tokens:
                return pooled, tokens
            return pooled

        return hidden_states