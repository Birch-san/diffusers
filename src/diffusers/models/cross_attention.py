# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Union
from .sub_quadratic_attention import efficient_dot_product_attention

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ..utils.import_utils import is_xformers_available


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        processor: Optional["AttnProcessor"] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

        # set attention processor
        processor = processor if processor is not None else CrossAttnProcessor()
        self.set_processor(processor)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if use_memory_efficient_attention_xformers:
            if self.added_kv_proj_dim is not None:
                # TODO(Anton, Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
                # which uses this type of cross attention ONLY because the attention mask of format
                # [0, ..., -10.000, ..., 0, ...,] is not supported
                raise NotImplementedError(
                    "Memory efficient attention with `xformers` is currently not supported when"
                    " `self.added_kv_proj_dim` is defined."
                )
            elif not is_xformers_available():
                raise ModuleNotFoundError(
                    "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                    " xformers",
                    name="xformers",
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    _ = xformers.ops.memory_efficient_attention(
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                    )
                except Exception as e:
                    raise e

            processor = XFormersCrossAttnProcessor()
        else:
            processor = CrossAttnProcessor()

        self.set_processor(processor)

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = CrossAttnAddedKVProcessor()
        else:
            processor = CrossAttnProcessor()

        self.set_processor(processor)
    
    def set_subquadratic_attention(
        self,
        query_chunk_size = 1024,
        kv_chunk_size: Optional[int] = None,
        kv_chunk_size_min: Optional[int] = None,
        chunk_threshold_bytes: Optional[int] = None,
    ):
        r"""
        Args:
            query_chunk_size (`int`, *optional*, defaults to `1024`)
            kv_chunk_size (`int`, *optional*, defaults to `None`): if None, sqrt(key_tokens) is used.
            kv_chunk_size_min (`int`, *optional*, defaults to `None`): only considered when `kv_chunk_size is None`. changes `sqrt(key_tokens)` into `max(sqrt(key_tokens), kv_chunk_size_min)`, to ensure our chunk sizes don't get too small (smaller chunks = more chunks = less concurrent work done).
            chunk_threshold_bytes (`int`, *optional*, defaults to `None`): if defined: only bother chunking if the self-attn matmul would allocate more bytes than this. whenever we can fit traditional attention into memory: we should prefer to do so, as the unchunked algorithm is faster.
        """
        processor = SubQuadraticCrossAttnProcessor(
            query_chunk_size=query_chunk_size,
            kv_chunk_size=kv_chunk_size,
            kv_chunk_size_min=kv_chunk_size_min,
            chunk_threshold_bytes=chunk_threshold_bytes,
        )

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor"):
        self.processor = processor

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `CrossAttention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length):
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        if attention_mask.shape[-1] != target_length:
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        return attention_mask


class CrossAttnProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class SubQuadraticCrossAttnProcessor:
    query_chunk_size: int
    kv_chunk_size: Optional[int]
    kv_chunk_size_min: Optional[int]
    chunk_threshold_bytes: Optional[int]
    def __init__(
        self,
        query_chunk_size = 1024,
        kv_chunk_size: Optional[int] = None,
        kv_chunk_size_min: Optional[int] = None,
        chunk_threshold_bytes: Optional[int] = None,
    ):
        r"""
        Args:
            query_chunk_size (`int`, *optional*, defaults to `1024`)
            kv_chunk_size (`int`, *optional*, defaults to `None`): if None, sqrt(key_tokens) is used.
            kv_chunk_size_min (`int`, *optional*, defaults to `None`): only considered when `kv_chunk_size is None`. changes `sqrt(key_tokens)` into `max(sqrt(key_tokens), kv_chunk_size_min)`, to ensure our chunk sizes don't get too small (smaller chunks = more chunks = less concurrent work done).
            chunk_threshold_bytes (`int`, *optional*, defaults to `None`): if defined: only bother chunking if the self-attn matmul would allocate more bytes than this. whenever we can fit traditional attention into memory: we should prefer to do so, as the unchunked algorithm is faster.
        """
        self.query_chunk_size = query_chunk_size
        self.kv_chunk_size = kv_chunk_size
        self.kv_chunk_size_min = kv_chunk_size_min
        self.chunk_threshold_bytes = chunk_threshold_bytes

    def __call__(
        self,
        attn: CrossAttention,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor]=None,
        attention_mask: Optional[Tensor]=None,
    ):
        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        assert attention_mask is None, "attention-mask not currently implemented for SubQuadraticCrossAttnProcessor."
        # I don't know what test case can be used to determine whether softmax is computed at sufficient bit-width,
        # but sub-quadratic attention has a pretty bespoke softmax (defers computation of the denominator) so this needs some thought.
        assert not attn.upcast_softmax or torch.finfo(hidden_states.dtype).bits >= 32, "upcast_softmax was requested, but is not implemented"

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = query.unflatten(-1, (attn.heads, -1)).transpose(1,2).flatten(end_dim=1)
        key_t = key.transpose(1,2).unflatten(1, (attn.heads, -1)).flatten(end_dim=1)
        del key
        value = value.unflatten(-1, (attn.heads, -1)).transpose(1,2).flatten(end_dim=1)

        dtype = query.dtype
        # TODO: do we still need to do *everything* in float32, given how we delay the division?
        # TODO: do we need to support upcast_softmax too? SD 2.1 seems to work without it
        if attn.upcast_attention:
            query = query.float()
            key_t = key_t.float()

        bytes_per_token = torch.finfo(query.dtype).bits//8
        batch_x_heads, q_tokens, _ = query.shape
        _, _, k_tokens = key_t.shape
        qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens

        query_chunk_size = self.query_chunk_size
        kv_chunk_size = self.kv_chunk_size

        if self.chunk_threshold_bytes is not None and qk_matmul_size_bytes <= self.chunk_threshold_bytes:
            # the big matmul fits into our memory limit; do everything in 1 chunk,
            # i.e. send it down the unchunked fast-path
            query_chunk_size = q_tokens
            kv_chunk_size = k_tokens

        hidden_states = efficient_dot_product_attention(
            query,
            key_t,
            value,
            query_chunk_size=query_chunk_size,
            kv_chunk_size=kv_chunk_size,
            kv_chunk_size_min=self.kv_chunk_size_min,
            use_checkpoint=attn.training,
        )

        hidden_states = hidden_states.to(dtype)

        hidden_states = hidden_states.unflatten(0, (-1, attn.heads)).transpose(1,2).flatten(start_dim=2)

        out_proj, dropout = attn.to_out
        hidden_states = out_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        return hidden_states


class CrossAttnAddedKVProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2)

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
        value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


class XFormersCrossAttnProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SlicedAttnProcessor:
    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // attn.heads), device=query.device, dtype=query.dtype
        )

        for i in range(hidden_states.shape[0] // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SlicedAttnAddedKVProcessor:
    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(self, attn: "CrossAttention", hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)

        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
        value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)

        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // attn.heads), device=query.device, dtype=query.dtype
        )

        for i in range(hidden_states.shape[0] // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


AttnProcessor = Union[
    CrossAttnProcessor,
    XFormersCrossAttnProcessor,
    SlicedAttnProcessor,
    CrossAttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
]
