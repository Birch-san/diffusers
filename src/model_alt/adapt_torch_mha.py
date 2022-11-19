from torch import nn, Tensor, cat
from typing import Optional

from diffusers.models.attention import CrossAttention

class MultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
    ):
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        super().__init__(
            embed_dim=inner_dim,
            num_heads=heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            kdim=cross_attention_dim,
            vdim=cross_attention_dim,
        )

    def forward(self, hidden_states: Tensor, context: Optional[Tensor]=None) -> Tensor:
        context = context if context is not None else hidden_states
        out, _ = super().forward(
            query=hidden_states,
            key=context,
            value=context,
            need_weights=False,
        )
        return out

def to_mha(ca: CrossAttention) -> MultiheadAttention:
    bias = ca.to_k.bias is not None
    assert bias == False
    mha = MultiheadAttention(
        query_dim=ca.to_q.in_features,
        cross_attention_dim=ca.to_k.in_features,
        heads=ca.heads,
        dim_head=ca.to_q.out_features//ca.heads,
        dropout=ca.to_out[1].p,
        bias=bias,
    )
    # is self-attention?
    if ca.to_q.in_features == ca.to_k.in_features:
        mha.get_parameter('in_proj_weight').data = cat([ca.to_q.weight, ca.to_k.weight, ca.to_v.weight])
    else:
        mha.get_parameter('q_proj_weight').data = ca.to_q.weight
        mha.get_parameter('k_proj_weight').data = ca.to_k.weight
        mha.get_parameter('v_proj_weight').data = ca.to_v.weight
    mha.out_proj.weight = ca.to_out[0].weight
    mha.out_proj.bias = ca.to_out[0].bias
    return mha