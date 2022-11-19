from apple.multihead_attention import MultiHeadAttention
from torch import Tensor
from typing import Optional

from diffusers.models.attention import CrossAttention
from .init_conv2d import initialize_from_linear

class AMHADelegator(MultiHeadAttention):
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
    # debug
    self.inner_dim = inner_dim
    self.embed_dim = inner_dim
    self.dim_head = dim_head
    self.heads = heads
    self.cross_attention_dim = cross_attention_dim
    super().__init__(
      embed_dim=inner_dim,
      n_head=heads,
      dropout=dropout,
      bias=bias,
      batch_first=True,
      d_q=query_dim,
      d_k=cross_attention_dim,
      d_v=cross_attention_dim,
      d_out=query_dim,
    )
  
  def forward(self, hidden_states: Tensor, context: Optional[Tensor]=None) -> Tensor:
    # TODO: we're probably undoing a permute from upstream, so just eliminate that
    context = context.transpose(2,1).unsqueeze(2) if context is not None else hidden_states
    out, _ = super().forward(
      q=hidden_states,
      k=context,
      v=context,
    )
    # out = out.squeeze(2).transpose(2,1)
    return out

def to_amha(ca: CrossAttention) -> AMHADelegator:
  bias = ca.to_k.bias is not None
  assert bias == False
  mha = AMHADelegator(
    query_dim=ca.to_q.in_features,
    cross_attention_dim=ca.to_k.in_features,
    heads=ca.heads,
    dim_head=ca.to_q.out_features//ca.heads,
    dropout=ca.to_out[1].p,
    bias=bias,
  )
  initialize_from_linear(mha.q_proj, ca.to_q)
  initialize_from_linear(mha.k_proj, ca.to_k)
  initialize_from_linear(mha.v_proj, ca.to_v)
  initialize_from_linear(mha.out_proj, ca.to_out[0])
  return mha