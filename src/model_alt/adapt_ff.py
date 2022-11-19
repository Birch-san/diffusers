from torch import nn
from typing import Tuple

from apple.ffn import FFN
from diffusers.models.attention import FeedForward, GEGLU
from .init_conv2d import initialize_from_linear


def to_aff(ff: FeedForward) -> FFN:
  ff_layers: Tuple[GEGLU, nn.Dropout, nn.Linear] = ff.net
  geglu, dropout, lin_out = ff_layers
  relu_in_dim = geglu.proj.out_features
  relu_out_dim, io_dim = lin_out.in_features, lin_out.out_features
  aff = FFN(
    embed_dim=io_dim,
    ffn_dim=relu_out_dim,
    dropout=dropout.p
  )
  _, _, dropout_, c2d_out = aff.layers
  relu_out = nn.Conv2d(io_dim, relu_in_dim, 1)
  initialize_from_linear(relu_out, geglu.proj)
  initialize_from_linear(c2d_out, lin_out)
  aff.layers = nn.ModuleList([
    geglu,
    dropout_,
    c2d_out,
  ])
  return aff