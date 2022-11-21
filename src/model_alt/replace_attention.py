from torch import nn
from apple.ffn import FFN
from apple.layer_norm import LayerNormANE
from diffusers.models.attention import BasicTransformerBlock, CrossAttention, Transformer2DModel
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, DownBlock2D, UNetMidBlock2DCrossAttn, UpBlock2D

from .adapt_torch_mha import MultiheadAttention, to_mha
from .adapt_ane_mha import AMHADelegator, to_amha
from .adapt_cadb import adapt_cadb
from .adapt_mbca import adapt_mbca
from .adapt_db import adapt_db
from .adapt_ub import adapt_ub
from .adapt_ff import to_aff
from .adapt_layernorm import to_aln
from .adapt_transformer import adapt_t2dm
from .adapt_rnb import adapt_rnb

def replace_attention(
  module: nn.Module,
  replacing_self_attention: bool,
  using_torch_self_attention: bool,
  using_ane_self_attention: bool,
  using_ane_cross_attention: bool,
  ) -> None:
  for name, m in module.named_children():
    if isinstance(m, CrossAttention):
      # is self-attention?
      is_self_attention = m.to_q.in_features == m.to_k.in_features
      if is_self_attention:
        if replacing_self_attention:
          if using_torch_self_attention:
            mha: MultiheadAttention = to_mha(m)
          elif using_ane_self_attention:
            mha: AMHADelegator = to_amha(m)
          setattr(module, name, mha)
      elif using_ane_cross_attention:
        mha: AMHADelegator = to_amha(m)
        setattr(module, name, mha)
    elif isinstance(m, BasicTransformerBlock):
      if using_ane_self_attention:
        aln: LayerNormANE = to_aln(m.norm1)
        setattr(m, 'norm1', aln)
      if using_ane_cross_attention:
        aln: LayerNormANE = to_aln(m.norm2)
        setattr(m, 'norm2', aln)
      aln: LayerNormANE = to_aln(m.norm3)
      setattr(m, 'norm3', aln)
      aff: FFN = to_aff(m.ff)
      setattr(m, 'ff', aff)
    elif isinstance(m, Transformer2DModel):
      adapt_t2dm(m)
    elif isinstance(m, ResnetBlock2D):
      adapt_rnb(m)
    elif isinstance(m, CrossAttnDownBlock2D):
      adapt_cadb(m)
    elif isinstance(m, DownBlock2D):
      adapt_db(m)
    elif isinstance(m, UNetMidBlock2DCrossAttn):
      adapt_mbca(m)
    elif isinstance(m, UpBlock2D):
      adapt_ub(m)