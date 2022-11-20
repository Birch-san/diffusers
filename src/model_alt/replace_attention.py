from torch import nn
from apple.ffn import FFN
from apple.layer_norm import LayerNormANE
from diffusers.models.attention import BasicTransformerBlock, CrossAttention, Transformer2DModel
from model_alt.adapt_transformer import adapt_t2dm

from .adapt_torch_mha import MultiheadAttention, to_mha
from .adapt_ane_mha import AMHADelegator, to_amha
from .adapt_ff import to_aff
from .adapt_layernorm import to_aln

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