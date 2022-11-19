from torch import nn

from apple.layer_norm import LayerNormANE

def to_aln(ln: nn.LayerNorm) -> LayerNormANE:
  dim, = ln.normalized_shape
  aln = LayerNormANE(dim)
  aln.weight.data = ln.weight.data
  aln.bias.data = ln.bias.data / ln.weight.data
  return aln