from diffusers.models.resnet import ResnetBlock2D

from .adapt_groupnorm import to_agn, ANEGroupNorm

def adapt_rnb(rnb: ResnetBlock2D) -> None:
  agn1: ANEGroupNorm = to_agn(rnb.norm1)
  setattr(rnb, 'norm1', agn1)
  agn2: ANEGroupNorm = to_agn(rnb.norm2)
  setattr(rnb, 'norm2', agn2)