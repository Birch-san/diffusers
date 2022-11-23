from torch import nn, Tensor

class ANEGroupNorm(nn.GroupNorm):
  def forward(self, input: Tensor) -> Tensor:
    batch, channels, height, width = input.shape
    input: Tensor = input.reshape(batch, channels, 1, height * width)
    input: Tensor = super().forward(input)
    batch, channels, *_ = input.shape
    input: Tensor = input.reshape(batch, channels, height, width)
    return input

def to_agn(gn: nn.GroupNorm) -> ANEGroupNorm:
  agn = ANEGroupNorm(
    num_groups=gn.num_groups,
    num_channels=gn.num_channels,
    eps=gn.eps,
    affine=gn.affine,
  )
  setattr(agn, 'weight', gn.weight)
  setattr(agn, 'bias', gn.bias)
  return agn