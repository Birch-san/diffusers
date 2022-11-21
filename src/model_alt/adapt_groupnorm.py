from torch import nn, Tensor

class ANEGroupNorm(nn.GroupNorm):
  def forward(self, input: Tensor) -> Tensor:
    batch, channels, height, width = input.shape
    input: Tensor = input.flatten(2).unsqueeze(2)
    input: Tensor = super().forward(input)
    input: Tensor = input.unflatten(3, (height, width)).squeeze(2)
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