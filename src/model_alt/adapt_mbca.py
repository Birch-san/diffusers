from functools import partial
from typing import Optional, Protocol
from torch import Tensor
from diffusers.models.unet_2d_blocks import UNetMidBlock2DCrossAttn

class MBCAForward(Protocol):
  def forward(self, hidden_states: Tensor, temb: Optional[Tensor]=None): ...

def forward(
  self: UNetMidBlock2DCrossAttn,
  orig_fn: MBCAForward,
  hidden_states: Tensor,
  temb: Tensor,
  height: int,
  width: int,
  encoder_hidden_states: Optional[Tensor]=None,
) -> Tensor:
  first, *rest = self.resnets
  hidden_states = first(hidden_states, temb, height=height, width=width)
  for attn, resnet in zip(self.attentions, rest):
    hidden_states: Tensor = attn(hidden_states, encoder_hidden_states).sample
    hidden_states: Tensor = resnet(hidden_states, temb, height=height, width=width)
  
  return hidden_states

def adapt_mbca(mbca: UNetMidBlock2DCrossAttn) -> None:
  setattr(mbca, 'forward', partial(forward.__get__(mbca), mbca.forward))