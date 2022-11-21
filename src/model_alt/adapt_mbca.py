from functools import partial
from typing import Optional, Protocol, Tuple
from torch import Tensor
from diffusers.models.unet_2d_blocks import UNetMidBlock2DCrossAttn

class MBCAForward(Protocol):
  def forward(self, hidden_states: Tensor, temb: Optional[Tensor]=None): ...

def forward(
  self: UNetMidBlock2DCrossAttn,
  orig_fn: MBCAForward,
  hidden_states: Tensor,
  temb: Optional[Tensor]=None,
  encoder_hidden_states: Optional[Tensor]=None,
) -> Tensor:
  batch, channels, height, width = hidden_states.shape
  # self.height = height
  # self.width = width
  hidden_states: Tensor = hidden_states.flatten(2).unsqueeze(2)
  # hidden_states: Tensor = orig_fn(hidden_states, temb, encoder_hidden_states)
  first, *rest = self.resnets
  first.height = height
  first.width = width
  hidden_states = first(hidden_states, temb)
  for attn, resnet in zip(self.attentions, rest):
    resnet.height = height
    resnet.width = width
    hidden_states: Tensor = attn(hidden_states, encoder_hidden_states).sample
    hidden_states: Tensor = resnet(hidden_states, temb)
  hidden_states: Tensor = hidden_states.unflatten(3, (height, width)).squeeze(2)
  
  return hidden_states

def adapt_mbca(mbca: UNetMidBlock2DCrossAttn) -> None:
  setattr(mbca, 'forward', partial(forward.__get__(mbca), mbca.forward))