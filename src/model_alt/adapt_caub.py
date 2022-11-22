from functools import partial
from typing import Optional, Protocol, Tuple
from torch import Tensor, cat
from diffusers.models.unet_2d_blocks import CrossAttnUpBlock2D

class CAUBForward(Protocol):
  def forward(
    self: CrossAttnUpBlock2D,
    hidden_states: Tensor,
    res_hidden_states_tuple: Tuple[Tensor, ...],
    temb: Optional[Tensor]=None,
    upsample_size: Optional[int]=None,
  ): ...

def forward(
  self: CrossAttnUpBlock2D,
  orig_fn: CAUBForward,
  hidden_states: Tensor,
  res_hidden_states_tuple: Tuple[Tensor, ...],
  height: int,
  width: int,
  temb: Optional[Tensor]=None,
  encoder_hidden_states: Optional[Tensor]=None,
  upsample_size: Optional[int]=None,
) -> Tensor:
  for resnet, attn in zip(self.resnets, self.attentions):
    # pop res hidden states
    res_hidden_states: Tensor = res_hidden_states_tuple[-1]
    res_hidden_states_tuple: Tuple[Tensor, ...] = res_hidden_states_tuple[:-1]
    hidden_states: Tensor = cat([hidden_states, res_hidden_states], dim=1)

    hidden_states: Tensor = resnet(hidden_states, temb, height=height, width=width)
    hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample

  if self.upsamplers is not None:
    # needs to be 2D again for upsampler, probably
    hidden_states: Tensor = hidden_states.unflatten(3, (height, width)).squeeze(2)
    for upsampler in self.upsamplers:
      hidden_states: Tensor = upsampler(hidden_states, upsample_size)
      _, _, height, width = hidden_states.shape
    
    hidden_states: Tensor = hidden_states.flatten(2).unsqueeze(2)

  return hidden_states, height, width

def adapt_caub(caub: CrossAttnUpBlock2D) -> None:
  setattr(caub, 'forward', partial(forward.__get__(caub), caub.forward))