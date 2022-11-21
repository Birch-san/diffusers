from functools import partial
from typing import Optional, Protocol, Tuple
from torch import Tensor
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D

class CADBForward(Protocol):
  def forward(self, hidden_states: Tensor, temb: Optional[Tensor]=None, encoder_hidden_states: Optional[Tensor]=None): ...

def forward(
  self: CrossAttnDownBlock2D,
  orig_fn: CADBForward,
  hidden_states: Tensor,
  temb: Optional[Tensor]=None,
  encoder_hidden_states: Optional[Tensor]=None
):
  output_states: Tuple[Tensor, ...] = ()
  batch, channels, height, width = hidden_states.shape
  # self.height = height
  # self.width = width
  hidden_states: Tensor = hidden_states.flatten(2).unsqueeze(2)
  # hidden_states: Tensor = orig_fn(hidden_states, temb, encoder_hidden_states)
  for resnet, attn in zip(self.resnets, self.attentions):
    resnet.height = height
    resnet.width = width
    hidden_states: Tensor = resnet(hidden_states, temb)
    hidden_states: Tensor = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
  hidden_states: Tensor = hidden_states.unflatten(3, (height, width)).squeeze(2)
  output_states += (hidden_states,)
  if self.downsamplers is not None:
    for downsampler in self.downsamplers:
      hidden_states = downsampler(hidden_states)

    output_states += (hidden_states,)

  return hidden_states, output_states

def adapt_cadb(cadb: CrossAttnDownBlock2D) -> None:
  setattr(cadb, 'forward', partial(forward.__get__(cadb), cadb.forward))