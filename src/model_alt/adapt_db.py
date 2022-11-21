from functools import partial
from typing import Optional, Protocol, Tuple
from torch import Tensor
from diffusers.models.unet_2d_blocks import DownBlock2D

class DBForward(Protocol):
  def forward(self, hidden_states: Tensor, temb: Optional[Tensor]=None): ...

def forward(
  self: DownBlock2D,
  orig_fn: DBForward,
  hidden_states: Tensor,
  temb: Optional[Tensor]=None,
):
  output_states: Tuple[Tensor, ...] = ()
  batch, channels, height, width = hidden_states.shape
  # self.height = height
  # self.width = width
  hidden_states: Tensor = hidden_states.flatten(2).unsqueeze(2)
  # hidden_states: Tensor = orig_fn(hidden_states, temb, encoder_hidden_states)
  for resnet in self.resnets:
    resnet.height = height
    resnet.width = width
    hidden_states: Tensor = resnet(hidden_states, temb)
  hidden_states: Tensor = hidden_states.unflatten(3, (height, width)).squeeze(2)
  output_states += (hidden_states,)
  if self.downsamplers is not None:
    for downsampler in self.downsamplers:
      hidden_states = downsampler(hidden_states)

    output_states += (hidden_states,)

  return hidden_states, output_states

def adapt_db(db: DownBlock2D) -> None:
  setattr(db, 'forward', partial(forward.__get__(db), db.forward))