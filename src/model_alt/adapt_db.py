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
  height: int,
  width: int,
  temb: Optional[Tensor]=None,
):
  output_states: Tuple[Tensor, ...] = ()
  for resnet in self.resnets:
    hidden_states: Tensor = resnet(hidden_states, temb, height=height, width=width)
    output_states += (hidden_states,)
  if self.downsamplers is not None:
    # needs to be 2D again before downsampler convolves it.
    batch, channels, *_ = hidden_states.shape
    hidden_states: Tensor = hidden_states.reshape(batch, channels, height, width)
    for downsampler in self.downsamplers:
      hidden_states = downsampler(hidden_states)
      _, _, height, width = hidden_states.shape
    batch, channels, *_ = hidden_states.shape
    hidden_states: Tensor = hidden_states.reshape(batch, channels, 1, height * width)

    output_states += (hidden_states,)

  return hidden_states, output_states, height, width

def adapt_db(db: DownBlock2D) -> None:
  setattr(db, 'forward', partial(forward.__get__(db), db.forward))