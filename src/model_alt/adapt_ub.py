from functools import partial
from typing import Optional, Protocol, Tuple
from torch import Tensor, cat
from diffusers.models.unet_2d_blocks import UpBlock2D

class UBForward(Protocol):
  def forward(
    self: UpBlock2D,
    hidden_states: Tensor,
    res_hidden_states_tuple: Tuple[Tensor, ...],
    temb: Optional[Tensor]=None,
    upsample_size: Optional[int]=None,
  ): ...

def forward(
  self: UpBlock2D,
  orig_fn: UBForward,
  hidden_states: Tensor,
  res_hidden_states_tuple: Tuple[Tensor, ...],
  temb: Optional[Tensor]=None,
  upsample_size: Optional[int]=None,
) -> Tensor:
  batch, channels, height, width = hidden_states.shape
  hidden_states: Tensor = hidden_states.flatten(2).unsqueeze(2)

  for resnet in self.resnets:
    # pop res hidden states
    res_hidden_states: Tensor = res_hidden_states_tuple[-1]
    res_hidden_states_tuple: Tuple[Tensor, ...] = res_hidden_states_tuple[:-1]
    hidden_states: Tensor = cat([hidden_states, res_hidden_states], dim=1)

    resnet.height = height
    resnet.width = width
    hidden_states: Tensor = resnet(hidden_states, temb)

  hidden_states: Tensor = hidden_states.unflatten(3, (height, width)).squeeze(2)

  if self.upsamplers is not None:
    for upsampler in self.upsamplers:
      hidden_states = upsampler(hidden_states, upsample_size)

  return hidden_states

def adapt_ub(db: UpBlock2D) -> None:
  setattr(db, 'forward', partial(forward.__get__(db), db.forward))