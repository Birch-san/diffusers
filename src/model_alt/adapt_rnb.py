from functools import partial
from torch import Tensor
from diffusers.models.resnet import ResnetBlock2D
from typing import Protocol

from .tensor_decorator import TensorDecorator

class ResnetBlock2DForward(Protocol):
  def forward(
    self: ResnetBlock2D,
    input_tensor: Tensor,
    temb: Tensor
  ) -> Tensor: ...

def forward(
  self: ResnetBlock2D,
  orig_fn: ResnetBlock2DForward,
  input_tensor: Tensor,
  temb: Tensor,
  height: int,
  width: int,
) -> Tensor:
  # store height and width somewhere the conv2d can find it. sorry.
  self.height = height
  self.width = width
  input_tensor: Tensor = orig_fn(input_tensor, temb)
  return input_tensor

def adapt_conv(
  self: ResnetBlock2D,
  orig_fn: TensorDecorator,
  input_tensor: Tensor
) -> Tensor:
  input_tensor: Tensor = input_tensor.unflatten(3, (self.height, self.width)).squeeze(2)
  input_tensor: Tensor = orig_fn(input_tensor)
  input_tensor: Tensor = input_tensor.flatten(2).unsqueeze(2)
  return input_tensor


def adapt_rnb(rnb: ResnetBlock2D) -> None:
  setattr(rnb, 'forward', partial(forward.__get__(rnb), rnb.forward))

  setattr(rnb.conv1, 'forward', partial(adapt_conv.__get__(rnb), rnb.conv1.forward))
  setattr(rnb.conv2, 'forward', partial(adapt_conv.__get__(rnb), rnb.conv2.forward))