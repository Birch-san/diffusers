from functools import partial
from torch import Tensor
from diffusers.models.resnet import ResnetBlock2D
from typing import Callable
from typing_extensions import TypeAlias

# from .adapt_groupnorm import to_agn, ANEGroupNorm
from .tensor_decorator import TensorDecorator

Forward: TypeAlias = Callable[[Tensor, Tensor], Tensor]

def forward(self: ResnetBlock2D, orig_fn: Forward, input_tensor: Tensor, temb: Tensor) -> Tensor:
  batch, channels, height, width = input_tensor.shape
  self.height = height
  self.width = width
  input_tensor: Tensor = input_tensor.flatten(2).unsqueeze(2)
  input_tensor: Tensor = orig_fn(input_tensor, temb)
  input_tensor: Tensor = input_tensor.unflatten(3, (height, width)).squeeze(2)
  return input_tensor

def adapt_conv(self: ResnetBlock2D, orig_fn: TensorDecorator, input_tensor: Tensor) -> Tensor:
  input_tensor: Tensor = input_tensor.unflatten(3, (self.height, self.width)).squeeze(2)
  input_tensor: Tensor = orig_fn(input_tensor)
  input_tensor: Tensor = input_tensor.flatten(2).unsqueeze(2)
  return input_tensor


def adapt_rnb(rnb: ResnetBlock2D) -> None:
  # agn1: ANEGroupNorm = to_agn(rnb.norm1)
  # setattr(rnb, 'norm1', agn1)
  # agn2: ANEGroupNorm = to_agn(rnb.norm2)
  # setattr(rnb, 'norm2', agn2)
  setattr(rnb, 'forward', partial(forward.__get__(rnb), rnb.forward))

  setattr(rnb.conv1, 'forward', partial(adapt_conv.__get__(rnb), rnb.conv1.forward))
  setattr(rnb.conv2, 'forward', partial(adapt_conv.__get__(rnb), rnb.conv2.forward))