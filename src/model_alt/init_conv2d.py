from torch import Tensor, zeros, nn

def linear_to_conv2d(state: Tensor) -> Tensor:
  """
  adapts the weights or biases of an nn.Linear to be compatible with an nn.Conv2d
  by unsqueezing the final dimension twice
  """
  # TODO: would this benefit from .contiguous()?
  return state.view(*state.shape, 1, 1)

def initialize_from_linear(conv: nn.Conv2d, linear: nn.Linear) -> None:
  """
  initializes an nn.Conv2d layer with the weights and biases from a nn.Linear layer
  """
  conv.weight.data = linear_to_conv2d(linear.weight.data)
  if linear.bias is None:
    # since there's no bias Tensor to copy: we don't get a device/dtype transfer for free, so must do so explicitly
    # conv.bias.data = conv.bias.data.to(device=conv.weight.data.device, dtype=conv.weight.data.dtype)
    conv.bias.data = zeros(linear.out_features, device=conv.weight.data.device, dtype=conv.weight.data.dtype)
  else:
    conv.bias.data = linear.bias.data