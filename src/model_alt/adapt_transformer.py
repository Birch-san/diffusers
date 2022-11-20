from functools import partial
from torch import Tensor
from typing import Union, Tuple
from diffusers.models.attention import Transformer2DModel, Transformer2DModelOutput

def forward(
  self: Transformer2DModel,
  hidden_states: Tensor,
  encoder_hidden_states=None,
  timestep=None,
  return_dict: bool = True,
) -> Union[Transformer2DModelOutput, Tuple[Tensor]]:
  batch, channel, height, weight = hidden_states.shape
  residual = hidden_states
  hidden_states = self.norm(hidden_states)
  hidden_states = self.proj_in(hidden_states)
  _, inner_dim, *_ = hidden_states.shape
  # hidden_states: Tensor = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
  hidden_states: Tensor = hidden_states.flatten(2).unsqueeze(2)
  for block in self.transformer_blocks:
    hidden_states: Tensor = block(hidden_states, context=encoder_hidden_states, timestep=timestep)

  hidden_states: Tensor = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2)
  hidden_states: Tensor = self.proj_out(hidden_states)
  output: Tensor = hidden_states + residual

  if not return_dict:
    return (output,)

  return Transformer2DModelOutput(sample=output)

def adapt_t2dm(t: Transformer2DModel) -> None:
  setattr(t, 'forward', forward.__get__(t))