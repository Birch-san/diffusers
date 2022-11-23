from functools import partial
from typing import Optional, Protocol, Tuple, Union
from torch import Tensor, FloatTensor, tensor, is_tensor
from diffusers.models.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from diffusers.utils import logging
import torch

logger = logging.get_logger(UNet2DConditionModel.__name__)

class UNCMForward(Protocol):
  def forward(
    self: UNet2DConditionModel,
    sample: FloatTensor,
    timestep: Union[Tensor, float, int],
    encoder_hidden_states: Tensor,
    return_dict: bool = True,
  ): ...

def forward(
  self: UNet2DConditionModel,
  orig_fn: UNCMForward,
  sample: FloatTensor,
  timestep: Union[Tensor, float, int],
  encoder_hidden_states: Tensor,
  return_dict: bool = True,
):
  default_overall_up_factor = 2**self.num_upsamplers

  forward_upsample_size = False
  upsample_size = None

  if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
    logger.info("Forward upsample size to force interpolation output size.")
    forward_upsample_size = True

  if self.config.center_input_sample:
    sample = 2 * sample - 1.0

  timesteps = timestep
  if not is_tensor(timesteps):
    timesteps = tensor([timesteps], dtype=torch.long, device=sample.device)
  elif is_tensor(timesteps) and len(timesteps.shape) == 0:
    timesteps = timesteps[None].to(sample.device)
  
  timesteps = timesteps.expand(sample.shape[0])

  t_emb = self.time_proj(timesteps)

  t_emb = t_emb.to(dtype=self.dtype)
  emb = self.time_embedding(t_emb)

  sample = self.conv_in(sample)

  # record height and width so we don't lose them in the flatten;
  # enables us to return to height * width format (e.g. for convolutions in resnet blocks)
  batch, channels, height, width = sample.shape

  # adapt BCHW to BC1S format; optimal for Neural Engine
  sample = sample.reshape(batch, channels, 1, height * width)

  down_block_res_samples = (sample,)
  for downsample_block in self.down_blocks:
    if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
      sample, res_samples, height, width = downsample_block(
        hidden_states=sample,
        temb=emb,
        encoder_hidden_states=encoder_hidden_states,
        height=height,
        width=width,
      )
    else:
      sample, res_samples, height, width = downsample_block(
        hidden_states=sample,
        temb=emb,
        height=height,
        width=width,
      )

    down_block_res_samples += res_samples
  
  sample = self.mid_block(
    sample,
    emb,
    encoder_hidden_states=encoder_hidden_states,
    height=height,
    width=width,
  )

  for i, upsample_block in enumerate(self.up_blocks):
    is_final_block = i == len(self.up_blocks) - 1

    res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
    down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

    # if we have not reached the final block and need to forward the
    # upsample size, we do it here
    if not is_final_block and forward_upsample_size:
      upsample_size = down_block_res_samples[-1].shape[2:]

    if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
      sample, height, width = upsample_block(
        hidden_states=sample,
        temb=emb,
        res_hidden_states_tuple=res_samples,
        encoder_hidden_states=encoder_hidden_states,
        upsample_size=upsample_size,
        height=height,
        width=width,
      )
    else:
      sample, height, width = upsample_block(
        hidden_states=sample,
        temb=emb,
        res_hidden_states_tuple=res_samples,
        upsample_size=upsample_size,
        height=height,
        width=width,
      )

  batch, channels, *_ = sample.shape
  sample: Tensor = sample.reshape(batch, channels, height, width)

  sample = self.conv_norm_out(sample)
  sample = self.conv_act(sample)
  sample = self.conv_out(sample)

  if not return_dict:
    return (sample,)

  return UNet2DConditionOutput(sample=sample)

def adapt_uncm(uncm: UNCMForward) -> None:
  setattr(uncm, 'forward', partial(forward.__get__(uncm), uncm.forward))