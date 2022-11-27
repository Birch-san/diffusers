from functools import partial
import sys, os
# put repository root on CWD so that local diffusers is used
sys.path.insert(1, f'{os.getcwd()}/src')
sys.path.insert(1, f'{os.getcwd()}/src/coremltools')
sys.path.insert(1, f'{os.getcwd()}/src/k-diffusion')

# monkey-patch _randn to use CPU random before k-diffusion uses it
from torchsde._brownian.brownian_interval import _randn
from torchsde._brownian import brownian_interval
brownian_interval._randn = lambda size, dtype, device, seed: (
  _randn(size, dtype, 'cpu' if device.type == 'mps' else device, seed).to(device)
)

from k_diffusion import sampling
sampling.default_noise_sampler = lambda x: (
  lambda sigma, sigma_next: torch.randn_like(x, device='cpu' if x.device.type == 'mps' else x.device).to(x.device)
)

import torch
from torch import Generator, Tensor, randn, linspace, cumprod, no_grad, nn
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from k_diffusion.external import DiscreteEpsDDPMDenoiser
from k_diffusion.sampling import get_sigmas_karras, sample_dpmpp_2m
from model_alt.replace_attention import replace_attention
from transformers import CLIPTextModel, PreTrainedTokenizer, CLIPTokenizer, logging
from typing import Tuple, TypeAlias, Union, List, Optional, Callable, TypedDict
from PIL import Image
import time
from random import randint
import numpy as np

import coremltools as ct
from pathlib import Path
from coremltools.models import MLModel

# 5 DPMSolver++ steps, limited sigma schedule
prototyping = True
# 128x128 images, for even more extreme prototyping
smol = prototyping and True
tall = False
cfg_enabled = True
benchmarking = not prototyping and False
coreml_sampler = True
saving_coreml_model = True
searching = False
start_seed = 2178792735
# start_seed = 68673924
seeds = [
  start_seed,
  # 2178792735,
  # 1167622662,
]
# seeds = None
saving_coreml_ane = saving_coreml_model and True
# we shouldn't attempt to load CoreML model during the same run as when saving it, because sampling+VAE+encoder will be on-CPU and wrong dtype
loading_coreml_model = not saving_coreml_model and False
loading_coreml_ane = loading_coreml_model and False
using_ane_self_attention = False
using_ane_self_attention = True
using_ane_cross_attention = True
using_torch_self_attention = False
replacing_self_attention = using_ane_self_attention or using_torch_self_attention
replacing_attention = replacing_self_attention or using_ane_cross_attention
# workaround for two bad combinations
# on MPS, batching encounters correctness issues when using ANE-optimized self-attention
# on CoreML, batching encounters "Error computing NN outputs" issues when **not** using ANE-optimized self-attention
one_at_a_time = using_ane_self_attention != loading_coreml_model
# we save CoreML models in half-precision since that's all ANE supports
half = not loading_coreml_model and not saving_coreml_model and False

class KSamplerCallbackPayload(TypedDict):
  x: Tensor
  i: int
  sigma: Tensor
  sigma_hat: Tensor
  denoised: Tensor

KSamplerCallback: TypeAlias = Callable[[KSamplerCallbackPayload], None]

DeviceType: TypeAlias = Union[torch.device, str]

def get_betas(
  num_train_timesteps: int = 1000,
  beta_start: float = 0.00085,
  beta_end: float = 0.012,
  device: Optional[DeviceType] = None
) -> Tensor:
  return linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32, device=device) ** 2

def get_alphas(betas: Tensor) -> Tensor:
  return 1.0 - betas

def get_alphas_cumprod(alphas: Tensor) -> Tensor:
  return cumprod(alphas, dim=0)

class DiffusersSDDenoiser(DiscreteEpsDDPMDenoiser):
  inner_model: UNet2DConditionModel
  def __init__(self, unet: UNet2DConditionModel, alphas_cumprod: Tensor):
    super().__init__(unet, alphas_cumprod, quantize=True)

  def get_eps(
    self,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    return_dict: bool = True,
    ) -> Tensor:
    # if isinstance(self.inner_model, UNetWrapper):
    #   orig_dtype, orig_device = sample.dtype, sample.device
    #   sample = sample.to(dtype=torch.float16, device='cpu')
    #   timestep = timestep.cpu()
    #   encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float16, device='cpu')
    out: UNet2DConditionOutput = self.inner_model(
      sample,
      timestep,
      encoder_hidden_states=encoder_hidden_states,
      return_dict=return_dict,
    )
    # if isinstance(self.inner_model, UNetWrapper):
    #   out.sample = out.sample.to(dtype=orig_dtype, device=orig_device)
    return out.sample

  def sigma_to_t(self, sigma: Tensor, quantize=None) -> Tensor:
    return super().sigma_to_t(sigma, quantize=quantize).to(dtype=self.inner_model.dtype)

class CFGDenoiser():
  denoiser: DiffusersSDDenoiser
  def __init__(self, denoiser: DiffusersSDDenoiser):
    self.denoiser = denoiser
  
  def __call__(
    self,
    x: Tensor,
    sigma: Tensor,
    uncond: Tensor,
    cond: Tensor, 
    cond_scale: float
  ) -> Tensor:
    if uncond is None or cond_scale == 1.0:
      return self.denoiser(input=x, sigma=sigma, encoder_hidden_states=cond)
    if one_at_a_time:
      # if batching doesn't work: don't batch
      uncond = self.denoiser(input=x, sigma=sigma, encoder_hidden_states=uncond)
      cond = self.denoiser(input=x, sigma=sigma, encoder_hidden_states=cond)
      return uncond + (cond - uncond) * cond_scale
    cond_in = torch.cat([uncond, cond])
    del uncond, cond
    x_in = x.expand(cond_in.size(dim=0), -1, -1, -1)
    del x
    uncond, cond = self.denoiser(input=x_in, sigma=sigma, encoder_hidden_states=cond_in).chunk(cond_in.size(dim=0))
    del x_in, cond_in
    return uncond + (cond - uncond) * cond_scale

class Sampler(nn.Module):
  denoiser: CFGDenoiser
  sigmas: Tensor
  def __init__(self, unet: UNet2DConditionModel) -> None:
    super().__init__()
    self.unet = unet
    alphas_cumprod: Tensor = get_alphas_cumprod(get_alphas(get_betas(device=device))).to(dtype=unet.dtype)
    unet_k_wrapped = DiffusersSDDenoiser(unet, alphas_cumprod)
    self.denoiser = CFGDenoiser(unet_k_wrapped)
    if prototyping:
      # aggressively short sigma schedule; for cheap iteration
      steps=5
      sigma_max=torch.tensor(7.0796, device=alphas_cumprod.device, dtype=alphas_cumprod.dtype)
      sigma_min=torch.tensor(0.0936, device=alphas_cumprod.device, dtype=alphas_cumprod.dtype)
      rho=9.
    elif searching:
      # cheap but reasonably good results; for exploring seeds to pick one to subsequently master in more detail
      steps=8
      sigma_max=unet_k_wrapped.sigma_max
      sigma_min=torch.tensor(0.0936, device=alphas_cumprod.device, dtype=alphas_cumprod.dtype)
      rho=7.
    else:
      # higher quality, but still not too expensive
      steps=15
      sigma_max=unet_k_wrapped.sigma_max
      sigma_min=unet_k_wrapped.sigma_min
      rho=7.
    sigmas: Tensor = get_sigmas_karras(
      n=steps,
      sigma_max=sigma_max,
      sigma_min=sigma_min,
      rho=rho,
      device=device,
    ).to(unet.dtype)
    self.sigmas = sigmas
  
  def forward(
    self,
    latents: Tensor,
    cond: Tensor,
    uncond: Optional[Tensor]=None,
    cond_scale: Union[torch.Tensor, float, int] = 7.5 if cfg_enabled else 1.,
    brownian_tree_seed: Optional[int] = None,
  ) -> Tensor:
    # CoreML can only pass tensors
    if torch.is_tensor(cond_scale):
      cond_scale = cond_scale.item()
    extra_args = {
      'cond': cond,
      'uncond': uncond,
      'cond_scale': cond_scale
    }
    # noise_sampler = BrownianTreeNoiseSampler(
    #   latents,
    #   sigma_min=sigma_min,
    #   sigma_max=sigma_max,
    #   # there's no requirement that the noise sampler's seed be coupled to the init noise seed;
    #   # I'm just re-using it because it's a convenient arbitrary number
    #   seed=seed,
    # )
    # latents: Tensor = sample_heun(
    latents: Tensor = sample_dpmpp_2m(
    # latents: Tensor = sample_dpmpp_2s_ancestral(
      self.denoiser,
      latents * self.sigmas[0],
      self.sigmas,
      extra_args=extra_args,
      # callback=log_intermediate,
      # noise_sampler=noise_sampler,
    )
    # latents: Tensor = sample_dpm_adaptive(
    #   denoiser,
    #   latents * sigmas[0],
    #   sigma_min=sigma_min,
    #   sigma_max=sigma_max,
    #   extra_args=extra_args,
    #   # noise_sampler=noise_sampler,
    #   rtol=.003125,
    #   atol=.0004875,
    # )
    return latents

class log_level:
  orig_log_level: int
  log_level: int
  def __init__(self, log_level: int):
    self.log_level = log_level
    self.orig_log_level = logging.get_verbosity()
  def __enter__(self):
    logging.set_verbosity(self.log_level)
  def __exit__(self, exc_type, exc_value, exc_traceback):
    logging.set_verbosity(self.orig_log_level)

revision=None
torch_dtype=None
if saving_coreml_model:
  # gotta trace model on-CPU, and only float32 is supported
  # device = torch.device('cpu')
  device = torch.device('mps')
  revision='fp16'
  torch_dtype=torch.float16
  # could fp16 model revision make it trace any faster? probably not
else:
  if half:
    revision='fp16'
    torch_dtype=torch.float16
  device = torch.device('mps')

model_name = (
  # 'CompVis/stable-diffusion-v1-4'
  'hakurei/waifu-diffusion'
  # 'runwayml/stable-diffusion-v1-5'
)
if not loading_coreml_model:
  unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
    model_name,
    subfolder='unet',
    revision=revision,
    torch_dtype=torch_dtype,
  ).to(device).eval()
  sampler = Sampler(unet)

# vae_model_name = 'hakurei/waifu-diffusion-v1-4' if model_name == 'hakurei/waifu-diffusion' else model_name
vae: AutoencoderKL = AutoencoderKL.from_pretrained(
  model_name,
  subfolder='vae',
  revision=revision,
  torch_dtype=torch_dtype,
).to(device)

tokenizer: PreTrainedTokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
with log_level(logging.ERROR):
  text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch_dtype).to(device)

if replacing_attention and not loading_coreml_model:
  unet.apply(
    partial(
      replace_attention,
      replacing_self_attention=replacing_self_attention,
      using_torch_self_attention=using_torch_self_attention,
      using_ane_self_attention=using_ane_self_attention,
      using_ane_cross_attention=using_ane_cross_attention,
    )
  )

def get_mlp_name(module_name: str) -> str:
  return f'{module_name}.mlpackage'

def get_scriptmodule_name(module_name: str) -> str:
  return f'{module_name}.scriptmodule.pt'

class Undictifier(nn.Module):
  model: nn.Module
  def __init__(self, model: nn.Module):
    super().__init__()
    self.model = model
  def forward(self, *args, **kwargs): 
    return self.model(*args, **kwargs)["sample"]

def convert_unet(pt_model: UNet2DConditionModel, module_name: str) -> None:
  # from coremltools.converters.mil import Builder as mb
  # from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op, _TORCH_OPS_REGISTRY
  # import coremltools.converters.mil.frontend.torch.ops as cml_ops

  # coremltools 6.1.0 supports baddbmm
  # orig_baddbmm = torch.baddbmm
  # def fake_baddbmm(_: Tensor, batch1: Tensor, batch2: Tensor, beta: float, alpha: float):
  #   return torch.bmm(batch1, batch2) * alpha
  # torch.baddbmm = fake_baddbmm

  # orig_unflatten = torch.Tensor.unflatten
  # def fake_unflatten(self: Tensor, dim: int, shape: Tuple[int, int]) -> Tensor:
  #   assert dim == 3
  #   return self.reshape(*self.shape[:3], *shape)
  # torch.Tensor.unflatten = fake_unflatten

  # if "broadcast_to" in _TORCH_OPS_REGISTRY: del _TORCH_OPS_REGISTRY["broadcast_to"]
  # @register_torch_op
  # def broadcast_to(context, node): return cml_ops.expand(context, node)

  # if "gelu" in _TORCH_OPS_REGISTRY: del _TORCH_OPS_REGISTRY["gelu"]
  # @register_torch_op
  # def gelu(context, node): context.add(mb.gelu(x=context[node.inputs[0]], name=node.name))

  print("tracing")
  b = 1 if one_at_a_time or not cfg_enabled else 2
  latents_shape = (b, 4, 64, 64)
  timestep_shape = (1,)
  embeddings_shape = (b, 77, 768)
  with no_grad(), torch.jit.optimized_execution(True): # not sure whether no_grad is necessary but can't hurt
    trace = torch.jit.trace(
      Undictifier(pt_model),
      (
        torch.zeros(*latents_shape, device=device, dtype=torch_dtype),
        torch.zeros(*timestep_shape, device=device, dtype=torch_dtype),
        torch.zeros(*embeddings_shape, device=device, dtype=torch_dtype)
      ),
      strict=False,
      check_trace=False
    )
  print(f"finished tracing")
  scriptmod_name: str = get_scriptmodule_name(module_name)
  print(f"saving to '{scriptmod_name}'")
  trace.save(scriptmod_name)
  print(f"saved to '{scriptmod_name}'")

  print("converting to CoreML")
  # https://github.com/apple/coremltools/blob/870213ba6545369335ac72e61127c8d20ea745e5/coremltools/converters/mil/mil/ops/defs/iOS15/elementwise_unary.py
  # /Users/birch/anaconda3/envs/diffnightly/lib/python3.10/site-packages/coremltools/converters/mil/mil/ops/defs/iOS15/elementwise_unary.py
  # ERROR: 'float' object has no attribute 'astype'
  # changed:
  # + if isinstance(input_var.val, float):
  # +   return type_map[dtype_val](input_var.val)
  #   if not types.is_tensor(input_var.sym_type):
  compute_units=ct.ComputeUnit.ALL if saving_coreml_ane else ct.ComputeUnit.CPU_AND_GPU
  # dtype=ct.converters.mil.mil.types.fp16
  dtype=np.float16
  cm_model = ct.convert(
    trace.eval(), 
    inputs=[
      ct.TensorType(shape=latents_shape, dtype=dtype),
      ct.TensorType(shape=timestep_shape, dtype=dtype),
      ct.TensorType(shape=embeddings_shape, dtype=dtype)
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS13,
    compute_units=compute_units,
    skip_model_load=True
  )

  mlp_name: str = get_mlp_name(module_name)
  print(f"saving to '{mlp_name}'")
  cm_model.save(f"{mlp_name}")
  print(f"saved to '{mlp_name}'")

  # torch.baddbmm = orig_baddbmm
  # torch.Tensor.unflatten = orig_unflatten

def convert_sampler(pt_model: Sampler, module_name: str) -> None:
  from coremltools.converters.mil import Builder as mb
  from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op, _TORCH_OPS_REGISTRY
  import coremltools.converters.mil.frontend.torch.ops as cml_ops

  # coremltools 6.1.0 supports baddbmm
  # orig_baddbmm = torch.baddbmm
  # def fake_baddbmm(_: Tensor, batch1: Tensor, batch2: Tensor, beta: float, alpha: float):
  #   return torch.bmm(batch1, batch2) * alpha
  # torch.baddbmm = fake_baddbmm

  orig_new_ones = torch.Tensor.new_ones
  def fake_new_ones(self: Tensor, shape: Tuple[int, ...], *args, **kwargs):
    return torch.full(shape, 1)#, dtype=self.dtype, device=self.device)
  torch.Tensor.new_ones = fake_new_ones

  orig_expm1 = torch.Tensor.expm1
  def fake_expm1(self: Tensor, *args, **kwargs):
    return self.exp() - 1
  torch.Tensor.expm1 = fake_expm1

  orig_argmin = torch.Tensor.argmin
  def fake_argmin(self: Tensor, dim: Optional[int]=None, keepdim=False, *args, **kwargs):
    assert dim == 0
    assert keepdim == False
    _, indices = self.min(0)
    return indices
  torch.Tensor.argmin = fake_argmin

  if "broadcast_to" in _TORCH_OPS_REGISTRY: del _TORCH_OPS_REGISTRY["broadcast_to"]
  @register_torch_op
  def broadcast_to(context, node): return cml_ops.expand(context, node)

  if "gelu" in _TORCH_OPS_REGISTRY: del _TORCH_OPS_REGISTRY["gelu"]
  @register_torch_op
  def gelu(context, node): context.add(mb.gelu(x=context[node.inputs[0]], name=node.name))

  print("tracing")
  latents_shape = (1, 4, 64, 64)
  embedding_shape = (1, 77, 768)
  cond_scale_shape = (1,)
  with no_grad(): # not sure whether no_grad is necessary but can't hurt
    trace = torch.jit.trace(
      pt_model,
      (
        torch.zeros(*latents_shape),
        torch.zeros(*embedding_shape),
        torch.zeros(*embedding_shape),
        torch.full(cond_scale_shape, 1.),
      ),
      strict=False,
      check_trace=False
    )
  print(f"finished tracing")
  scriptmod_name: str = get_scriptmodule_name(module_name)
  print(f"saving to '{scriptmod_name}'")
  trace.save(scriptmod_name)
  print(f"saved to '{scriptmod_name}'")

  print("converting to CoreML")
  cm_model = ct.convert(
    trace, 
    inputs=[
      ct.TensorType(shape=latents_shape),
      ct.TensorType(shape=embedding_shape),
      ct.TensorType(shape=embedding_shape),
      ct.TensorType(shape=cond_scale_shape),
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    skip_model_load=True
  )

  mlp_name: str = get_mlp_name(module_name)
  print(f"saving to '{mlp_name}'")
  cm_model.save(f"{mlp_name}")
  print(f"saved to '{mlp_name}'")

  # torch.baddbmm = orig_baddbmm
  torch.Tensor.new_ones = orig_new_ones
  torch.Tensor.argmin = orig_argmin
  torch.Tensor.expm1 = orig_expm1

module_name = 'sampler' if coreml_sampler else 'unet_ane_optimized_fp16_2'
mlp_name: str = get_mlp_name(module_name)
if saving_coreml_model:
  if Path(mlp_name).exists():
    print(f"CoreML model '{mlp_name}' already exists")
  else:
    print("generating CoreML model")
    convert_sampler(sampler, module_name) if coreml_sampler else convert_unet(unet, module_name) 
    print(f"saved CoreML model '{mlp_name}'")
  # we refrain from loading the model and continuing, because our Unet, etc are on CPU/float32
  sys.exit()

class UNetWrapper:
  ml_model: MLModel
  dtype: torch.dtype
  def __init__(self, ml_model: MLModel):
    self.ml_model = ml_model
    self.device = device
    self.dtype = torch.float16

  def __call__(
    self, 
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    return_dict: bool = True,
  ) -> UNet2DConditionOutput:
    dtype = sample.dtype
    device = sample.device
    args = {
      "sample_1": sample.to(dtype=self.dtype, device='cpu').numpy(),
      "timestep": timestep.to(dtype=self.dtype, device='cpu').int().numpy(),
      "context": encoder_hidden_states.to(dtype=self.dtype, device='cpu').numpy(),
    }
    prediction = self.ml_model.predict(args)
    v, *_ = prediction.values()
    sample=torch.tensor(v, dtype=dtype, device=device)
    return UNet2DConditionOutput(sample=sample)

class SamplerWrapper:
  ml_model: MLModel
  def __init__(self, ml_model: MLModel):
    self.ml_model = ml_model

  def __call__(
    self, 
    latents: Tensor,
    cond: Tensor,
    uncond: Optional[Tensor]=None,
    cond_scale: Union[torch.Tensor, float, int] = 7.5 if cfg_enabled else 1.,
    brownian_tree_seed: Optional[int] = None,
  ) -> UNet2DConditionOutput:
    dtype = latents.dtype
    device = latents.device
    if torch.is_tensor(cond_scale):
      cond_scale = cond_scale.to(dtype=torch.float16, device='cpu')
    else:
      cond_scale = torch.tensor([cond_scale], dtype=torch.float16, device='cpu')
    if uncond is None:
      uncond = torch.zeros_like(cond)
    args = {
      "latents": latents.to(dtype=torch.float16, device='cpu').numpy(),
      "cond": cond.to(dtype=torch.float16, device='cpu').numpy(),
      "uncond": uncond.to(dtype=torch.float16, device='cpu').numpy(),
      "cond_scale": cond_scale.numpy(),
    }
    prediction = self.ml_model.predict(args)
    for v in prediction.values():
      sample=torch.tensor(v, dtype=dtype, device=device)
      return sample

if loading_coreml_model:
  compute_units=ct.ComputeUnit.ALL if loading_coreml_ane else ct.ComputeUnit.CPU_AND_GPU
  print(f"loading CoreML model '{mlp_name}'")
  assert Path(mlp_name).exists()
  cm_model = MLModel(mlp_name, compute_units=compute_units)
  print("loaded")
  if coreml_sampler:
    sampler = SamplerWrapper(cm_model)
  else:
    unet = UNetWrapper(cm_model)
    sampler = Sampler(unet)

@no_grad()
def latents_to_pils(latents: Tensor) -> List[Image.Image]:
  latents = 1 / 0.18215 * latents

  images: Tensor = vae.decode(latents).sample

  images = (images / 2 + 0.5).clamp(0, 1)

  # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
  images = images.cpu().permute(0, 2, 3, 1).float().numpy()
  images = (images * 255).round().astype("uint8")

  pil_images: List[Image.Image] = [Image.fromarray(image) for image in images]
  return pil_images

intermediates_path='intermediates'
os.makedirs(intermediates_path, exist_ok=True)
def log_intermediate(payload: KSamplerCallbackPayload) -> None:
  sample_pils: List[Image.Image] = latents_to_pils(payload['denoised'])
  for img in sample_pils:
    img.save(os.path.join(intermediates_path, f"inter.{payload['i']}.png"))

sample_path='out'
os.makedirs(sample_path, exist_ok=True)

# prompt = "masterpiece character portrait of a blonde girl, full resolution, 4k, mizuryuu kei, akihiko. yoshida, Pixiv featured, baroque scenic, by artgerm, sylvain sarrailh, rossdraws, wlop, global illumination, vaporwave"
# prompt = 'aqua (konosuba), carnelian, general content, one girl, looking at viewer, blue hair, bangs, medium breasts, frills, blue skirt, blue shirt, detached sleeves, long hair, blue eyes, green ribbon, sleeveless shirt, gem, thighhighs under boots, watercolor (medium), traditional media'
# prompt = 'artoria pendragon (fate), carnelian, 1girl, general content, upper body, white shirt, blonde hair, looking at viewer, medium breasts, hair between eyes, floating hair, green eyes, blue ribbon, long sleeves, light smile, hair ribbon, watercolor (medium), traditional media'
prompt = 'konpaku youmu, sazanami mio, from side, white shirt, green skirt, silver hair, looking at viewer, small breasts, hair between eyes, floating hair, short hair, neck ribbon, short sleeves, hair ribbon, hairband, bangs, miniskirt, vest, marker (medium), colored pencil (medium)'
# prompt = 'rem (re:zero), carnelian, 1girl, upper body, blue hair, looking at viewer, medium breasts, hair between eyes, floating hair, blue eyes, blue hair, short hair, roswaal mansion maid uniform, detached sleeves, detached collar, ribbon trim, maid headdress, x hair ornament, sunset, marker (medium)'
# prompt = 'matou sakura, carnelian, 1girl, purple hair, looking at viewer, medium breasts, hair between eyes, floating hair, purple eyes, long hair, long sleeves, collared shirt, brown vest, black skirt, white sleeves, school uniform, red ribbon, wide hips, lying, marker (medium)'
# prompt = 'willy wonka'
unprompts = [''] if cfg_enabled else []
prompts = [*unprompts, prompt]

n_iter = len(seeds) if seeds is not None else (
  20 if benchmarking or searching else 1
)
batch_size = 1
num_images_per_prompt = 1
width = 128 if smol else 512
height = int(width * 1.25) if tall else width
latent_channels = 4 # could use unet.in_channels, but we won't have that if loading a CoreML Unet
latents_shape = (batch_size * num_images_per_prompt, latent_channels, height // 8, width // 8)
with no_grad():
  tokens = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
  text_input_ids: Tensor = tokens.input_ids
  text_embeddings: Tensor = text_encoder(text_input_ids.to(device))[0]
  chunked = text_embeddings.chunk(text_embeddings.size(0))
  if cfg_enabled:
    uc, c = chunked
  else:
    uc = None
    c, = chunked

  batch_tic = time.perf_counter()
  for iter in range(n_iter):
    seed=seeds[iter] if seeds is not None else (
      (
        randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
      ) if searching else start_seed + iter
    )
    generator = Generator(device='cpu').manual_seed(seed)
    latents = randn(latents_shape, generator=generator, device='cpu', dtype=torch_dtype).to(device)

    tic = time.perf_counter()

    latents: Tensor = sampler(
      latents,
      cond=c,
      uncond=uc,
      cond_scale = 7.5 if cfg_enabled else 1.,
    )
    pil_images: List[Image.Image] = latents_to_pils(latents)
    print(f'generated {batch_size} images in {time.perf_counter()-tic} seconds')

    base_count = len(os.listdir(sample_path))
    for ix, image in enumerate(pil_images):
      image.save(os.path.join(sample_path, f"{base_count+ix:05}.{seed}.png"))

print(f'in total, generated {n_iter} batches of {num_images_per_prompt} images in {time.perf_counter()-batch_tic} seconds')