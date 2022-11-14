import sys, os
# put repository root on CWD so that local diffusers is used
sys.path.insert(1, f'{os.getcwd()}/src')
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
from apple.multihead_attention import MultiHeadAttention
from apple.layer_norm import LayerNormANE
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from diffusers.models.attention import BasicTransformerBlock, CrossAttention, MultiheadAttention, to_mha
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from k_diffusion.external import DiscreteEpsDDPMDenoiser
from k_diffusion.sampling import get_sigmas_karras, sample_heun, sample_dpmpp_2s_ancestral, BrownianTreeNoiseSampler, sample_dpm_adaptive, sample_dpmpp_2m
from transformers import CLIPTextModel, PreTrainedTokenizer
from typing import TypeAlias, Union, List, Optional, Callable, TypedDict
from PIL import Image
import time

import coremltools as ct
from pathlib import Path
import torch as th
from coremltools.models import MLModel

# 5 DPMSolver++ steps, limited sigma schedule
prototyping = True
# 128x128 images, for even more extreme prototyping
smol = prototyping and False
cfg_enabled = True
benchmarking = not prototyping and False
coreml_sampler = True
saving_coreml_model = False
# we shouldn't attempt to load CoreML model during the same run as when saving it, because sampling+VAE+encoder will be on-CPU and wrong dtype
loading_coreml_model = not saving_coreml_model and False
loading_coreml_ane = loading_coreml_model and False
using_ane_self_attention = False
using_torch_self_attention = False
replacing_self_attention = using_ane_self_attention or using_torch_self_attention
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
      steps=5
      sigma_max=torch.tensor(7.0796, device=alphas_cumprod.device, dtype=alphas_cumprod.dtype)
      sigma_min=torch.tensor(0.0936, device=alphas_cumprod.device, dtype=alphas_cumprod.dtype)
      rho=9.
    else:
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

revision=None
torch_dtype=None
if saving_coreml_model:
  # gotta trace model on-CPU, and only float32 is supported
  device = torch.device('cpu')
  # could fp16 model revision make it trace any faster? probably not
else:
  if half:
    revision='fp16'
    torch_dtype=torch.float16
  device = torch.device('mps')

coreml_modules = ['unet'] if loading_coreml_model else []
omit_modules = ['safety_checker', *coreml_modules]

pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
  # "/Users/birch/git/stable-diffusion-v1-4",
  'hakurei/waifu-diffusion',
  # 'runwayml/stable-diffusion-v1-5',
  revision=revision,
  torch_dtype=torch_dtype,
  **{ key: None for key in omit_modules }
)

pipe = pipe.to(device)

text_encoder: CLIPTextModel = pipe.text_encoder
tokenizer: PreTrainedTokenizer = pipe.tokenizer
if not loading_coreml_model:
  unet: UNet2DConditionModel = pipe.unet
  unet.eval()
  sampler = Sampler(unet)
vae: AutoencoderKL = pipe.vae

class AMHADelegator(MultiHeadAttention):
  def __init__(
    self,
    query_dim: int,
    cross_attention_dim: Optional[int] = None,
    heads: int = 8,
    dim_head: int = 64,
    dropout: float = 0.0,
    bias=False,
  ):
    inner_dim = dim_head * heads
    cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
    # debug
    self.inner_dim = inner_dim
    self.embed_dim = inner_dim
    self.dim_head = dim_head
    self.heads = heads
    self.cross_attention_dim = cross_attention_dim
    super().__init__(
      embed_dim=inner_dim,
      n_head=heads,
      dropout=dropout,
      bias=bias,
      batch_first=True,
      d_qk=cross_attention_dim,
      d_v=cross_attention_dim,
      # d_out=cross_attention_dim,
    )
  
  def forward(self, hidden_states: Tensor, context: Optional[Tensor]=None) -> Tensor:
    # TODO: we're probably undoing a permute from upstream, so just eliminate that
    # hidden_states = hidden_states.permute(0,2,1).unsqueeze(2)
    context = context.transpose(2,1).unsqueeze(2) if context is not None else hidden_states
    # hidden_states = hidden_states.unsqueeze(2)
    # context = context.unsqueeze(2) if context is not None else hidden_states
    out, _ = super().forward(
      q=hidden_states,
      k=context,
      v=context,
    )
    out = out.squeeze(2).transpose(2,1)
    return out

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
    conv.bias.data = torch.zeros(linear.out_features, device=conv.weight.data.device, dtype=conv.weight.data.dtype)
  else:
    conv.bias.data = linear.bias.data


def to_amha(ca: CrossAttention) -> AMHADelegator:
  bias = ca.to_k.bias is not None
  assert bias == False
  mha = AMHADelegator(
    query_dim=ca.to_q.in_features,
    cross_attention_dim=ca.to_k.in_features,
    heads=ca.heads,
    dim_head=ca.to_q.out_features//ca.heads,
    dropout=ca.to_out[1].p,
    bias=bias,
  )
  initialize_from_linear(mha.q_proj, ca.to_q)
  initialize_from_linear(mha.k_proj, ca.to_k)
  initialize_from_linear(mha.v_proj, ca.to_v)
  initialize_from_linear(mha.out_proj, ca.to_out[0])
  return mha

def to_aln(ln: nn.LayerNorm) -> LayerNormANE:
  dim, = ln.normalized_shape
  aln = LayerNormANE(dim)
  aln.weight.data = ln.weight.data
  aln.bias.data = ln.bias.data / ln.weight.data
  return aln

def replace_attention(module: nn.Module) -> None:
  assert using_torch_self_attention or using_ane_self_attention
  for name, m in module.named_children():
    if isinstance(m, CrossAttention):
      # is self-attention?
      if m.to_q.in_features == m.to_k.in_features:
        if using_torch_self_attention:
          mha: MultiheadAttention = to_mha(m)
        elif using_ane_self_attention:
          mha: AMHADelegator = to_amha(m)
        setattr(module, name, mha)
    elif using_ane_self_attention and isinstance(m, BasicTransformerBlock):
      aln: LayerNormANE = to_aln(m.norm1)
      setattr(m, 'norm1', aln)

if replacing_self_attention and not loading_coreml_model:
  unet.apply(replace_attention)

class Undictifier(nn.Module):
  model: nn.Module
  def __init__(self, model: nn.Module):
    super().__init__()
    self.model = model
  def forward(self, *args, **kwargs): 
    return self.model(*args, **kwargs)["sample"]

def convert_unet(pt_model: UNet2DConditionModel, out_name: str) -> None:
  from coremltools.converters.mil import Builder as mb
  from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op, _TORCH_OPS_REGISTRY
  import coremltools.converters.mil.frontend.torch.ops as cml_ops

  orig_baddbmm = torch.baddbmm
  def fake_baddbmm(_: Tensor, batch1: Tensor, batch2: Tensor, beta: float, alpha: float):
    return torch.bmm(batch1, batch2) * alpha
  torch.baddbmm = fake_baddbmm

  if "broadcast_to" in _TORCH_OPS_REGISTRY: del _TORCH_OPS_REGISTRY["broadcast_to"]
  @register_torch_op
  def broadcast_to(context, node): return cml_ops.expand(context, node)

  if "gelu" in _TORCH_OPS_REGISTRY: del _TORCH_OPS_REGISTRY["gelu"]
  @register_torch_op
  def gelu(context, node): context.add(mb.gelu(x=context[node.inputs[0]], name=node.name))

  print("tracing")
  b = 1 if one_at_a_time or not cfg_enabled else 2
  latents_shape = (b, 4, 64, 64)
  timestep_shape = (1,)
  embeddings_shape = (b, 77, 768)
  with no_grad(): # not sure whether no_grad is necessary but can't hurt
    trace = torch.jit.trace(
      Undictifier(pt_model),
      (
        torch.zeros(*latents_shape),
        torch.zeros(*timestep_shape),
        torch.zeros(*embeddings_shape)
      ),
      strict=False,
      check_trace=False
    )

  print("converting")
  cm_model = ct.convert(
    trace, 
    inputs=[
      ct.TensorType(shape=latents_shape),
      ct.TensorType(shape=timestep_shape),
      ct.TensorType(shape=embeddings_shape)
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    skip_model_load=True
  )

  print(f"saving to '{out_name}'")
  cm_model.save(f"{out_name}")
  print(f"saved")

  torch.baddbmm = orig_baddbmm

def convert_sampler(pt_model: Sampler, out_name: str) -> None:
  from coremltools.converters.mil import Builder as mb
  from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op, _TORCH_OPS_REGISTRY
  import coremltools.converters.mil.frontend.torch.ops as cml_ops

  orig_baddbmm = torch.baddbmm
  def fake_baddbmm(_: Tensor, batch1: Tensor, batch2: Tensor, beta: float, alpha: float):
    return torch.bmm(batch1, batch2) * alpha
  torch.baddbmm = fake_baddbmm

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
      Undictifier(pt_model),
      (
        torch.zeros(*latents_shape),
        torch.zeros(*embedding_shape),
        torch.zeros(*embedding_shape),
        torch.zeros(*cond_scale_shape),
      ),
      strict=False,
      check_trace=False
    )

  print("converting")
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

  print(f"saving to '{out_name}'")
  cm_model.save(f"{out_name}")
  print(f"saved")

  torch.baddbmm = orig_baddbmm

mlp_name='sampler.mlpackage' if coreml_sampler else 'unet.mlpackage'
if saving_coreml_model:
  if Path(mlp_name).exists():
    print(f"CoreML model '{mlp_name}' already exists")
  else:
    print("generating CoreML model")
    convert_sampler(sampler, mlp_name) if coreml_sampler else convert_unet(unet, mlp_name) 
    print(f"saved CoreML model '{mlp_name}'")
  # we refrain from loading the model and continuing, because our Unet, etc are on CPU/float32
  sys.exit()

class UNetWrapper:
  ml_model: MLModel
  def __init__(self, ml_model: MLModel):
    self.ml_model = ml_model
    self.device = device

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
      "sample": sample.to(dtype=torch.float16, device='cpu').numpy(),
      "timestep": timestep.to(dtype=torch.float16, device='cpu').int().numpy(),
      "input_35": encoder_hidden_states.to(dtype=torch.float16, device='cpu').numpy(),
    }
    prediction = self.ml_model.predict(args)
    for v in prediction.values():
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
    args = {
      "latents": latents.to(dtype=torch.float16, device='cpu').numpy(),
      "cond": cond.to(dtype=torch.float16, device='cpu').numpy(),
      "uncond": uncond.to(dtype=torch.float16, device='cpu').numpy(),
      "cond_scale": cond_scale.to(dtype=torch.float16, device='cpu').numpy(),
    }
    prediction = self.ml_model.predict(args)
    for v in prediction.values():
      sample=torch.tensor(v, dtype=dtype, device=device)
      return sample

if loading_coreml_model:
  compute_units=ct.ComputeUnit.ALL if loading_coreml_ane else ct.ComputeUnit.CPU_AND_GPU
  print(f"loading CoreML model '{mlp_name}'")
  assert Path(mlp_name).exists()
  cm_model = MLModel(mlp_name, compute_units=compute_units, dtype=torch.float16)
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
prompt = 'artoria pendragon (fate), carnelian, 1girl, general content, upper body, white shirt, blonde hair, looking at viewer, medium breasts, hair between eyes, floating hair, green eyes, blue ribbon, long sleeves, light smile, hair ribbon, watercolor (medium), traditional media'
# prompt = 'rem (re:zero), carnelian, 1girl, upper body, blue hair, looking at viewer, medium breasts, hair between eyes, floating hair, blue eyes, blue hair, short hair, roswaal mansion maid uniform, detached sleeves, detached collar, ribbon trim, maid headdress, x hair ornament, sunset, marker (medium)'
# prompt = 'matou sakura, carnelian, 1girl, purple hair, looking at viewer, medium breasts, hair between eyes, floating hair, purple eyes, long hair, long sleeves, collared shirt, brown vest, black skirt, white sleeves, school uniform, red ribbon, wide hips, lying, marker (medium)'
# prompt = 'willy wonka'
unprompts = [''] if cfg_enabled else []
prompts = [*unprompts, prompt]

n_iter = 20 if benchmarking else 1
batch_size = 1
num_images_per_prompt = 1
width = 128 if smol else 512
height = width
latent_channels = 4 # could use unet.in_channels, but we won't have that if loading a CoreML Unet
latents_shape = (batch_size * num_images_per_prompt, latent_channels, height // 8, width // 8)
with no_grad():
  tokens = pipe.tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
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
    # seed=68673924
    seed=2178792735
    generator = Generator(device='cpu').manual_seed(seed+iter)
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
      image.save(os.path.join(sample_path, f"{base_count+ix:05}.{seed+iter}.png"))

print(f'in total, generated {n_iter} batches of {num_images_per_prompt} images in {time.perf_counter()-batch_tic} seconds')