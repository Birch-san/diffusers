import sys, os
# put repository root on CWD so that local diffusers is used
sys.path.insert(1, f'{os.getcwd()}/src')
sys.path.insert(1, f'{os.getcwd()}/src/k-diffusion')

import torch
from torch import Generator, Tensor, randn, linspace, cumprod, no_grad
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from k_diffusion.external import DiscreteEpsDDPMDenoiser
from k_diffusion.sampling import sample_heun, get_sigmas_karras
from transformers import CLIPTextModel, PreTrainedTokenizer
from typing import TypeAlias, Union, List, Optional
from PIL import Image
import time

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

  def get_eps(self, *args, **kwargs) -> Tensor:
    out: UNet2DConditionOutput = self.inner_model(*args, **kwargs)
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
    cond_in = torch.cat([uncond, cond])
    del uncond, cond
    x_in = x.expand(cond_in.size(dim=0), -1, -1, -1)
    del x
    uncond, cond = self.denoiser(input=x_in, sigma=sigma, encoder_hidden_states=cond_in).chunk(cond_in.size(dim=0))
    del x_in, cond_in
    return uncond + (cond - uncond) * cond_scale

pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
  # "/Users/birch/git/stable-diffusion-v1-4",
  # 'hakurei/waifu-diffusion',
  'runwayml/stable-diffusion-v1-5',
  # revision='fp16',
  # torch_dtype=torch.float16,
  safety_checker=None,
)

device = torch.device('mps')
pipe = pipe.to(device)

text_encoder: CLIPTextModel = pipe.text_encoder
tokenizer: PreTrainedTokenizer = pipe.tokenizer
unet: UNet2DConditionModel = pipe.unet
vae: AutoencoderKL = pipe.vae

generator = Generator(device='cpu').manual_seed(68673924)

prompt = "masterpiece character portrait of a blonde girl, full resolution, 4k, mizuryuu kei, akihiko. yoshida, Pixiv featured, baroque scenic, by artgerm, sylvain sarrailh, rossdraws, wlop, global illumination, vaporwave"
prompts = ['', prompt]

batch_size = 1
num_images_per_prompt = 1
width = 512
height = 512
latents_shape = (batch_size * num_images_per_prompt, pipe.unet.in_channels, height // 8, width // 8)
with no_grad():
  tokens = pipe.tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
  text_input_ids: Tensor = tokens.input_ids
  text_embeddings: Tensor = text_encoder(tokens.input_ids.to(device))[0]
  uc, c = text_embeddings.chunk(text_embeddings.size(0))
  latents = randn(latents_shape, generator=generator, device='cpu', dtype=unet.dtype).to(device)

  alphas_cumprod: Tensor = get_alphas_cumprod(get_alphas(get_betas(device=device))).to(dtype=unet.dtype)
  unet_k_wrapped = DiffusersSDDenoiser(pipe.unet, alphas_cumprod)
  denoiser = CFGDenoiser(unet_k_wrapped)

  sigmas: Tensor = get_sigmas_karras(
    n=15,
    sigma_max=unet_k_wrapped.sigma_max,
    sigma_min=unet_k_wrapped.sigma_min,
    rho=7.,
    device=device,
  ).to(unet.dtype)
  extra_args = {
    'cond': c,
    'uncond': uc,
    'cond_scale': 7.5,
  }
  tic = time.perf_counter()
  latents: Tensor = sample_heun(
    denoiser,
    latents * sigmas[0],
    sigmas,
    extra_args=extra_args,
  )
  latents = 1 / 0.18215 * latents

  images: Tensor = vae.decode(latents).sample

  images = (images / 2 + 0.5).clamp(0, 1)

  # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
  images = images.cpu().permute(0, 2, 3, 1).float().numpy()
  images = (images * 255).round().astype("uint8")

toc = time.perf_counter()

sample_path="out"
base_count = len(os.listdir(sample_path))
pil_images: List[Image.Image] = [Image.fromarray(image) for image in images]
for ix, image in enumerate(pil_images):
  image.save(os.path.join(sample_path, f"{base_count+ix:05}.png"))

print(f'in total, generated {batch_size} batches of {num_images_per_prompt} images in {toc-tic} seconds')