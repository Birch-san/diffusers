import sys, os
# put repository root on CWD so that local diffusers is used
sys.path.insert(1, f'{os.getcwd()}/src')

import torch
# from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, KarrasVeScheduler
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from PIL import Image
import time

# eds = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
# kve = KarrasVeScheduler(
#     sigma_max=14.6146,
#     # sigma_min=0.0936,
#     sigma_min=0.0292,
#     s_churn=0.
# )
lms = LMSDiscreteScheduler(
  beta_start=0.00085,
  beta_end=0.012,
  beta_schedule="scaled_linear"
)

pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
  # "/Users/birch/git/stable-diffusion-v1-4",
  # 'hakurei/waifu-diffusion',
  'runwayml/stable-diffusion-v1-5',
  # revision='fp16',
  # torch_dtype=torch.float16,
  safety_checker=None,
)

tic = time.perf_counter()
pipe = pipe.to("mps")

prompt = "masterpiece character portrait of a blonde girl, full resolution, 4k, mizuryuu kei, akihiko. yoshida, Pixiv featured, baroque scenic, by artgerm, sylvain sarrailh, rossdraws, wlop, global illumination, vaporwave"
generator = torch.Generator(device="cpu").manual_seed(68673924)
image: Image.Image = pipe(
	prompt,
	# guidance_scale=1.,
	generator=generator,
  scheduler=lms,
  # scheduler=kve,
  num_inference_steps=8,
).images[0]

sample_path="out"
base_count = len(os.listdir(sample_path))
image.save(os.path.join(sample_path, f"{base_count:05}.png"))
toc = time.perf_counter()
print(f'in total, generated 1 image in {toc-tic} seconds')