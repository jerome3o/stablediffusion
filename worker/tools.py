import argparse
import os
from contextlib import nullcontext
from itertools import islice

import cv2
import numpy as np
import torch
from einops import rearrange
from imwatermark import WatermarkEncoder
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config

torch.set_grad_enabled(False)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


class TextToImageConfig(BaseModel):
    prompt: str
    output_dir: str
    steps: int = 50  # number of ddim sampling steps
    ddim_eta: float = 0.0  # ddim eta, eta=0.0 corresponds to deterministic sampling
    n_iter: int = 3  # sample this often
    height: int = 512  # image height, in pixel space
    width: int = 512  # image width, in pixel space
    latent_channels: int = 4  # latent channels
    downsampling_factor: int = 8  # downsampling factor, most often 8 or 16
    n_samples: int = (
        3  # how many samples to produce for each given prompt. A.k.a batch size
    )
    n_rows: int = 0  # rows in the grid (default: n_samples)
    scale: float = 9.0  # unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
    config: str = "configs/stable-diffusion/v2-inference.yaml"  # path to config which constructs model
    seed: int = 42  # the seed (for reproducible sampling)

    # choices=["full", "autocast"],
    precision: str = autocast  # evaluate at this precision
    repeat: int = 1  # repeat each prompt in file this often
    ckpt: str = "TODO"  # path to checkpoint of model
    plms: bool = True  # use plms sampling
    dpm: bool = True  # use DPM (2) sampler
    fixed_code: bool = (
        True  # if enabled, uses the same starting code across all samples
    )


def main(config: TextToImageConfig):
    seed_everything(config.seed)

    omega_config = OmegaConf.load(f"{config.config}")
    model = load_model_from_config(omega_config, f"{config.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if config.plms:
        sampler = PLMSSampler(model)
    elif config.dpm:
        sampler = DPMSolverSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(config.output_dir, exist_ok=True)
    outpath = config.output_dir

    batch_size = config.n_samples
    n_rows = config.n_rows if config.n_rows > 0 else batch_size
    prompt = config.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if config.fixed_code:
        start_code = torch.randn(
            [config.n_samples, config.latent_channels, config.height // config.downsampling_factor, config.width // config.downsampling_factor],
            device=device,
        )

    precision_scope = autocast if config.precision == "autocast" else nullcontext
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        all_samples = list()
        for n in trange(config.n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                uc = None
                if config.scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)
                shape = [config.latent_channels, config.height // config.downsampling_factor, config.width // config.downsampling_factor]
                samples, _ = sampler.sample(
                    S=config.steps,
                    conditioning=c,
                    batch_size=config.n_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=config.scale,
                    unconditional_conditioning=uc,
                    eta=config.ddim_eta,
                    x_T=start_code,
                )

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255.0 * rearrange(
                        x_sample.cpu().numpy(), "c h w -> h w c"
                    )
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1
                    sample_count += 1

                all_samples.append(x_samples)

        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, "n b c h w -> (n b) c h w")
        grid = make_grid(grid, nrow=n_rows)

        # to image
        grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
        grid = Image.fromarray(grid.astype(np.uint8))
        grid.save(os.path.join(outpath, f"grid-{grid_count:04}.png"))
        grid_count += 1

    print(
        f"Your samples are ready and waiting for you here: \n{outpath} \n" f" \nEnjoy."
    )


if __name__ == "__main__":
    config = TextToImageConfig.parse_file("scratch/text_config.json")
    main(config)
