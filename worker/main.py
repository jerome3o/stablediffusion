from pydantic import BaseModel
from typing import Any, List
from pathlib import Path
import json
import torch
import os

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
from matplotlib import pyplot as plt
import numpy
from torchvision import transforms as tfms
import numpy as np

# For video display:
from IPython.display import HTML
from base64 import b64encode

_DEFAULT_HEIGHT = 512
_DEFAULT_WIDTH = 768
_DEFAULT_NUM_INFERENCE_STEPS = 50
_DEFAULT_GUIDANCE_SCALE = 7.5
_DEFAULT_BATCH_SIZE = 1


# Using torchvision.transforms.ToTensor
to_tensor_tfm = tfms.ToTensor()
_token = os.environ["HUGGING_FACE_TOKEN"]


def initialise_models(token):
    # Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=token
    )

    # Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=token
    )

    # The noise scheduler
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    return vae, tokenizer, text_encoder, unet, scheduler


def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(
            to_tensor_tfm(input_im).unsqueeze(0).to(torch_device) * 2 - 1
        )  # Note scaling
    return 0.18215 * latent.mode()  # or .mean or .sample


def latents_to_pil(latents):

    # batch of latents -> list of images
    latents = (1 / 0.18215) * latents

    with torch.no_grad():
        image = vae.decode(latents)[0]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


# potentially list of str needed?


def prep_text(prompt: str, tokenizer: Any, max_length: int = None):
    max_length = max_length or tokenizer.model_max_length

    if isinstance(prompt, str):
        prompt = [prompt]

    # Prep text
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    max_length = text_input.input_ids.shape[-1]
    return text_embeddings, max_length


def generate_image(
    text_embeddings,  # tensor?
    loading_bar=True,
    generator=None,
    height=_DEFAULT_HEIGHT,
    width=_DEFAULT_WIDTH,
    num_inference_steps=_DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale=_DEFAULT_GUIDANCE_SCALE,
    batch_size=_DEFAULT_BATCH_SIZE,
    latents=None,
):
    if generator is None:
        generator = torch.manual_seed(42)

    # Prep Scheduler
    scheduler.set_timesteps(num_inference_steps)

    # Prep latents
    if latents is None:
        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )

    latents = latents.to(torch_device)
    latents = latents * scheduler.sigmas[0]  # Need to scale to match k

    loading_bar_function = tqdm if loading_bar else lambda x: x

    # Loop
    with autocast("cuda"):
        for i, t in loading_bar_function(enumerate(scheduler.timesteps)):

            # expand the latents if we are doing classifier-free guidance
            # to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                )["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

    return latents_to_pil(latents)[0]


def get_output_dir(base: Path):
    base.mkdir(exist_ok=True, parents=True)
    index = max(map(lambda p: int(p.stem), base.glob("[0-9]*")), default=0) + 1
    interp_output = base / str(index)
    interp_output.mkdir()
    return interp_output


from typing import Tuple, List
from pydantic import BaseModel


class Config(BaseModel):
    prompt_list: List[str]

    n_steps_latent: int = 60
    n_steps_prompt: int = 360


class _Step(BaseModel):
    # (Prompt, Weight)
    # [("prompt_1", 0.5), ...]
    prompts: List[Tuple[str, float]]

    # (Seed integer, Weight)
    latents: List[Tuple[int, float]]


def generate_steps(config: Config):
    latent_fracs = np.linspace(0, 1, config.n_steps_latent)
    prompt_fracs = np.linspace(0, 1, config.n_steps_prompt)

    c = 0
    steps = []

    for prompt_i, prompt in enumerate(config.prompt_list[:-1]):
        prompt_2 = config.prompt_list[prompt_i + 1]
        for prompt_frac in prompt_fracs:

            # calc latent seeds
            latent_frac = latent_fracs[c % config.n_steps_latent]
            seed_1 = c // config.n_steps_latent
            seed_2 = seed_1 + 1

            steps.append(
                _Step(
                    prompts=[(prompt, 1 - prompt_frac), (prompt_2, prompt_frac)],
                    latents=[(seed_1, 1 - latent_frac), (seed_2, latent_frac)],
                )
            )

            c += 1
    return steps


def generate_latent(
    latent_spec: List[Tuple[int, float]],
    height: int = _DEFAULT_HEIGHT,
    width: int = _DEFAULT_WIDTH,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> torch.Tensor:
    latent = sum(
        [
            frac
            * torch.randn(
                (batch_size, unet.in_channels, height // 8, width // 8),
                generator=torch.manual_seed(seed),
            )
            for seed, frac in latent_spec
        ]
    )
    latent -= latent.mean()
    latent /= latent.std()
    return latent


def generate_images(
    steps: List[_Step],
    output_dir: str,
    height: int = _DEFAULT_HEIGHT,
    width: int = _DEFAULT_WIDTH,
    num_inference_steps: int = _DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = _DEFAULT_GUIDANCE_SCALE,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    scout_steps: int = 20,
):
    output_dir = Path(output_dir)
    uncond_embeddings, _ = prep_text([""], tokenizer)

    all_prompts = list({prompt for step in steps for prompt, _ in step.prompts})
    embeddings = {
        prompt: prep_text([prompt], tokenizer)[0] for prompt in list(set(all_prompts))
    }
    enumerated_steps = list(enumerate(steps))
    if len(steps) > scout_steps:
        scout_step = len(enumerated_steps) // scout_steps

        ordered_steps = enumerated_steps[::scout_step] + [
            s for s in enumerated_steps if (s[0] % scout_step) != 0
        ]
    else:
        ordered_steps = enumerated_steps 


    for c, step in tqdm(ordered_steps):

        im_file = output_dir / f"image_{c:05d}.png"

        if im_file.exists():
            continue

        latent = generate_latent(step.latents)

        embedding = sum([embeddings[prompt] * frac for prompt, frac in step.prompts])
        embedding = torch.cat([uncond_embeddings, embedding])

        im = generate_image(
            embedding,
            latents=latent,
            loading_bar=False,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            batch_size=batch_size,
        )
        im.save(im_file)

def generate_single_step(prompt: str, seed: int = 5):
    return _Step(prompts=[(prompt, 1)], latents=[(seed, 1)])


def get_output_dir(base: Path):
    base.mkdir(exist_ok=True, parents=True)
    index = max(map(lambda p: int(p.stem), base.glob("[0-9]*")), default=0) + 1
    interp_output = base / str(index)
    interp_output.mkdir()
    return interp_output


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Set device
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    vae, tokenizer, text_encoder, unet, scheduler = initialise_models(_token)

    # To the GPU we go!
    vae = vae.to(torch_device)
    text_encoder = text_encoder.to(torch_device)
    unet = unet.to(torch_device)
    
    output_dir = "outputs/"
    output_dir = Path(output_dir)
    current_output = get_output_dir(output_dir / "tests")

    step = generate_single_step("a penguin with three eyes")
    generate_images([step], current_output)
