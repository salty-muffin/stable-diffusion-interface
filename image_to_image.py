import os
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import torch
import click
from PIL import Image
from urllib.parse import urlparse


def is_local(url):
    url_parsed = urlparse(url)
    if url_parsed.scheme in ("file", ""):  # Possibly a local file
        return os.path.exists(url_parsed.path)
    return False


def parse_image(s: str):
    if is_local(s):
        return Image.open(s).convert("RGB")
    else:
        return load_image(s).convert("RGB")


# fmt: off
@click.command()
@click.argument("filepath", type=click.Path(dir_okay=False), metavar="PATH")
@click.option("--image", type=parse_image, help="image imput", required=True)
@click.option("--prompt", type=str, help="prompt for image generation", required=True)
# fmt: on
def image_to_image(filepath: str, image: Image, prompt: str):
    # load both refiner
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    # pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    # resize image
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    image = pipe(
        prompt,
        image=image,
    ).images[0]

    image.save(filepath)


if __name__ == "__main__":
    image_to_image()
