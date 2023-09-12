from diffusers import DiffusionPipeline
import torch
import click


# fmt: off
@click.command()
@click.argument("filepath", type=click.Path(dir_okay=False), metavar="PATH")
@click.option("--prompt", type=str, help="prompt for image generation", required=True)
@click.option("--steps", type=int, help="generation steps", default=50)
# fmt: on
def generate_image(filepath: str, prompt: str, steps: int):
    # load both base & refiner
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    base.to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    # refiner.to("cuda")
    refiner.enable_model_cpu_offload()

    # base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
    # refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

    # Define what % of steps to be run on each experts (80/20) here
    high_noise_frac = 0.8

    # run both experts
    latent = base(
        prompt=prompt,
        num_inference_steps=steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images

    image = refiner(
        prompt=prompt,
        num_inference_steps=steps,
        denoising_start=high_noise_frac,
        image=latent,
    ).images[0]

    image.save(filepath)


if __name__ == "__main__":
    generate_image()
