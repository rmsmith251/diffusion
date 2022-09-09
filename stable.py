from typing import List

import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

"""
Mostly copied from the Stable Diffusion example notebook found at
https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb
"""


class StableDiffusion:
    """
    Currently is tied pretty heavily to the official Stable Diffusion implementation. The goal is to
    untie it to be more customizable but maybe in a different class.

    The VAE and UNet will only work with versions of 'CompVis/stable-diffusion-v1-x'. The tokenizer and
    text encoder currently use CLIP and haven't been tested on any other encoders.

    Support for the different schedulers is planned.
    """

    def __init__(
        self,
        vae_name: str = "CompVis/stable-diffusion-v1-4",
        tokenizer_name: str = "openai/clip-vit-large-patch14",
        text_encoder_name: str = "openai/clip-vit-large-patch14",
        unet_name: str = "CompVis/stable-diffusion-v1-4",
        scheduler_name: str = "lmsd",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vae_name = vae_name
        self.tokenizer_name = tokenizer_name
        self.text_encoder_name = text_encoder_name
        self.unet_name = unet_name

        self.scheduler_key = {"lmsd": LMSDiscreteScheduler}
        self.scheduler_name = scheduler_name

        self.vae = None
        self.tokenizer = None
        self.text_encoder = None
        self.unet = None
        self.scheduler = None
        self.init_model()

    def init_model(self):
        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(
            self.vae_name,
            subfolder="vae",
            use_auth_token=True,
        )

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_name)
        self.text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_name)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(
            self.unet_name,
            subfolder="unet",
            use_auth_token=True,
        )

        # TODO: Generalize this to work with the other schedulers
        self.scheduler = self.scheduler_key[self.scheduler_name](
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)

    def save_mosaic(self, images: List, name: str, cols: int, rows: int):
        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(images):
            grid.paste(img, box=(i % cols * w, i // cols * h))

        grid.save(f"{name}.jpg")
        return grid

    def save(self, image, name: str):
        # breakpoint()
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        fn = name.replace(" ", "_")
        pil_images = [Image.fromarray(image) for image in images]
        if len(pil_images) > 1:
            if len(pil_images) < 4:
                rows = 1
                cols = len(pil_images)
            else:
                cols = len(pil_images) // 2
                rows = len(pil_images) - cols
            self.save_mosaic(images, fn, cols, rows)
        else:
            pil_images[0].save(f"{fn}.jpg")

        return pil_images

    def __call__(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 51,
        guidance_scale: float = 7.5,
        generator_seed: int = 32,
        batch_size: int = 1,
    ):
        generator = torch.manual_seed(generator_seed)
        text_input = self.tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(self.device)
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.sigmas[0]

        with autocast("cuda"):
            for i, t in tqdm(enumerate(self.scheduler.timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                sigma = self.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )["sample"]
                # breakpoint()

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, i, latents)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents)

        return image


if __name__ == "__main__":
    sd = StableDiffusion()
    keywords = ["photo", "photograph", "painting", "image"]
    while True:
        print(
            "Enter the text you would like to generate. If you want to exit, use CTRL+C or type 'exit'"
        )
        prompt = input("Text: ")
        if prompt.lower() == "exit":
            break
        steps = input("Enter number of steps or leave blank for default (51): ")
        seed = input("Enter the generator seed or leave blank for default (32): ")

        if not steps:
            steps = 51
        else:
            steps = int(steps)

        if not seed:
            seed = 32
        else:
            seed = int(seed)

        keys = []
        for keyword in keywords:
            if keyword not in prompt:
                keys.append(0)
            else:
                keys.append(1)
        if sum(keys) == 0:
            prompt = f"a photo of {prompt}"

        print(f"Generating image from prompt: {prompt}")
        sd.save(
            sd(
                prompt=prompt,
                num_inference_steps=51,
                batch_size=1,
                generator_seed=seed,
            ),
            prompt,
        )
