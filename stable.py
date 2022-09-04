import torch
from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

"""
Mostly copied from the Stable Diffusion example notebook found at
https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb
"""


class StableDiffusion:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vae = None
        self.tokenizer = None
        self.text_encoder = None
        self.unet = None
        self.scheduler = None
        self.init_model()

    def init_model(self):
        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            use_auth_token=True,
        )

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="unet",
            use_auth_token=True,
        )

        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)

    def save(self, image, name: str):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        fn = name.replace(" ", "_")
        pil_images = [
            Image.fromarray(image).save(f"{fn}-{i}.jpg")
            for i, image in enumerate(images)
        ]
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
            [prompt],
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
    prompt = "a painting of a cozy candlelit brewery in the winter"

    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion

    num_inference_steps = 100  # Number of denoising steps

    guidance_scale = 7.5  # Scale for classifier-free guidance

    generator = torch.manual_seed(
        32
    )  # Seed generator to create the inital latent noise

    batch_size = 1
    sd = StableDiffusion()
    images = sd(prompt=prompt)
    sd.save(images, prompt)
    breakpoint()
