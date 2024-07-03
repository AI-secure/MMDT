from diffusers import DiffusionPipeline
import torch


class StableDiffusionxlClient:
    def __init__(self, model_id):
        self.device = "cuda"
        assert model_id == "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.pipe = self.pipe.to(self.device)

    def generate(self, text, seed, save_path):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(text, generator=generator).images[0]
        image.save(save_path)
        return image
