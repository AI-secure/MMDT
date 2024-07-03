from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch


class StableDiffusion2Client:
    def __init__(self, model_id):
        self.device = "cuda"
        assert model_id == "stabilityai/stable-diffusion-2"
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=self.scheduler, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(self.device)

    def generate(self, text, seed, save_path):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(text, generator=generator).images[0]
        image.save(save_path)
        return image
