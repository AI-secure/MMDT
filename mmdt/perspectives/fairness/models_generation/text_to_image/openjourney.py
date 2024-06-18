from diffusers import StableDiffusionPipeline
import torch


class OpenjourneyClient:
    def __init__(self, model_id, device):
        self.device = device
        assert model_id == "prompthero/openjourney-v4"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, text):
        image = self.pipe(text, generator=self.generator, num_images_per_prompt=1).images
        return image
