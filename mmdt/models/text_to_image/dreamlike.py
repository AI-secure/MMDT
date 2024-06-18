from diffusers import StableDiffusionPipeline
import torch


class DreamlikeClient:
    def __init__(self, model_id):
        self.device = "cuda"
        assert model_id == "dreamlike-art/dreamlike-photoreal-2.0"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(self.device)

    def generate(self, text, seed, save_path):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(text, generator=generator).images[0]
        image.save(save_path)
        return image
