from diffusers import AutoPipelineForText2Image
import torch


class KandinskyClient:
    def __init__(self, model_id):
        self.device = "cuda"
        assert model_id == "kandinsky-community/kandinsky-3"
        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, variant="fp16", torch_dtype=torch.float16)
        self.pipe = self.pipe.to(self.device)

    def generate(self, text, seed, save_path):  # This model is not deterministic
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(text, generator=generator).images[0]
        image.save(save_path)
        return image
