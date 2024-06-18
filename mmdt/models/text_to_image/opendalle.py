from diffusers import AutoPipelineForText2Image
import torch


class OpenDalleClient:
    def __init__(self, model_id):
        self.device = "cuda"
        assert model_id == "dataautogpt3/OpenDalleV1.1"
        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(self.device)

    def generate(self, text, seed, save_path):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(text, generator=generator).images[0]
        image.save(save_path)
        return image
