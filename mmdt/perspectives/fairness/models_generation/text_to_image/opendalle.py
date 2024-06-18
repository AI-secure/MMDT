from diffusers import AutoPipelineForText2Image
import torch


class OpenDalleClient:
    def __init__(self, model_id, device):
        self.device = device
        assert model_id == "dataautogpt3/OpenDalleV1.1"
        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, text):
        image = self.pipe(text, generator=self.generator, num_images_per_prompt=1).images
        return image
