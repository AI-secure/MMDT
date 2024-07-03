from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch


class IFClient:
    def __init__(self, model_id):
        self.device = "cuda"
        assert model_id == "DeepFloyd/IF-I-M-v1.0"
        self.stage_1 = DiffusionPipeline.from_pretrained(model_id, variant="fp16", torch_dtype=torch.float16)

        stage_2_id = "DeepFloyd/IF-II-M-v1.0"
        self.stage_2 = DiffusionPipeline.from_pretrained(stage_2_id, text_encoder=None, variant="fp16", torch_dtype=torch.float16)

        stage_3_id = "stabilityai/stable-diffusion-x4-upscaler"
        safety_modules = {"feature_extractor": self.stage_1.feature_extractor,
                          "safety_checker": self.stage_1.safety_checker,
                          "watermarker": self.stage_1.watermarker}
        self.stage_3 = DiffusionPipeline.from_pretrained(stage_3_id, **safety_modules, torch_dtype=torch.float16)

        self.stage_1 = self.stage_1.to(self.device)
        self.stage_2 = self.stage_2.to(self.device)
        self.stage_3 = self.stage_3.to(self.device)

    def generate(self, text, seed, save_path):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(text)
        image = self.stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator,
                        output_type="pt").images
        image = self.stage_2(
            image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator,
            output_type="pt"
        ).images
        image = self.stage_3(prompt=text, image=image, generator=generator, noise_level=100).images[0]
        image.save(save_path)
        return image
