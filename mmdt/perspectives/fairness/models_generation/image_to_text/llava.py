from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image


class LlavaClient:
    def __init__(self, model_id):
        self.device = "cuda"
        assert model_id == "llava-hf/llava-v1.6-mistral-7b-hf"
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_id, cache_dir="/data1/common/mintong/cache")
        self.processor = LlavaNextProcessor.from_pretrained(model_id, cache_dir="/data1/common/mintong/cache")
        self.model.to(self.device)

    def generate(self, text, image_path, **kwargs):
        image = Image.open(image_path)
        prompt_template = "[INST] <image>\n{} [/INST]"
        text = prompt_template.format(text)
        inputs = self.processor(text, image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **kwargs)
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip().split('[/INST] ')[1]
        return generated_text
