from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image


class LlavaClient:
    def __init__(self, model_id):
        self.device = "cuda"
        self.model_id = model_id
        assert model_id in ["llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf"]
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_id)
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model.to(self.device)

    def generate(self, text, image_path, **kwargs):
        image = Image.open(image_path)
        if 'mistral' in self.model_id:
            prompt_template = "[INST] <image>\n{} [/INST]"
        else:
            prompt_template = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{} ASSISTANT:"
        text = prompt_template.format(text)
        inputs = self.processor(text, image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **kwargs)
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        if 'mistral' in self.model_id:
            generated_text = generated_text.split('[/INST]')[1]  # The model will usually output a space first. Here we do not do any post-process and keep the space.
        else:
            generated_text = generated_text.split('ASSISTANT:')[1]
        return generated_text
