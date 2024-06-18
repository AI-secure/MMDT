from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image


class InstructBLIPClient:
    def __init__(self, model_id):
        self.device = "cuda"
        assert model_id == "Salesforce/instructblip-vicuna-7b"
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_id)
        self.processor = InstructBlipProcessor.from_pretrained(model_id)
        self.model.to(self.device)

    def generate(self, text, image_path, **kwargs):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **kwargs)
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return generated_text
