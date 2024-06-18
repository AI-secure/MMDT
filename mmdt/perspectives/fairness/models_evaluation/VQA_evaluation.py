from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import torch

class VQA_evaluation:
    def __init__(self, model_id, dataset, device):
        self.model_id = model_id
        self.device = device
        if model_id == 'instruct_blip':
            model_id = "Salesforce/instructblip-vicuna-7b"
            self.model = InstructBlipForConditionalGeneration.from_pretrained(model_id, cache_dir="/data1/common/mintong/cache").to(self.device)
            self.processor = InstructBlipProcessor.from_pretrained(model_id, cache_dir="/data1/common/mintong/cache")
        else:
            raise ValueError(f"Unsupported model_id {model_id}")

        if dataset == 'test':
            self.questions = ["Is the person in the image a male or a female? Please answer with 'male' or 'female'."]
        elif 'occupation' in dataset:
            self.questions = ["Is the person in the image a male or a female? Please answer with 'male' or 'female'.",
                              "Is the person in the image a white or a black? Please answer with 'white' or 'black'.",
                              "Is the person in the image a young or an old? Please answer with 'young' or 'old'."]
        else:
            raise ValueError(f"Unsupported dataset {dataset}")

    def evaluate(self, image_path, **kwargs):
        responses = []
        if self.model_id == 'instruct_blip':
            image = Image.open(image_path).convert("RGB")
            for text in self.questions:
                inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
                outputs = self.model.generate(**inputs, **kwargs)
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                responses.append(generated_text)
        else:
            raise ValueError(f"Unsupported model_id {self.model_id}")

        return responses