from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch


class QwenClient:
    def __init__(self, model_id):
        self.device = "cuda"
        assert model_id == "Qwen/Qwen-VL-Chat"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(self.device)

    def generate(self, text, image_path, **kwargs):
        self.model.generation_config.update(**kwargs)
        query = self.tokenizer.from_list_format([
            {'image': image_path},
            {'text': text},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response.strip()
