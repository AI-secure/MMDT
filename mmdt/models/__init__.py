import os


class Image2TextClient:
    def __init__(self, model_id):
        if model_id == "models/gemini-1.5-pro-001":
            from .image_to_text.gemini import GeminiClient
            self.client = GeminiClient(model_id)
        elif model_id in ["gpt-4-vision-preview", "gpt-4o-2024-05-13", "gpt-4o"]:
            from .image_to_text.gpt4v import GPT4VClient
            self.client = GPT4VClient(model_id)
        elif model_id == "claude-3-opus-20240229":
            from .image_to_text.claude3 import ClaudeClient
            self.client = ClaudeClient(model_id)
        elif model_id in ["llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf"]:
            from .image_to_text.llava import LlavaClient
            self.client = LlavaClient(model_id)
        elif model_id == "Salesforce/instructblip-vicuna-7b":
            from .image_to_text.instructblip import InstructBLIPClient
            self.client = InstructBLIPClient(model_id)
        elif model_id == "Qwen/Qwen-VL-Chat":
            from .image_to_text.qwen import QwenClient
            self.client = QwenClient(model_id)
        elif model_id in ['OpenGVLab/InternVL2-8B', 'OpenGVLab/Mini-InternVL-Chat-4B-V1-5']:
            from .image_to_text.internvl import InternVLClient
            self.client = InternVLClient(model_id)
        else:
            raise Exception(f"Model {model_id} is not supported.")

    def generate(self, text, image_path, **kwargs):
        return self.client.generate(text, image_path, **kwargs)
    
    def generate_multiple_img(self, text, image_path_list, **kwargs):
        return self.client.generate_multiple_img(text, image_path_list, **kwargs)


class Text2ImageClient:
    def __init__(self, model_id):
        if model_id in ["dall-e-2", "dall-e-3"]:
            from .text_to_image.dalle import DalleClient
            self.client = DalleClient(model_id)
        elif model_id == "DeepFloyd/IF-I-M-v1.0":
            from .text_to_image.deepfloyd_if import IFClient
            self.client = IFClient(model_id)
        elif model_id == "dreamlike-art/dreamlike-photoreal-2.0":
            from .text_to_image.dreamlike import DreamlikeClient
            self.client = DreamlikeClient(model_id)
        elif model_id == "kandinsky-community/kandinsky-3":
            from .text_to_image.kandinsky import KandinskyClient
            self.client = KandinskyClient(model_id)
        elif model_id == "dataautogpt3/OpenDalleV1.1":
            from .text_to_image.opendalle import OpenDalleClient
            self.client = OpenDalleClient(model_id)
        elif model_id == "prompthero/openjourney-v4":
            from .text_to_image.openjourney import OpenjourneyClient
            self.client = OpenjourneyClient(model_id)
        elif model_id == "stabilityai/stable-diffusion-2":
            from .text_to_image.stablediffusion2 import StableDiffusion2Client
            self.client = StableDiffusion2Client(model_id)
        elif model_id == "stabilityai/stable-diffusion-xl-base-1.0":
            from .text_to_image.stablediffusionxl import StableDiffusionxlClient
            self.client = StableDiffusionxlClient(model_id)
        else:
            raise Exception(f"Model {model_id} is not supported.")

    def generate(self, text, seed, save_path):
        return self.client.generate(text, seed, save_path)

