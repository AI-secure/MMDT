# Image to text
from .image_to_text.gemini import GeminiClient
from .image_to_text.gpt4v import GPT4VClient
from .image_to_text.claude3 import ClaudeClient
from .image_to_text.instructblip import InstructBLIPClient
from .image_to_text.llava import LlavaClient
from .image_to_text.qwen import QwenClient

# Text to image
from .text_to_image.dalle import DalleClient
from .text_to_image.deepfloyd_if import IFClient
from .text_to_image.dreamlike import DreamlikeClient
from .text_to_image.kandinsky import KandinskyClient
from .text_to_image.opendalle import OpenDalleClient
from .text_to_image.openjourney import OpenjourneyClient
from .text_to_image.stablediffusion2 import StableDiffusion2Client
from .text_to_image.stablediffusionxl import StableDiffusionxlClient


class Image2TextClient:
    def __init__(self, model_id):
        if model_id == "models/gemini-1.5-pro-001":
            self.client = GeminiClient(model_id)
        elif model_id in ["gpt-4-vision-preview", "gpt-4o-2024-05-13"]:
            self.client = GPT4VClient(model_id)
        elif model_id == "claude-3-opus-20240229":
            self.client = ClaudeClient(model_id)
        elif model_id in ["llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf"]:
            self.client = LlavaClient(model_id)
        elif model_id == "Salesforce/instructblip-vicuna-7b":
            self.client = InstructBLIPClient(model_id)
        elif model_id == "Qwen/Qwen-VL-Chat":
            self.client = QwenClient(model_id)
        else:
            raise Exception(f"Model {model_id} is not supported.")

    def generate(self, text, image_path, **kwargs):
        return self.client.generate(text, image_path, **kwargs)


class Text2ImageClient:
    def __init__(self, model_id):
        if model_id in ["dall-e-2", "dall-e-3"]:
            self.client = DalleClient(model_id)
        elif model_id == "DeepFloyd/IF-I-M-v1.0":
            self.client = IFClient(model_id)
        elif model_id == "dreamlike-art/dreamlike-photoreal-2.0":
            self.client = DreamlikeClient(model_id)
        elif model_id == "kandinsky-community/kandinsky-3":
            self.client = KandinskyClient(model_id)
        elif model_id == "dataautogpt3/OpenDalleV1.1":
            self.client = OpenDalleClient(model_id)
        elif model_id == "prompthero/openjourney-v4":
            self.client = OpenjourneyClient(model_id)
        elif model_id == "stabilityai/stable-diffusion-2":
            self.client = StableDiffusion2Client(model_id)
        elif model_id == "stabilityai/stable-diffusion-xl-base-1.0":
            self.client = StableDiffusionxlClient(model_id)
        else:
            raise Exception(f"Model {model_id} is not supported.")

    def generate(self, text, seed, save_path):
        return self.client.generate(text, seed, save_path)

