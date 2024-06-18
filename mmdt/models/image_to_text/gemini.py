import google.generativeai as genai
from PIL import Image
import os


class GeminiClient:
    def __init__(self, model_id):
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # https://aistudio.google.com/app/apikey
        genai.configure(api_key=self.GOOGLE_API_KEY)

        assert model_id == "models/gemini-1.5-pro-001"
        self.model = genai.GenerativeModel(model_id)

    def generate(self, text, image_path, **kwargs):  # The output can be slightly different even t = 0. See [this](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts?_gl=1*wt5nw9*_ga*MTQ5NDkxMTU0Ny4xNzEyODQ1NTE5*_ga_WH2QY8WWF5*MTcxMzMwNTkwNC4zLjEuMTcxMzMwNzI5Ni4wLjAuMA..&_ga=2.208482322.-1494911547.1712845519#temperature:~:text=A%20temperature%20of%200%20means%20that%20the%20highest%20probability%20tokens%20are%20always%20selected.%20In%20this%20case%2C%20responses%20for%20a%20given%20prompt%20are%20mostly%20deterministic%2C%20but%20a%20small%20amount%20of%20variation%20is%20still%20possible.)
        image = Image.open(image_path)
        temperature = kwargs['temperature'] if 'temperature' in kwargs else 0
        response = self.model.generate_content(
            [text, image],
            stream=True,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        response.resolve()
        return response.text
