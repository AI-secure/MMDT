import requests
import os
import openai
from openai import OpenAI
import time
from base64 import b64decode


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class DalleClient:
    def __init__(self, model_id):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI()

        assert model_id in ["dall-e-2", "dall-e-3"]
        self.model_id = model_id

    def generate(self, text, seed, save_path):
        success = False
        while not success:
            try:
                response = self.client.images.generate(
                    model=self.model_id,
                    prompt=text,
                    response_format='b64_json',
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )

            except openai.RateLimitError as e:
                print('retrying...')
                time.sleep(1)
                continue
            success = True
        image_data = b64decode(response.data[0].b64_json)
        with open(save_path, mode="wb") as f:
            f.write(image_data)
        return image_data
