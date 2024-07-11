import base64
import requests
import os


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class GPT4VClient:
    def __init__(self, model_id):
        self.api_key = os.getenv('OPENAI_API_KEY')

        assert model_id in ["gpt-4-vision-preview", "gpt-4o-2024-05-13"]
        self.model_id = model_id

    def generate(self, text, image_path, **kwargs):
        temperature = kwargs['temperature'] if 'temperature' in kwargs else 0
        base64_image = encode_image(image_path)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": temperature
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        responses = [resp['message']['content'] for resp in response.json()['choices']]
        return responses[0]


    def generate_multiple_img(self, text, image_path_list, **kwargs):
        temperature = kwargs['temperature'] if 'temperature' in kwargs else 0
        
        base64_images = [encode_image(path) for path in image_path_list if os.path.exists(path)]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text}
                    ] + [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} 
                        for base64_image in base64_images
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": temperature
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        responses = [resp['message']['content'] for resp in response.json()['choices']]
        return responses[0]