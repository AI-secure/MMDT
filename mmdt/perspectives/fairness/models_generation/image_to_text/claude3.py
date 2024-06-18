import anthropic
import base64
import os


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class ClaudeClient:
    def __init__(self, model_id):
        self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

        assert model_id == "claude-3-opus-20240229"
        self.client = anthropic.Anthropic()

    def generate(self, text, image_path, **kwargs):
        temperature = kwargs['temperature'] if 'temperature' in kwargs else 0
        base64_image = encode_image(image_path)
        message = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": text
                        }
                    ],
                }
            ],
            temperature=temperature,
        )
        return message.content[0].text
