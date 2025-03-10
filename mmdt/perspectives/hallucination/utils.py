import torch.nn as nn
import transformers
import torch
from pycocotools.coco import COCO
import os
from PIL import Image, ImageDraw
import random
import requests

def determine_relative_position(bbox1, bbox2):
    """ Determine the relative position between two bounding boxes. """
    x1 = bbox1[0] + bbox1[2] / 2
    y1 = bbox1[1] + bbox1[3] / 2
    x2 = bbox2[0] + bbox2[2] / 2
    y2 = bbox2[1] + bbox2[3] / 2
    
    if x1 < x2 and abs(x1 - x2) > abs(y1 - y2):
        return 'left'
    elif x1 > x2 and abs(x1 - x2) > abs(y1 - y2):
        return 'right'
    elif y1 < y2:
        return 'above'
    else:
        return 'below'
    
class COCOLoader:
    def __init__(self, dataDir='data', dataType='train2017'):
        """
        Initialize CocoLoader to manage COCO dataset operations.
        
        Args:
        dataDir (str): Directory where the COCO data is stored.
        dataType (str): Specific subset of COCO dataset, e.g., 'train2017'.
        """
        self.dataDir = dataDir
        self.dataType = dataType
        self.instances_path = os.path.join(dataDir, 'annotations', f'instances_{dataType}.json')
        self.captions_path = os.path.join(dataDir, 'annotations', f'captions_{dataType}.json')
        self.instances = COCO(self.instances_path)
        self.captions = COCO(self.captions_path)
        self.imgIds = sorted(self.instances.getImgIds())
    
    def getImgIds(self):
        """
        Get all the imgIds.
        """
        return self.imgIds
    
    def load_image(self, image_id):
        """
        Load an image by its ID.
        """
        img_info = self.instances.loadImgs(image_id)[0]
        path = os.path.join(self.dataDir, self.dataType, img_info['file_name'])
        return Image.open(path)

    def load_annotation(self, image_id, iscrowd=None):
        """
        Load instance annotations for a specific image ID.
        """
        annIds = self.instances.getAnnIds(imgIds=image_id, iscrowd=iscrowd)
        return self.instances.loadAnns(annIds)
        
    def load_caption(self, image_id):
        """
        Retrieve captions for a given image ID.
        """
        annIds = self.captions.getAnnIds(imgIds=image_id)
        return self.captions.loadAnns(annIds)
    
    
def modify_image_with_boxes(image, bboxes):
    
    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)

    # Draw each bounding box with a dynamically scaled outline width
    for bbox in bboxes:
        # Calculate the outline width as a fraction of the average size of the bounding box
        outline_width = min(max(2, int((bbox[2] * bbox[3]) * 0.0025)), 5)  # 2.5% of the average dimension
        
        # Draw a rectangle on the image using the bbox coordinates
        draw.rectangle(
            [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
            outline="red",
            width=outline_width
        )

    # Return the modified image
    return image

class LLMChat(nn.Module):
    def __init__(self, model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'):
        super(LLMChat, self).__init__()
        if model_name == 'gpt-4o':
            self.api_key = os.getenv('OPENAI_API_KEY')
            self.forward = self.forward_gpt4o
        else:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": torch.float16},
                device="cuda",
            )

    def forward(self, prompt, max_new_tokens = 500):
        messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
        
        prompt = self.pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=False,
        )
        
        return outputs[0]["generated_text"][len(prompt):]

    def forward_gpt4o(self, prompt):
        temperature = 0

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
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
