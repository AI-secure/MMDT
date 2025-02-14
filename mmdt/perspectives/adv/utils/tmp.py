from pycocotools.coco import COCO
import os
from PIL import Image
import torch.nn as nn
import transformers
import torch

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

class LLMChat(nn.Module):
    def __init__(self, model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'):
        super(LLMChat, self).__init__()
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

