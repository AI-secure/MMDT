import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from functools import partial
import cv2
import requests
import io
from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path
import sys, os
sys.path.append("./")
sys.path.append("./utils/GroundingDINO/")
import warnings
warnings.filterwarnings("ignore")
import torch
from torchvision.ops import box_convert

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T

from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionInpaintPipeline
import supervision as sv
from nltk.corpus import wordnet
import jsonlines
from tqdm import tqdm
import random

api_key = 'sk-xxxxxx'


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()

    return model


def load_model_local(config_file, cache_file, device='cpu'):

    args = SLConfig.fromfile(config_file) 
    model = build_model(args)
    args.device = device

    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()

    return model



def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(io.BytesIO(r.content)) as im:
        im.save(image_file_path)

    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))

def generate_masks_with_grounding(image_source, boxes):
    h, w, _ = image_source.shape
    boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    mask = np.zeros_like(image_source)
    for box in boxes_xyxy:
        x0, y0, x1, y1 = box
        mask[int(y0):int(y1), int(x0):int(x1), :] = 255
    return mask


# Use this command for evaluate the Grounding DINO model

# ckpt_repo_id = "ShilongLiu/GroundingDINO"
# ckpt_filenmae = "groundingdino_swint_ogc.pth"
# ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"

    

class ImageEditor:

    def __init__(self, inpainting_model_id = "stabilityai/stable-diffusion-2-inpainting", debugger=False, box_threshold=0.5, text_threshold=0.2):
        
        # self.model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
        config_file = "/home/czr/MMHallucination/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        cache_file = "/home/czr/MMHallucination/utils/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        self.model = load_model_local(config_file, cache_file)


        self.BOX_TRESHOLD = box_threshold
        self.TEXT_TRESHOLD = text_threshold

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            inpainting_model_id,
            torch_dtype=torch.float16,
        ).to("cuda")

        self.debugger = debugger
        
    def edit_image(self, local_image_path, target_entity, new_entity=None, box_threshold=None, text_threshold=None, save_dir=None):
        
        if box_threshold == None:
            BOX_TRESHOLD = self.BOX_TRESHOLD
        else:
            BOX_TRESHOLD = box_threshold
        if text_threshold == None:
            TEXT_TRESHOLD = self.TEXT_TRESHOLD
        else:
            TEXT_TRESHOLD = text_threshold

        image_source, image = load_image(local_image_path)

        TEXT_PROMPT = target_entity
        try:
            boxes, logits, phrases = predict(
                model=self.model, 
                image=image, 
                caption=TEXT_PROMPT, 
                box_threshold=BOX_TRESHOLD, 
                text_threshold=TEXT_TRESHOLD
            )
        except:
            boxes, logits, phrases = predict(
            model=self.model, 
            image=image, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD/2, 
            text_threshold=TEXT_TRESHOLD/2
            )
        assert len(boxes) > 0, "Target entity not detected, try to adjust the threshold!"

        print("logits", logits)
        # print("boxes", boxes)
        # input()

        # offset_x = 0.659
        # offset_y = 0.047
        # width = 0.254
        # height = 0.254

        # boxes = [[0.7663, 0.2232, 0.1722, 0.1865]]
        # boxes = torch.tensor(boxes)

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB

        image_mask = generate_masks_with_grounding(image_source, boxes)

        # image source
        image_source = Image.fromarray(image_source)
        
        # annotated image
        annotated_image = Image.fromarray(annotated_frame)
        
        # image mask
        image_mask = Image.fromarray(image_mask)
        
        if self.debugger:
            image_source.save(local_image_path)
            annotated_image.save(f"utils/image_editor_cache/{local_image_path}_annotated.jpg")
            image_mask.save(f"utils/image_editor_cache/{local_image_path}_masked.jpg")
            

        image_source_for_inpaint = image_source.resize((512, 512))
        image_mask_for_inpaint = image_mask.resize((512, 512))

        if new_entity == None:
            prompt = "None"
        else:
            prompt = new_entity

        #The mask structure is white for inpainting and black for keeping as is
        image_inpainting = self.pipe(prompt=prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]

        image_inpainting = image_inpainting.resize((image_source.size[0], image_source.size[1]))

        if save_dir != None:
            local_image_path = local_image_path.split("/")[-1].split(".")[0]
            # new_entity = new_entity.split(" ").join("_")
            # AttributeError: 'list' object has no attribute 'join'
            new_entity = new_entity.replace(" ", "_")
            # image_inpainting.save(f"{save_dir}/{local_image_path}_edited_{new_entity}.jpg")
            # image_inpainting.save(f"{save_dir}/{local_image_path}_{new_entity}.jpg")
            image_inpainting.save(f"{save_dir}/{local_image_path}_.jpg")
            print(f"\nImage cached to {save_dir}/{local_image_path}_.jpg ")

        return image_inpainting

            


if __name__ == "__main__":


    query_prompt = """
            Suppose that I have an image that contain two objects. 
            Now I want to remove one of the objects from the image, 
            and replace it with another. Your task is to choose one
            object to place the original one. There are mainly two criteria
            for the new object. 1. It has to be different from the original one,
            and cannot be a synonym of the original one. 
            2. The new object should be as misleading as possible, which means it should
            guide the detection model to think that this new object is the original one,
            however it is not. 
            3. The new object and the other object should be reasonble to co-occur in the same image.
            Now you should provide five candidate objects and generate nothing else.
            For example:
            Original objects: surfboard, person
            Object to replace: surfboard
            New object: skateboard, boat, ship, beach, motorcycle
            Original objects: surfboard, person
            Object to replace: person
            New object: dog, cat, tiger, box, ropes
            Original objects: car, bicycle
            Object to replace: bicycle
            New object: motorcycle, truck, bus, person, charger
            Original objects: {object1}, {object2}
            Object to replace: {object2}
            New object:
            """

    image_editor = ImageEditor(debugger=False)


    image_url = 'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/cats.png'
    local_image_path = 'examples/cats.png'
    # download_image(image_url, local_image_path)


    target_entity = "black cat"


    # determine the target entity to inpaint
    if True:
        new_entity = "dog" 

    if False:
        hypernyms_set = []
        hyponyms_set = []
        for ss in wordnet.synsets(concept2):
            for lemma in ss.lemmas():
                if lemma.antonyms():
                    for antonym in lemma.antonyms():
                        antonyms_set.append(antonym.name())

        for hp in wordnet.synsets(concept2):
            for hypernym in hp.hypernyms():
                hypernyms_set.append(hypernym.name().split(".")[0])
                # then get the hyponym of the hypernym
                for hyponym in hypernym.hyponyms():
                    for lemma in hyponym.lemmas():
                        hyponyms_set.append(lemma.name())
                # hypernyms_set.append(lemma.name())
                # antonyms_set.append(lemma.name())
        
        #random choose one from the synonyms
        print(f"hypernyms_set: ", hypernyms_set)
        print(f"hyponyms_set: ", hyponyms_set)

        new_entity = random.choice(hyponyms_set)

    if False:
        # using GPT-3.5 to get the new entity
        processed_prompt = query_prompt.format(object1=concept1, object2=concept2)
        print(f"Prompt: ", processed_prompt)
        response = requests.post(
        # 'https://azure-openai-api.shenmishajing.workers.dev/v1/chat/completions',
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {api_key}'},
        json={'model': "gpt-3.5-turbo", "messages": [{"role": "user", "content": processed_prompt}], 'max_tokens': 256, 'n': 1, 'temperature': 0.4}
        )
        data = response.json()
        
        generation = data['choices'][0]['message']['content']
        generation = generation.strip('"')

        entity_list = generation.split(",")

        print(f"entity_list: ", entity_list)
        # randomly choose one
        new_entity = random.choice(entity_list)


    image_inpainting = image_editor.edit_image(local_image_path, target_entity, new_entity, save_dir="examples")

    # image_inpainting.save(f"examples/{local_image_path}_inpainted_{new_entity}.jpg")
