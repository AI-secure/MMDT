import numpy as np
import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "utils"))
sys.path.append("../../../")

import time
import random
from glob import glob
from PIL import Image

from utils.image_detector_utils import ImageDetector, draw_bbox
from mmdt.models import Text2ImageClient

from tqdm import tqdm
import json
import pandas as pd

import argparse

from datasets import load_dataset

seed = 20

def generate(client, prompt, seed, save_path):
    if not os.path.exists(save_path):
        image = client.generate(text=prompt, seed=seed, save_path=save_path)
        image.save(save_path)
    else:
        print(f"Image already exists. Skipping generation.")
        image = Image.open(save_path)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, help='Model ID to use for generation', default="stabilityai/stable-diffusion-2")
    parser.add_argument('--sub_task', type=str, help='Sub task to be executed', default="object")
    parser.add_argument('--image_number', type=int, help='Image number to be generated', default=1)

    args = parser.parse_args()

    ds = load_dataset("AI-Secure/MMDecodingTrust-T2I", "adv", split=args.sub_task)
    iter_ds = ds.to_iterable_dataset()

    result_root_dir = os.path.join("./results/text_to_image", args.model_id, args.sub_task)
    os.makedirs(result_root_dir, exist_ok=True)

    image_dir = os.path.join(result_root_dir, "output_images")
    os.makedirs(image_dir, exist_ok=True)

    client = Text2ImageClient(model_id=args.model_id)

    # print(ds[0]['adv_prompt'])

    for row in tqdm(iter_ds):
        prompt = row['adv_prompt']
        index = row['id']
        for i in range(args.image_number):
            image_path = os.path.join(image_dir, f"{index}_{i}.png")
            generate(client, prompt, seed+i, image_path)
            print(f"Image {image_path} generated")

    print(f"Images generated at {image_dir}")