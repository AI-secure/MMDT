import numpy as np
import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("./")

import time
import random
from glob import glob
from PIL import Image
import copy

from mmdt.detection.image_detector_utils import ImageDetector, draw_bbox
from mmdt.models import Text2ImageClient

from tqdm import tqdm
import json
import pandas as pd

import argparse

from datasets import load_dataset

seed = 20


def generate_image(client, prompt, seed_val, save_path, dry_run=False):
    if not os.path.exists(save_path):
        if dry_run:
            print(f"Dry run enabled. Generating an empty image at {save_path}.")
            # Create an empty white image of size 512x512
            image = Image.new("RGB", (512, 512), "white")
        else:
            image = client.generate(text=prompt, seed=seed_val, save_path=save_path)
        image.save(save_path)
    else:
        print(f"Image already exists. Skipping generation.")
        image = Image.open(save_path)
    return image


def generate_single_task(args):
    ds = load_dataset("AI-Secure/MMDecodingTrust-T2I", "adv", split=args.task)
    iter_ds = ds.to_iterable_dataset()

    result_root_dir = os.path.join("./results/text_to_image/adv", args.model_id, args.task)
    os.makedirs(result_root_dir, exist_ok=True)

    image_dir = os.path.join(result_root_dir, "output_images")
    os.makedirs(image_dir, exist_ok=True)

    client = Text2ImageClient(model_id=args.model_id)

    for row in tqdm(iter_ds, total=len(ds)):
        prompt = row['adv_prompt']
        index = row['id']
        for i in range(args.image_number):
            image_path = os.path.join(image_dir, f"{index}_{i}.png")
            generate_image(client, prompt, seed + i, image_path, dry_run=args.dry_run)
            print(f"Image {image_path} generated")

    print(f"Images generated at {image_dir}")


def generate(args):
    # Allow multiple tasks if needed (comma-separated)
    tasks = args.task if args.task != '' else 'object,attribute,spatial'
    tasks = [t.strip() for t in tasks.split(',')]
    args.image_number = 3  # set default value
    for task in tasks:
        temp_args = copy.deepcopy(args)
        temp_args.task = task
        generate_single_task(temp_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, help='Model ID to use for generation',
                        default="stabilityai/stable-diffusion-2")
    parser.add_argument('--task', type=str, help='Sub task to be executed', default="object")
    parser.add_argument('--image_number', type=int,
                        help='Number of images to be generated per sample', default=1)
    parser.add_argument('--dry_run', action='store_true',
                        help='Enable dry run mode to generate empty image')

    args = parser.parse_args()
    generate(args)
