import os
import pandas as pd
from tqdm import tqdm
import argparse
import torch
import sys
sys.path.append("../../../")
from mmdt.models import Text2ImageClient

def generate(model_id, task):

    model_name = model_id.split("/")[1] if '/' in model_id else model_id
    # Adjusted path to read data from the correct directory
    file_path = os.path.join('../../../data/text-to-image/adv', f'{task}.csv')
    data = pd.read_csv(file_path)

    # Adjusted path for output directory
    output_dir = os.path.join('../../../results/text-to-image/adv', task, model_name)
    os.makedirs(output_dir, exist_ok=True)

    seed = 20

    client = Text2ImageClient(model_id=model_id)

    for i, row in tqdm(data.iterrows()):
        img_id = row['id']
        prompt = row['adv_prompt']
        image_path = os.path.join(output_dir, f'{img_id}.png')
        if not os.path.exists(image_path):
            image = client.generate(prompt, seed=seed, save_path=image_path)
        else:
            print(f"Image {img_id} already exists. Skipping generation.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for image generation')
    parser.add_argument('--task', type=str, default="object", help='Specify the task to execute')
    args = parser.parse_args()

    generate(args.model_id, args.task)