from mmdt.models import Text2ImageClient
import os
import pandas as pd
from tqdm import tqdm
import argparse
from prompt_manager import PromptManager


def generate(model_id, scenario, task):
    # data loading
    prompt_manager = PromptManager(args.scenario)
    prompts = prompt_manager.get_prompt()

    # Define the output directory based on model name, scenario, and task
    model_name = model_id.split("/")[1] if '/' in model_id else model_id
    output_dir = os.path.join('../../results/text_to_image/fairness', scenario, task, model_name)
    os.makedirs(output_dir, exist_ok=True)

    seed = 0

    client = Text2ImageClient(model_id=model_id)

    # Iterate over each prompt to generate images one by one
    for prompt in tqdm(prompts, desc=f"Generating images for {model_name}", total=len(prompts)):
        dir_path = os.path.join(output_dir, prompt)
        for img_id in range(args.num_images_per_prompt):
            image_path = os.path.join(dir_path, f'{img_id}.png')
            if not os.path.exists(image_path):
                image = client.generate(prompt, seed=seed)
                image.save(image_path)
            else:
                print(f"Image {img_id} already exists. Skipping generation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for image generation')
    parser.add_argument('--scenario', type=str, choices=['test', 'occupation', 'occupation_with_sex', 'occupation_with_race', 'occupation_with_age',
                                                                        'education', 'education_with_sex', 'education_with_race', 'activity', 'activity_with_sex'], help='Specify the scenario')
    parser.add_argument('--task', type=str, default="attribute", help='Specify the task to execute')
    parser.add_argument('--num_images_per_prompt', type=int, default=10)
    args = parser.parse_args()

    generate(args.model_id, args.scenario, args.task)