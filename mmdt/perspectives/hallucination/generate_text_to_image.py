from mmdt.models import Text2ImageClient
import os
import pandas as pd
from tqdm import tqdm
import argparse
from mmdt.perspectives.hallucination.cooccurrence import generate_cooc_text_to_image
from mmdt.perspectives.hallucination.ocr import generate_ocr_text_to_image
from mmdt.perspectives.hallucination.misleading import generate_misleading_text_to_image
from mmdt.perspectives.hallucination.scenario_list import all_scenarios
from datasets import load_dataset

def generate(kwargs):
    seed = 0
    model_id, scenario, task = kwargs.model_id, kwargs.scenario, kwargs.task
    # Define the output directory based on model name, scenario, and task
    model_name = model_id.replace('/', '_')
    output_dir = os.path.join('results/text-to-image/hallucination', model_name, scenario, task)
    os.makedirs(output_dir, exist_ok=True)

    client = Text2ImageClient(model_id=model_id)

    if scenario == "cooccurrence":
        generate_cooc_text_to_image(model_id, task, client, seed, output_dir)
        return
    elif scenario == "ocr":
        generate_ocr_text_to_image(model_id, task, client, seed, output_dir)
        return
    elif scenario == "misleading":
        generate_misleading_text_to_image(model_id, task, client, seed, output_dir)
        return
    else:
        # Adjusted path to read data from the correct directory
        ds = load_dataset("AI-Secure/MMDecodingTrust-T2I", "hallucination")
        data = ds[scenario].filter(lambda x: x['task'] == task)
        img_ids = data['id'][:10]
        prompts = data['prompt'][:10]
        
    # Iterate over each prompt to generate images one by one
    for img_id, prompt in tqdm(zip(img_ids, prompts), desc=f"Generating images for {model_name}", total=len(prompts)):
        image_path = os.path.join(output_dir, f"{img_id}.jpg")
        if os.path.exists(image_path):
            print(f"Image {img_id} already exists. Skipping generation.")
        else:
            image = client.generate(prompt, seed, image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for generation')
    parser.add_argument('--scenario', type=str, help='Scenario type')
    parser.add_argument('--task', type=str, help='Task to be executed')
    args = parser.parse_args()

    text_to_image_scenarios = all_scenarios['text_to_image']

    if args.scenario is None or args.task is None:
        for scenario, tasks in text_to_image_scenarios.items():
            for task in tasks:
                generate(args.model_id, scenario, task)
    else:
        generate(args.model_id, args.scenario, args.task)
