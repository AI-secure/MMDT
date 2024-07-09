from mmdt.models import Text2ImageClient
import os
import pandas as pd
from tqdm import tqdm
import argparse
from mmdt.perspectives.hallucination.cooccurrence import generate_cooc_text_to_image
from scenario_list import all_scenarios

def generate(model_id, scenario, task):
    # Adjusted path to read data from the correct directory
    file_path = os.path.join('../../data/text_to_image/hallucination', scenario, f'{task}.csv')
    data = pd.read_csv(file_path)
    img_ids = data['img_id'].tolist()
    prompts = data['prompt'].tolist()
    

    # Define the output directory based on model name, scenario, and task
    model_name = model_id.split("/")[1] if '/' in model_id else model_id
    output_dir = os.path.join('../../results/text_to_image/hallucination', model_name, scenario, task)
    os.makedirs(output_dir, exist_ok=True)

    seed = 0

    client = Text2ImageClient(model_id=model_id)

    if scenario == "cooccurrence":
        generate_cooc_text_to_image(model_id, task, client, seed, output_dir)
        return


    # Iterate over each prompt to generate images one by one
    for img_id, prompt in tqdm(zip(img_ids, prompts), desc=f"Generating images for {model_name}", total=len(prompts)):
        image_path = os.path.join(output_dir, f"{img_id}.jpg")
        if not os.path.exists(image_path):
            image = client.generate(prompt, seed=seed)
            image.save(image_path)
        else:
            print(f"Image {img_id} already exists. Skipping generation.")

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
