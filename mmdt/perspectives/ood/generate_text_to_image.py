import argparse
import numpy as np
import os
import sys
import json
import time
sys.path.append("../../../")
from mmdt.models import Image2TextClient, Text2ImageClient
sys.path.remove("../../../")
from datasets import load_dataset

max_retries = 50
retry_delay = 1
# Args parse
seed = 42
num_exp = 3


def generate(args):
    for exp_id in range(num_exp):
        args.exp_id = exp_id
        generate_single(args)

def generate_single(args):
    model_id = args.model_id
    task = args.task
    style = args.scenario
    exp_id = args.exp_id
    output_base = args.output_dir

    # Load dataset from Hugging Face
    ds = load_dataset("AI-Secure/MMDecodingTrust-T2I", "ood", split=style)
    iter_ds = ds.to_iterable_dataset()
    # Filter for current task
    data = [row for row in iter_ds if row['task'] == task]
    
    model_name = model_id.split("/")[-1]
    save_path = os.path.join(output_base, f"text-to-image/ood/{style}/{task}/{model_name}")

    os.makedirs(save_path + f"/images_{exp_id}", exist_ok=True)
    if os.path.exists(save_path + f'/results_dict_{exp_id}.json'):
        with open(save_path + f'/results_dict_{exp_id}.json', 'r') as file:
            results_dict = json.load(file)
    else:
        results_dict = {}
    result_save_path = save_path + f'/results_dict_{exp_id}.json'
    
    # Find data that hasn't been processed yet
    processed_ids = set(results_dict.keys())
    left_data = [row for row in data if str(row['id']) not in processed_ids]

    if len(left_data) != 0:
        client = Text2ImageClient(model_id=model_id)
    else:
        print("All data has been processed. Exiting...")
        client = None
        
    refusal = False
    for left_idx, row in enumerate(left_data):
        index = str(row['id'])
        print(f"Processing {left_idx}/{len(left_data)} on model {model_name}")
        results_dict[index] = {
            'meta_prompt': row['meta_prompt'] if "meta_prompt" in row else None,
            'objects': row['objects'] if "objects" in row else None,
            'task': row['task'],
            'vanilla_prompt': row['vanilla_prompt'] if "vanilla_prompt" in row else None
        }
        
        image_path = save_path + f"/images_{exp_id}/{index}.png"
        drawing_prompt = row['meta_prompt']
        for attempt in range(max_retries):
            try:
                image = client.generate(drawing_prompt, seed=seed + exp_id, save_path=image_path)
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if "content_policy_violation" in str(e) and attempt >= 10:
                    results_dict[index]["error"] = str(e)
                    print(f"Content policy violation: {e}")
                    refusal = True
                    break
                if attempt == max_retries - 1:
                    raise  # Re-raise the last exception after the last retry
                time.sleep(retry_delay)  # Wait before the next retry
                
        if model_id != "dall-e-2" and model_id != "dall-e-3":
            image.save(image_path)
            
        if refusal:
            refusal = False
            continue

        if left_idx % 10 == 0 or left_idx == len(left_data) - 1:
            with open(result_save_path, 'w') as file:
                json.dump(results_dict, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="helpfulness")
    parser.add_argument("--scenario", type=str, default="Shake")
    parser.add_argument("--model_id", type=str, default="dall-e-3")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    generate(args)
