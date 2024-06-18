import argparse
import numpy as np
import os
import sys
import json
import time
sys.path.append("../../../")
from mmdt.models import Image2TextClient, Text2ImageClient
sys.path.remove("../../../")

max_retries = 50
retry_delay = 1
# Args parse
seed = 42



def generation(model_id, style, task, exp_id=0):
    with open(f'../../../data/text-to-image/ood/{task}.json', 'r') as file:
        data_all = json.load(file)
    model_name = model_id.split("/")[-1]
    save_path = f"../../../results/text-to-image/ood/{style}/{task}/{model_name}"

    os.makedirs(save_path + f"/images_{exp_id}", exist_ok=True)
    if os.path.exists(save_path + f'/results_dict_{exp_id}.json'):
        with open(save_path + f'/results_dict_{exp_id}.json', 'r') as file:
            results_dict = json.load(file)
    else:
        results_dict = {}
    result_save_path = save_path + f'/results_dict_{exp_id}.json'
    data = data_all[style]
    left_data = [index for index in data.keys() if index not in results_dict.keys()]

    if len(left_data) != 0:
        client = Text2ImageClient(model_id=model_id)
    else:
        print("All data has been processed. Exiting...")
        client = None
    refusal = False
    for left_idx, index in enumerate(left_data):
        print(f"Processing {left_idx}/{len(left_data)} on model {model_name}")
        results_dict[index] = data[index]
        image_path = save_path + f"/images_{exp_id}/{index}.png"
        drawing_prompt = data[index]['meta_prompt']
        for attempt in range(max_retries):
            try:
                image = client.generate(drawing_prompt, seed=seed + exp_id, save_path = image_path)
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if "content_policy_violation" in str(e) and attempt >= 10:
                    results_dict[index] = data[index]
                    results_dict[index]["error"] = str(e)
                    print(f"Content policy violation: {e}")
                    refusal = True
                    break
                if attempt == max_retries - 1:
                    # raise  # Re-raise the last exception after the last retry
                    raise # Re-raise the last exception after the last retry
                time.sleep(retry_delay)  # Wait before the next retry
        if model_id != "dall-e-2" and model_id != "dall-e-3":
            image.save(image_path)
        results_dict[index] = data[index]
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
    parser.add_argument("--num_generation", type=int, default=1)
    args = parser.parse_args()

    for exp_id in range(args.num_generation):
        generation(args.model_id, args.scenario, args.task, exp_id)
