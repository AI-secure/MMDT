from mmdt.models import Text2ImageClient
import os
import pandas as pd
from tqdm import tqdm
import argparse
import json
import jsonlines
from tqdm import tqdm
from datasets import load_dataset


def model_responses(client, model_name, prompt_list, batch_size, benchmark_filename):
    save_dir = f"model_responses/{model_name}/{benchmark_filename}"
    os.makedirs(save_dir, exist_ok=True)

    data_list = []
    for idx, prompt_json in enumerate(tqdm(prompt_list)):
        repeat = 5
        if (prompt_json['fairness'] == 'stats-based'):
            for i in range(len(prompt_json['prompt'])):
                for j in range(0, repeat, batch_size):
                    batch_prompts = [prompt_json['prompt'][i]] * min(batch_size, repeat - j)
                    index = [f"{idx}_{i}_{k}" for k in range(j, min(j + batch_size, repeat))]
                    for num, ind in enumerate(index):
                        if not os.path.exists(f"model_responses/{model_name}/{benchmark_filename}/{model_name}_{ind}.png"):
                            client.generate(batch_prompts[num], 0, f"model_responses/{model_name}/{benchmark_filename}/{model_name}_{ind}.png")
                    for q, ind in enumerate(index):
                        data_json = {"fairness": prompt_json['fairness'],
                                     "demographic attribute": prompt_json["demographic attribute"],
                                     "demographic attribute 1": prompt_json["demographic attribute 1"],
                                     "demographic attribute 2": prompt_json["demographic attribute 2"],
                                     "prompt_index": i, "repeat": j + q, "system_type": prompt_json['system_type'],
                                     "details": prompt_json['details'], "image_result": ind}
                        data_list.append(data_json)
        else:

            batch_prompts = [prompt_json['prompt']] * repeat
            index = [f"{idx}_{k}" for k in range(repeat)]

            for num, ind in enumerate(index):
                # print(f'num: {num}; ind: {ind}')
                if not os.path.exists(f"model_responses/{model_name}/{benchmark_filename}/{model_name}_{ind}.png"):
                    client.generate(batch_prompts[num], 0, f"model_responses/{model_name}/{benchmark_filename}/{model_name}_{ind}.png")
            for q, ind in enumerate(index):
                data_json = prompt_json.copy()
                data_json['repeat'] = q
                data_json['image_result'] = ind
                data_list.append(data_json)
    with jsonlines.open(f'model_responses/{model_name}_{benchmark_filename}.jsonl', mode='w') as writer:
        writer.write_all(data_list)

def load_data(args):
    # "social_stereotype", "decision_making", "overkill", "individual"

    # ds = load_dataset("AI-Secure/MMDecodingTrust-T2I", "fairness")
    # print(ds)
    # exit()


    prompt_list = []
    if args.scenario == "social_stereotype":
        with open('./mmdt/perspectives/fairness/data/text-to-image/stereotype.jsonl', "r", encoding="utf-8") as f:
            for line in f:
                prompt_list.append(json.loads(line.strip()))
    elif args.scenario == "decision_making":
        with open('./mmdt/perspectives/fairness/data/text-to-image/decision_making_education.jsonl', "r", encoding="utf-8") as f:
            for line in f:
                prompt_list.append(json.loads(line.strip()))
        with open('./mmdt/perspectives/fairness/data/text-to-image/decision_making_finance.jsonl', "r", encoding="utf-8") as f:
            for line in f:
                prompt_list.append(json.loads(line.strip()))
        with open('./mmdt/perspectives/fairness/data/text-to-image/decision_making_hiring.jsonl', "r", encoding="utf-8") as f:
            for line in f:
                prompt_list.append(json.loads(line.strip()))
    elif args.scenario == "overkill":
        with open('./mmdt/perspectives/fairness/data/text-to-image/overkill_fairness.jsonl', "r", encoding="utf-8") as f:
            for line in f:
                prompt_list.append(json.loads(line.strip()))
    elif args.scenario == "individual":
        with open('./mmdt/perspectives/fairness/data/text-to-image/individual_fairness.jsonl', "r", encoding="utf-8") as f:
            for line in f:
                prompt_list.append(json.loads(line.strip())["prompt"])
    else:
        raise ValueError(f"Unsupported scenario {args.scenario}")
    return prompt_list

def generate(args):

    model_id = args.model_id
    scenario = args.scenario
    num_images_per_prompt = 10
    seed = 0

    # data loading
    prompt_list = load_data(args)

    # Define the output directory based on model name, scenario, and task
    model_name = model_id.split("/")[1] if '/' in model_id else model_id
    output_dir = os.path.join('./results/text_to_image/fairness', scenario, model_name)
    os.makedirs(output_dir, exist_ok=True)

    client = Text2ImageClient(model_id=model_id)

    if args.scenario == "social_stereotype":
        # 'stereotype','factual_accuracy','hiring','education','finance'
        model_responses(client, model_name, prompt_list, batch_size=1, benchmark_filename='stereotype')
    elif args.scenario == "decision_making":
        for type in ['hiring','education','finance']:
            model_responses(client, model_name, prompt_list, batch_size=1, benchmark_filename=type)
    elif args.scenario == "overkill":
        model_responses(client, model_name, prompt_list, batch_size=1, benchmark_filename='overkill')
    elif args.scenario == "individual":
        # Iterate over each prompt to generate images one by one
        for prompt in tqdm(prompt_list, desc=f"Generating images for {model_name}", total=len(prompt_list)):
            dir_path = os.path.join(output_dir, prompt if len(prompt)<256 else prompt[:256])
            os.makedirs(dir_path, exist_ok=True)
            for img_id in range(num_images_per_prompt):
                image_path = os.path.join(dir_path, f'{img_id}.png')
                if not os.path.exists(image_path):
                    client.generate(prompt, seed=seed, save_path=image_path)
                else:
                    print(f"Image {img_id} already exists. Skipping generation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for image generation')
    parser.add_argument('--scenario', type=str, choices=["social_stereotype", "decision_making", "overkill", "individual"], help='Specify the scenario')
    args = parser.parse_args()
    generate(args)