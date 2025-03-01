import os
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from mmdt.models import Image2TextClient
from mmdt.perspectives.fairness.prompt_manager import PromptManager_i2t
from datasets import load_dataset

def generate(args):

    model_id = args.model_id
    scenario = args.scenario
    task = "attribute"

    prompt_dict_list = load_dataset("AI-Secure/MMDecodingTrust-I2T", "fairness")[f'group_{scenario}']


    model_name = model_id.split("/")[1] if '/' in model_id else model_id
    # Adjusted path to read data from the correct directory
    # prompt_manager = PromptManager_i2t(scenario)
    # prompt_dict_list = prompt_manager.get_prompt()
    generation_configs = {'do_sample': False, 'max_new_tokens': 128}

    responses = []
    # Adjusted path for output directory
    output_dir = os.path.join('./results/image_to_text/fairness', scenario)
    os.makedirs(output_dir, exist_ok=True)
    response_file = os.path.join(output_dir, f'generation.csv')

    if os.path.exists(response_file):
        print(f"Skipping generation as output file already exists: {response_file}")
        return

    client = Image2TextClient(model_id=model_id)
    keys = list(prompt_dict_list[0].keys())

    count = 0
    temp_dir = './temp/fairness/'
    os.makedirs(temp_dir, exist_ok=True)
    for record in tqdm(prompt_dict_list):
        temp_path = os.path.join(temp_dir, f'{count}.png')
        record['image'].save(temp_path)
        record['img_path'] = temp_path
        for img_id in range(10):
            if 'q_gender' in keys:
                answer_gender = client.generate(record['q_gender'], record['img_path'], **generation_configs)
                if img_id==0:
                    record['a_gender'] = []
                record['a_gender'].append(answer_gender)

            if 'q_race' in keys:
                answer_race = client.generate(record['q_race'], record['img_path'], **generation_configs)
                if img_id==0:
                    record['a_race'] = []
                record['a_race'].append(answer_race)

            if 'q_age' in keys:
                answer_age = client.generate(record['q_age'], record['img_path'], **generation_configs)
                if img_id==0:
                    record['a_age'] = []
                record['a_age'].append(answer_age)
        responses.append(record)

    del client
    torch.cuda.empty_cache()
    df = pd.DataFrame(responses)
    df.to_csv(response_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for generation')
    parser.add_argument('--scenario', type=str, choices=['occupation', 'education', 'activity', 'person_identification'], help='Scenario type')
    args = parser.parse_args()

    # generate(args.model_id, args.scenario, args.task)
