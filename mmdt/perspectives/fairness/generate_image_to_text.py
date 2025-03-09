import os
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from mmdt.models import Image2TextClient
from mmdt.perspectives.fairness.prompt_manager import PromptManager_i2t
from datasets import load_dataset
import json

def generate_stereotype(model_id, args):
    pass

def generate_decision(model_id, args):
    pass

def generate_overkill(model_id, args):
    pass

def generate_individual(model_id, args):
    client = Image2TextClient(args.model_id)
    scenario = args.scenario
    prompt_dict_list = load_dataset("AI-Secure/MMDecodingTrust-I2T", "fairness")[args.scenario]
    prompt_dict_list = [dict(row) for row in prompt_dict_list]


    temp_dir = './temp/fairness/'
    os.makedirs(temp_dir, exist_ok=True)


    file_path = './mmdt/perspectives/fairness/data/image-to-text/i2t_individual.jsonl'

    data = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            data.append(entry)

    results = []
    generate_config = {'do_sample': True, 'max_new_tokens': 10}
    for idx,record in tqdm(enumerate(data)):
        temp_path = os.path.join(temp_dir, f'{idx}.png')
        prompt_dict_list[idx]['image'].save(temp_path)
        record['img_path'] = temp_path

        for ans_id in range(10):
            answer = client.generate(record['q'], record['img_path'], **generate_config)
            if ans_id == 0:
                record['a'] = []
            record['a'].append(answer)
            results.append(record)
    os.makedirs('./temp_results/i2t', exist_ok=True)
    with open(os.path.join('./temp_results/i2t', f'individual_model_id.jsonl'), 'w') as file:
        for entry in results:
            json_line = json.dumps(entry)
            file.write(json_line + '\n')


def generate(args):

    model_id = args.model_id

    if args.scenario == "social_stereotype":
       generate_stereotype(model_id, args)
    elif args.scenario == "decision_making":
        generate_decision(model_id, args)
    elif args.scenario == "overkill":
        generate_overkill(model_id, args)
    elif args.scenario == "individual":
        generate_individual(model_id, args)