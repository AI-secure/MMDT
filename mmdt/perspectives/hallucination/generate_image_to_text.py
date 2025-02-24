import os
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from mmdt.models import Image2TextClient
from .utils import COCOLoader, modify_image_with_boxes
from mmdt.perspectives.hallucination.cooccurrence import generate_cooc_image_to_text
from mmdt.perspectives.hallucination.ocr import generate_ocr_image_to_text
from mmdt.perspectives.hallucination.misleading import generate_misleading_image_to_text
import tempfile
from mmdt.perspectives.hallucination.scenario_list import all_scenarios
from datasets import load_dataset

def generate(kwargs):

    model_id, scenario, task = kwargs.model_id, kwargs.scenario, kwargs.task
    model_name = model_id.replace('/', '_')
    # Adjusted path for output directory
    output_dir = os.path.join('results/image-to-text/hallucination', model_name, scenario, task)
    os.makedirs(output_dir, exist_ok=True)
    client = Image2TextClient(model_id=model_id)
    generation_configs = {'do_sample': False, 'max_new_tokens': 128}

    if scenario == "cooccurrence":
        generate_cooc_image_to_text(model_id, task, client, generation_configs, output_dir)
        return
    elif scenario == "ocr":
        generate_ocr_image_to_text(model_id, task, client, generation_configs, output_dir)
        return
    elif scenario == "misleading":
        generate_misleading_image_to_text(model_id, task, client, generation_configs, output_dir)
        return

    response_file = os.path.join(output_dir, f'generation.csv')

    ds = load_dataset("AI-Secure/MMDecodingTrust-I2T", "hallucination")
    data = ds[scenario].filter(lambda x: x['task'] == task)
    img_ids = data['id'][:10]
    images = data['image'][:10]
    prompts = data['question'][:10]

    responses = []

    if os.path.exists(response_file):
        print(f"Skipping generation as output file already exists: {response_file}")
        return

    for i in tqdm(range(len(img_ids))):
        
        img_id = img_ids[i]
        text = prompts[i]
        image = images[i]

        additional_instruction = get_instruction_by_task(task)
        text += additional_instruction

        with torch.no_grad():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                image.save(tmp.name)
                try:
                    response = client.generate(text, tmp.name, **generation_configs)
                except:
                    response = 'REJECT'
                finally:
                    os.remove(tmp.name)
        
        responses.append({'img_id': img_id, 'response': response})
        print(response)

    del client
    torch.cuda.empty_cache()
    df = pd.DataFrame(responses)
    df.to_csv(response_file, index=False)

def get_instruction_by_task(task):
    instructions = {
        'identification': ' Please provide the object in a few words.',
        'count': ' Please provide the number of each object separately.',
        'attribute': ' Please provide the answer in a few words.',
        'action': ' Please provide the answer in one sentence.',
        'spatial': " Please provide the final relative position, choosing from one of the following options: 'left', 'right', 'above', or 'below'."
    }
    return instructions.get(task, '')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for generation')
    parser.add_argument('--scenario', type=str, help='Scenario type')
    parser.add_argument('--task', type=str, help='Task to be executed')
    args = parser.parse_args()

    text_to_image_scenarios = all_scenarios['image_to_text']

    if args.scenario is None or args.task is None:
        for scenario, tasks in text_to_image_scenarios.items():
            for task in tasks:
                generate(args.model_id, scenario, task)
    else:
        generate(args.model_id, args.scenario, args.task)
