import os
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from mmdt.models import Image2TextClient
from .utils import COCOLoader, modify_image_with_boxes
from mmdt.perspectives.hallucination.cooccurrence import generate_cooc_image_to_text
import tempfile

def generate(model_id, scenario, task):

    model_name = model_id.split("/")[1] if '/' in model_id else model_id
    # Adjusted path to read data from the correct directory
    file_path = os.path.join('../../data/image_to_text/hallucination', scenario, f'{task}.csv')
    file = pd.read_csv(file_path)
    img_ids = file['img_id'].tolist()
    prompts = file['question'].tolist()
    generation_configs = {'do_sample': False, 'max_new_tokens': 128}

    responses = []
    # Adjusted path for output directory
    output_dir = os.path.join('../../results/image_to_text/hallucination', scenario, task, model_name)
    os.makedirs(output_dir, exist_ok=True)
    response_file = os.path.join(output_dir, f'generation.csv')

    if os.path.exists(response_file):
        print(f"Skipping generation as output file already exists: {response_file}")
        return

    client = Image2TextClient(model_id=model_id)

    if scenario == "cooccurrence":
        generate_cooc_image_to_text(model_id, task, client, generation_configs, output_dir)
        return


    for i in tqdm(range(len(img_ids))):
        img_id = int(img_ids[i])
        text = prompts[i]
        image = coco.load_image(img_id)  # Ensure coco or equivalent loader is defined

        additional_instruction = get_instruction_by_task(task)
        text += additional_instruction

        if scenario == 'distraction':
            image = modify_image_with_boxes(image, eval(bboxes[i]))  # Ensure bboxes are loaded

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
    parser.add_argument('--scenario', type=str, default="natural", help='Scenario type')
    parser.add_argument('--task', type=str, default="identification", help='Task to be executed')
    args = parser.parse_args()

    generate(args.model_id, args.scenario, args.task)
