from mmdt.models import Text2ImageClient
import os
from tqdm import tqdm
import argparse
import json

def generate(model_id, scenario, task, seed):
    # Adjusted path to read data from the correct directory
    file_path = os.path.join('../../data/text_to_image/privacy', task, f'caption_names.csv')
    data_captions = []
    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            data_captions.append(json_object)

    # Define the output directory based on model name, scenario, and task
    model_name = model_id.split("/")[1] if '/' in model_id else model_id
    output_dir = os.path.join('../../results/text_to_image/privacy', scenario, task, model_name, f"seed{seed}")
    os.makedirs(output_dir, exist_ok=True)

    client = Text2ImageClient(model_id=model_id)


    # Iterate over each prompt
    for _, data in tqdm(enumerate(data_captions), desc=f"Generating images for {model_name}"): 
        prompt=data['prompt']
        prompt_for_fname = prompt.replace(' ','_').replace('/','_')[:50]
        fname= "{}_{}.png".format(data['idx'],prompt_for_fname)
        image_path=os.path.join(output_dir, fname)
        if not os.path.exists(image_path):
            image = client.generate(prompt, seed=seed)
            image.save(image_path)
        else:
            print(f"Image {fname} already exists. Skipping generation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for image generation')
    parser.add_argument('--scenario', type=str, default="pretrain", help='Specify the scenario')
    parser.add_argument('--task', type=str, default="LAION_1K", help='Specify the task to execute')
    parser.add_argument('--seed', type=int, default=0, help='Specify the random seed')
    args = parser.parse_args()

    generate(args.model_id, args.scenario, args.task, args.seed)
