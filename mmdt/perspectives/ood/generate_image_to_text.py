import os
import time
import pandas as pd
from tqdm import tqdm
import argparse
import json
import sys
sys.path.append("../../../")
from mmdt.models import Image2TextClient
sys.path.remove("../../../")
import torch


max_retries = 100
retry_delay = 1

st_img_path = "../../../data/image-to-text/ood/images"
def generate(model_id, corruption, task):
    file_path = os.path.join("../../../data/image-to-text/ood", f"{task}.json")
    with open(file_path, "r") as f:
        data = json.load(f)

    print(f"Running model {model_id} on task {task} with corruption {corruption}")
    generation_configs = {'do_sample': False, 'temperature': 0, 'max_new_tokens': 128}
    responses = []
    output_dir = f'../../../results/image-to-text/ood/{corruption}/{task}/{model_id.split("/")[-1]}/'
    os.makedirs(output_dir, exist_ok=True)
    response_file = os.path.join(output_dir, 'generation.csv')
    if os.path.exists(response_file):
        df = pd.read_csv(response_file)
        num_current_responses = len(df)
        responses = df.to_dict('records')
    else:
        num_current_responses = 0
    
    client = Image2TextClient(model_id=model_id)
    all_index = list(data[corruption].keys())

    for i in tqdm(range(num_current_responses, len(data[corruption]))):
        img_id = data[corruption][all_index[i]]["img_id"]
        text = data[corruption][all_index[i]]["question"]
        image_path = os.path.join(st_img_path, f"{corruption}/{img_id}.jpg")

        if task == 'identification':
            additional_instruction = ' Please provide the object in a few words.'
        elif task == 'count':
            additional_instruction = " Please provide the number of each object separately."
        elif task == 'attribute':
            additional_instruction = ' Please provide the answer in a few words.'
        elif task == 'action':
            additional_instruction = ' Please provide the answer in one sentence.'
        elif task == 'spatial':
            additional_instruction = " Please provide the final relative position, choosing from one of the following options: 'left', 'right', 'above', or 'below'."

        with torch.no_grad():
            for attempt in range(max_retries):
                try:
                    response = client.generate(text + additional_instruction, image_path, **generation_configs)
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise  # Re-raise the last exception after the last retry
                    time.sleep(retry_delay)  # Wait before the next retry
                

        responses.append({'img_id': img_id, 'response': response})
        print(response)
        if i % 100 == 0:
            df = pd.DataFrame(responses)
            df.to_csv(response_file, index=False)
    del client
    torch.cuda.empty_cache()
    
    df = pd.DataFrame(responses)
    df.to_csv(response_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for image generation')
    parser.add_argument('--scenario', type=str, default="Van_Gogh", help='Specify the scenario')
    parser.add_argument('--task', type=str, default="identification", help='Specify the task to execute')
    args = parser.parse_args()
    generate(args.model_id, args.scenario, args.task)

