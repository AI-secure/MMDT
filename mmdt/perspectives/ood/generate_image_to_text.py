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
from datasets import load_dataset


max_retries = 100
retry_delay = 1

def generate(args):
    model_id = args.model_id
    corruption = args.scenario
    task = args.task
    output_base = args.output_dir

    # Load dataset from Hugging Face
    ds = load_dataset("AI-Secure/MMDecodingTrust-I2T", "ood", split=corruption)
    iter_ds = ds.to_iterable_dataset()
    # Filter for current task
    task_data = [row for row in iter_ds if row['task'] == task]

    print(f"Running model {model_id} on task {task} with corruption {corruption}")
    generation_configs = {'do_sample': False, 'temperature': 0, 'max_new_tokens': 128}
    responses = []
    output_dir = os.path.join(output_base, f'image-to-text/ood/{corruption}/{task}/{model_id.split("/")[-1]}/')
    os.makedirs(output_dir, exist_ok=True)
    response_file = os.path.join(output_dir, 'generation.csv')
    if os.path.exists(response_file):
        df = pd.read_csv(response_file)
        num_current_responses = len(df)
        responses = df.to_dict('records')
    else:
        num_current_responses = 0
    
    client = Image2TextClient(model_id=model_id)

    for i in tqdm(range(num_current_responses, len(task_data))):
        row = task_data[i]
        img_id = row["img_id"]
        text = row["question"]
        image = row["image"]
        # Save image temporarily for processing
        temp_image_path = os.path.join(output_dir, f"temp_{img_id}.jpg")
        image.save(temp_image_path)

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
                    response = client.generate(text + additional_instruction, temp_image_path, **generation_configs)
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise  # Re-raise the last exception after the last retry
                    time.sleep(retry_delay)  # Wait before the next retry

        # Clean up temporary image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

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
    parser.add_argument('--output_dir', type=str, default="./results", help='Output directory')
    args = parser.parse_args()
    generate(args)

