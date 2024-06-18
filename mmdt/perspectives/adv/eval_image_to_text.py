import pandas as pd
import os
import re
from tqdm import tqdm
import argparse
import json

def evaluate(model_id, task):

    model_name = model_id.split("/")[1] if '/' in model_id else model_id
    # Adjusted path to read data from the correct directory
    file_path = os.path.join('../../../data/image-to-text/adv', f'{task}.csv')
    data = pd.read_csv(file_path)

    output_dir = os.path.join('../../../results/image-to-text/adv', task, model_name)
    os.makedirs(output_dir, exist_ok=True)

    responses_path = os.path.join(output_dir, 'generation.csv')
    responses = pd.read_csv(responses_path)['response'].tolist()
    image_folder = os.path.join('../../../data/image-to-text/adv', f'{task}')

    results = []
    for i, row in tqdm(data.iterrows()):
        img_id = row['id']
        response = responses[i]
        if task == 'object':
            origin_obj = row['origin_object']
            if origin_obj in response or origin_obj.capitalize() in response:
                accuracy = 1
            else:
                accuracy = 0
        elif task == 'attribute':
            if "Yes" in response or "Yes".capitalize() in response:
                accuracy = 1
            else:
                accuracy = 0
        elif task == 'spatial':
            origin_relation = row['origin_relation']
            if origin_relation in response or origin_relation.capitalize() in response:
                accuracy = 1
            else:
                accuracy = 0
        results.append({
            'img_id': img_id,
            'accuracy': accuracy
        })

    # Save the results to a CSV file
    df_results = pd.DataFrame(results)
    result_file = os.path.join(output_dir, 'evaluation.csv')
    df_results.to_csv(result_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID of the model to be evaluated')
    parser.add_argument('--task', type=str, required=True, help='Task to be evaluated')
    args = parser.parse_args()

    evaluate(args.model_id, args.task)