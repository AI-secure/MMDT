import os
import argparse
import pandas as pd
from tqdm import tqdm
from mmdt.models import Image2TextClient
from mmdt.detection import ImageDetector
from .utils import determine_relative_position
from PIL import Image
from scenario_list import all_scenarios

def evaluate(model_id, scenario, task):
    """ Main function to perform image evaluations. """
    # Adjust the paths for the data and results directories
    file_path = os.path.join('../../data/text_to_image/hallucination', scenario, f'{task}.csv')
    file = pd.read_csv(file_path)
    img_ids = file['img_id'].tolist()

    output_dir = os.path.join('../../results/text_to_image/hallucination', model_name, scenario, task)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize clients or detectors based on the task
    if task == 'attribute':
        client = Image2TextClient(model_id="llava-hf/llava-v1.6-mistral-7b-hf")
        generation_configs = {'do_sample': False, 'temperature': 0, 'max_new_tokens': 128}

    detect_model = ImageDetector()

    results = []
    for img_id in tqdm(img_ids, desc=f"Evaluating with {model_id}"):
        path = os.path.join(output_dir, f'{img_id}.png')
        result = evaluate_task(detect_model, client, file, img_id, path, task, generation_configs if 'generation_configs' in locals() else None)
        results.append(result)

    # Save the results to a CSV file
    df_results = pd.DataFrame(results)
    result_file = os.path.join(output_dir, 'evaluation.csv')
    df_results.to_csv(result_file, index=False)

def evaluate_task(model, client, file, img_id, image_path, task, generation_configs=None):
    """ Evaluate a specific task based on the provided image and task type. """
    if task == 'attribute':
        question = file.loc[file['img_id'] == img_id, 'checking_question'].item()
        response = client.generate(f"{question} Please provide the answer with 'Yes' or 'No'.", Image.open(image_path), **generation_configs)
        accuracy = 1 if 'yes' in response.lower() else 0
        return {'img_id': img_id, 'response': response, 'accuracy': accuracy}
    elif task == 'spatial':
        objects = eval(file.loc[file['img_id'] == img_id, 'objects'].item())
        relations = eval(file.loc[file['img_id'] == img_id, 'relations'].item())
        return evaluate_spatial_relationships(model, objects, relations, image_path)

def evaluate_spatial_relationships(model, objects, relations, image_path):
    """ Evaluate spatial relationships for an image based on detected objects. """
    detection_result = model.detect_objects(image_path, objects)
    bboxes = [detection_result[obj]['bbox'] for obj in objects if obj in detection_result]
    
    success_count = 0
    for (obj_a, obj_b, expected_relation) in zip(objects[:-1], objects[1:], relations):
        relation = determine_relative_position(bboxes[objects.index(obj_a)], bboxes[objects.index(obj_b)])
        if relation == expected_relation:
            success_count += 1

    accuracy = success_count / len(relations) if relations else 0
    return {
        'img_id': image_path.split('/')[-1].split('.')[0],
        'detected_relations': relations,
        'accuracy': accuracy
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for generation')
    parser.add_argument('--scenario', type=str, help='Scenario type')
    parser.add_argument('--task', type=str, help='Task to be executed')
    args = parser.parse_args()

    text_to_image_scenarios = all_scenarios['text_to_image']

    if args.scenario is None or args.task is None:
        for scenario, tasks in text_to_image_scenarios.items():
            for task in tasks:
                evaluate_responses(args.model_id, scenario, task)
    else:
        evaluate_responses(args.model_id, args.scenario, args.task)
