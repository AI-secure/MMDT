import os
import argparse
import pandas as pd
from tqdm import tqdm
from mmdt.models import Image2TextClient
from mmdt.detection import ImageDetector
from .utils import determine_relative_position
from PIL import Image
from mmdt.perspectives.hallucination.scenario_list import all_scenarios
from mmdt.perspectives.hallucination.cooccurrence import evaluate_cooc_text_to_image
from mmdt.perspectives.hallucination.ocr import evaluate_ocr_text_to_image
from mmdt.perspectives.hallucination.misleading import evaluate_misleading_text_to_image
from mmdt.perspectives.hallucination.spatial import evaluate_spatial_relationships
from datasets import load_dataset

def evaluate(kwargs):

    if not os.path.exists('./mmdt/detection/GroundingDINO/weights'):
        os.makedirs('./mmdt/detection/GroundingDINO/weights')
    if not os.path.exists('./mmdt/detection/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth'):
        os.system('wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth -O ./mmdt/detection/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth')

    model_id, scenario, task = kwargs.model_id, kwargs.scenario, kwargs.task
    model_name = model_id.replace('/', '_')
    if scenario == "ocr":
        evaluate_ocr_text_to_image(model_name, scenario, task)
        return
    elif scenario == "misleading":
        evaluate_misleading_text_to_image(model_name, scenario, task)
        return
    elif scenario == "cooccurrence":
        evaluate_cooc_text_to_image(model_name, scenario, task)
        return

    ds = load_dataset("AI-Secure/MMDecodingTrust-T2I", "hallucination")
    data = ds[scenario].filter(lambda x: x['task'] == task)

    output_dir = os.path.join('results/text-to-image/hallucination', model_name, scenario, task)
    os.makedirs(output_dir, exist_ok=True)

    client = Image2TextClient(model_id="llava-hf/llava-v1.6-mistral-7b-hf")
    generation_configs = {'do_sample': False, 'temperature': 0, 'max_new_tokens': 128}

    results = []
    for idx, current_data in tqdm(enumerate(data), desc=f"Evaluating with {model_id}", total=len(data)):

        img_id = current_data['id']
        path = os.path.join(output_dir, f"{img_id}.png")
        result = evaluate_task(client, current_data, img_id, path, task, generation_configs)
        results.append(result)
        
    # Save the results to a CSV file
    df_results = pd.DataFrame(results)
    result_file = os.path.join(output_dir, 'evaluation.csv')
    df_results.to_csv(result_file, index=False)

def evaluate_task(client, current_data, img_id, image_path, task, generation_configs=None):
    """ Evaluate a specific task based on the provided image and task type. """
    if task == 'attribute':
        question = current_data['checking_question']
        response = client.generate(f"{question} Please provide the answer with 'Yes' or 'No'.", image_path, **generation_configs)
        accuracy = 1 if 'yes' in response.lower() else 0
        return {'img_id': img_id, 'response': response, 'accuracy': accuracy}
    elif task == 'spatial':
        model = ImageDetector()
        objects = eval(current_data['objects'])
        relations = (current_data['relations'])
        return evaluate_spatial_relationships(model, objects, relations, image_path)

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
                evaluate(args.model_id, scenario, task)
    else:
        evaluate(args.model_id, args.scenario, args.task)
