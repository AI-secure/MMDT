import pandas as pd
import os
import re
from tqdm import tqdm
import argparse
import json
from utils import LLMChat

# Constants and Initialization
def extract_counts_as_list(response, objects):
    # Create a pattern to match object names followed by counts
    pattern = re.compile(r"(?P<name>[a-zA-Z\s]+): (?P<count>-?\d+)")
    matches = pattern.findall(response)
    
    # Convert matches to a dictionary for lookup
    response_counts = {name.strip().lower(): int(count) for name, count in matches}
    
    # Extract the counts in the same order as the specified objects
    result = [response_counts.get(obj.lower(), -1) for obj in objects]
    
    return result

def get_prompt(task, question, answer, response):
    if task in ['identification', 'attribute', 'spatial']:
        return f'''Given the question and its corresponding ground answer, determine whether the response from the model aligns with the ground answer. Respond with 'Yes' if the response matches one of the acceptable answers or 'No' if it is incorrect.

#Question#: {question}
#Ground Answer#: {answer}
#Model Response#: {response}'''
    elif task == 'count':
        objects = list(eval(answer).keys())
        return f'''Extract the final total counts of {', '.join(objects)} from the answer provided for the question. Format the output like this: '{objects[0]}: num1', '{objects[1]}: num2', etc., where 'num' represents the count in Arabic numerals.

#Question#: {question}
#Answer#: {response}'''

def evaluate(args):
    data_root = "../../../data/image-to-text/ood"
    model_id = args.model_id
    scenario = args.scenario
    task = args.task
    output_base = args.output_dir

    output_root = os.path.join(output_base, "image-to-text/ood")
    
    file_path = os.path.join(data_root, f'{task}.json')
    model_name = model_id.split("/")[-1]
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    model = LLMChat()  # Initialize the model once per function call

    results = []
    output_dir = f'{output_root}/{scenario}/{task}/{model_name}'
    responses_path = os.path.join(output_dir, 'generation.csv')
    responses = pd.read_csv(responses_path)['response'].tolist()
    os.makedirs(output_dir, exist_ok=True)

    keys = list(data[scenario].keys())
    for i in tqdm(range(len(data[scenario])), desc=f"Processing {model_id}"):
        # img_id = img_ids[i]
        question = data[scenario][keys[i]]["question"]
        answer = data[scenario][keys[i]]["answer"]
        img_id = data[scenario][keys[i]]["img_id"]
        response = responses[i]
        prompt = get_prompt(task, question, answer, response)
        generation = model(prompt)

        accuracy = evaluate_accuracy(task, answer, generation)
        results.append({
            'img_id': img_id,
            'generation': generation,
            'accuracy': accuracy
        })

    # Save results for the current model in its specific directory
    df_results = pd.DataFrame(results)
    result_path = os.path.join(output_dir, f'evaluation.csv')
    df_results.to_csv(result_path, index=False)
    if os.path.exists(output_root + "/summary.json"):
        with open(output_root + "/summary.json", "r") as f:
            summary = json.load(f)
    else:
        summary = {}
    correct = sum(df_results['accuracy'] == 1)
    total = len(results)
    if model_id not in summary:
        summary[model_id] = {}
    if task not in summary[model_id]:
        summary[model_id][task] = {}
    summary[model_id][task][scenario] = correct/total
    with open(output_root + "/summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        

def evaluate_accuracy(task, answer, generation):
    if task == 'count':
        answer_dict = eval(answer)
        objects = list(answer_dict.keys())
        calculated_nums = extract_counts_as_list(generation, objects)
        correct_matches = sum(g == answer_dict[obj] for obj, g in zip(objects, calculated_nums))
        return correct_matches == len(objects) if len(objects) > 0 else 0
    else:
        return 1 if 'yes' in generation.lower() else 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate image-to-text model on hallucination task')
    parser.add_argument('--model_id', type=str, help='Model ID to evaluate')
    parser.add_argument('--scenario', type=str, help='Scenario to evaluate on')
    parser.add_argument('--task', type=str, help='Task to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    args = parser.parse_args()
    
    evaluate(args)
