import os
import pandas as pd
from tqdm import tqdm
import argparse
from PIL import Image
from collections import Counter
from utils import LLMChat

def extract_counts_as_list(response, objects):
    import re
    # Create a pattern to match object names followed by counts
    pattern = re.compile(r"(?P<name>[a-zA-Z\s]+): (?P<count>-?\d+)")
    matches = pattern.findall(response)

    # Convert matches to a dictionary for lookup
    response_counts = {name.strip().lower(): int(count) for name, count in matches}

    # Extract the counts in the same order as the specified objects
    result = [response_counts.get(obj.lower(), -1) for obj in objects]

    return result

def evaluate(model_id, scenario, task):
    model_name = model_id.split("/")[1] if '/' in model_id else model_id
    file_path = os.path.join('../../data/image_to_text', scenario, model_id, f'{task}.csv')
    data = pd.read_csv(file_path)
    img_ids = data['img_id'].tolist()
    questions = data['question'].tolist()
    answers = data['answer'].tolist()

    model = LLMChat()  # Initialize the model once per function call
    output_dir = os.path.join('../../results/image_to_text/hallucination', scenario, task, model_name)
    responses_path = os.path.join(output_dir, f'generation.csv')
    responses = pd.read_csv(responses_path)['response'].tolist()

    results = []
    for i in tqdm(range(len(img_ids)), desc=f"Processing {model_id}"):
        img_id = img_ids[i]
        question = questions[i]
        answer = answers[i]
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
    result_file = os.path.join(output_dir, f'evaluation.csv')
    df_results.to_csv(result_file, index=False)

def get_prompt(task, question, answer, response):
    if task in ['identification', 'attribute', 'action']:
        return f'''Given the question and its corresponding ground answer, determine whether the response from the model aligns with the ground answer. Respond with 'Yes' if the response matches one of the acceptable answers or 'No' if it is incorrect.

#Question#: {question}
#Ground Answer#: {answer}
#Model Response#: {response}'''
    elif task == 'count':
        objects = list(eval(answer).keys())
        return f'''Extract the final total counts of {', '.join(objects)} from the answer provided for the question. Format the output like this: '{objects[0]}: num1', '{objects[1]}: num2', etc., where 'num' represents the count in Arabic numerals.

#Question#: {question}
#Answer#: {response}'''

def evaluate_accuracy(task, answer, generation):
    if task == 'count':
        answer_dict = eval(answer)
        objects = list(answer_dict.keys())
        calculated_nums = extract_counts_as_list(generation, objects)
        correct_matches = sum(g == answer_dict[obj] for obj, g in zip(objects, calculated_nums))
        return correct_matches / len(objects) if len(objects) > 0 else 0
    else:
        return 1 if 'yes' in generation.lower() else 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for evaluation')
    parser.add_argument('--scenario', type=str, default="natural", help='Specify the scenario')
    parser.add_argument('--task', type=str, default="identification", help='Specify the task to execute')
    args = parser.parse_args()

    evaluate_responses(args.model_id, args.scenario, args.task)
