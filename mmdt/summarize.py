import os
import json

import numpy as np
import pandas as pd
from glob import glob


def sort_keys(obj):
    if isinstance(obj, dict):
        return {k: sort_keys(obj[k]) for k in sorted(obj.keys())}
    elif isinstance(obj, list):
        return [sort_keys(element) for element in obj]
    else:
        return obj


def get_safety_scores(result_dir="./results", breakdown=False):
    pass


def get_hallucination_scores(result_dir="./results", breakdown=False):
    aggregated_results = {"text_to_image": {}, "image_to_text": {}}
    breakdown_results = {"text_to_image": {}, "image_to_text": {}}

    # Categories to process
    categories = ["text_to_image", "image_to_text"]

    # Iterate through each category
    for category in categories:
        category_path = os.path.join(result_dir, category, "hallucination")
        
        # Iterate through each model's results directory
        for model_dir in glob(os.path.join(category_path, '*/*/*')):
            path_parts = model_dir.split(os.path.sep)
            model_id = path_parts[-3]
            scenario_type = path_parts[-2]
            task = path_parts[-1]

            # Load the results CSV
            results_file = os.path.join(model_dir, 'evaluation.csv')
            if os.path.exists(results_file):
                df = pd.read_csv(results_file)
                avg_accuracy = df['accuracy'].mean() * 100

                # Build the breakdown_results
                if model_id not in breakdown_results[category]:
                    breakdown_results[category][model_id] = {}
                if scenario_type not in breakdown_results[category][model_id]:
                    breakdown_results[category][model_id][scenario_type] = {}
                breakdown_results[category][model_id][scenario_type][task] = avg_accuracy

                # Aggregate the results for the scenario
                if model_id not in aggregated_results[category]:
                    aggregated_results[category][model_id] = {}
                if scenario_type not in aggregated_results[category][model_id]:
                    aggregated_results[category][model_id][scenario_type] = []
                aggregated_results[category][model_id][scenario_type].append(avg_accuracy)

        # Calculate scenario averages
        for model, scenarios in aggregated_results[category].items():
            for scenario, accuracies in scenarios.items():
                aggregated_results[category][model][scenario] = sum(accuracies) / len(accuracies)

    if breakdown:
        return breakdown_results
    else:
        return aggregated_results

def get_fairness_scores(result_dir="./results", breakdown=False):
    pass


def get_privacy_scores(result_dir="./results", breakdown=False):
    pass


def get_adv_scores(result_dir="./results", breakdown=False):
    pass


def get_ood_scores(result_dir="./results", breakdown=False):
    pass


def summarize_results(result_dir="./results"):
    summarized_results = {  # aggregated_results --> perspective --> modality --> model_id --> score
        "aggregated_results": {
            "safety": get_safety_scores(result_dir=result_dir),
            "hallucination": get_hallucination_scores(result_dir=result_dir),
            "fairness": get_fairness_scores(result_dir=result_dir),
            "privacy": get_privacy_scores(result_dir=result_dir),
            "adv": get_adv_scores(result_dir=result_dir),
            "ood": get_ood_scores(result_dir=result_dir),
        },
        "breakdown_results": {  # breakdown_results --> perspective --> modality --> model_id --> sub_scores
            "safety": get_safety_scores(result_dir=result_dir, breakdown=True),
            "hallucination": get_hallucination_scores(result_dir=result_dir, breakdown=True),
            "fairness": get_fairness_scores(result_dir=result_dir, breakdown=True),
            "privacy": get_privacy_scores(result_dir=result_dir, breakdown=True),
            "adv": get_adv_scores(result_dir=result_dir, breakdown=True),
            "ood": get_ood_scores(result_dir=result_dir, breakdown=True),
        }
    }

    summarized_results = sort_keys(summarized_results)

    with open(os.path.join(result_dir, "summary.json"), "w") as f:
        json.dump(summarized_results, f, indent=4)
        print(json.dumps(summarized_results["aggregated_results"], indent=4))

    return summarized_results


if __name__ == "__main__":
    summarize_results(result_dir="./results")
