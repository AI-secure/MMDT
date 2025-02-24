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
    """
    Collects accuracy scores for hallucination tasks for both text-to-image and image-to-text.

    Scenarios and file formats:
      1) Scenarios with CSV (evaluation.csv):
         ["natural", "distraction", "counterfactual"]
      2) Scenarios with JSON containing top-level "accuracy":
         ["misleading", "ocr"]
      3) Scenario "cooccurrence":
         - image-to-text: the JSON has a key matching the 'task' name, e.g. {"action": 0.2, ...}
         - text-to-image: the JSON has sub-keys (e.g. "high_cooc", "low_cooc", "historical_bias")
           each containing {"overall": <score>}. We take the average of the "overall" values.

    The final data structure has:
        aggregated_results[category][model_id][scenario] = average accuracy across tasks
    If breakdown=True, you get a deeper dictionary with per-task accuracies.

    If a file doesn't exist for a given scenario/task, None is recorded.
    """

    # Define which scenarios produce CSV or JSON and how they are parsed
    csv_scenarios = {"natural", "distraction", "counterfactual"}
    json_scenarios_accuracy = {"misleading", "ocr"}
    json_scenarios_keyed = {"cooccurrence"}

    # We handle both categories
    categories = ["image-to-text", "text-to-image"]

    # Prepare containers
    aggregated_results = {cat: {} for cat in categories}
    breakdown_results = {cat: {} for cat in categories}

    # Iterate over both categories
    for category in categories:
        category_path = os.path.join(result_dir, category, "hallucination")
        
        # Example folder structure:
        #   ./results/<category>/hallucination/<model>/<scenario>/<task>/(evaluation.csv or result.json)
        for model_dir in glob(os.path.join(category_path, '*/*/*')):
            path_parts = model_dir.split(os.path.sep)

            task = path_parts[-1]
            scenario_type = path_parts[-2]
            model_id = path_parts[-3]

            # Initialize dict structure if needed
            if model_id not in breakdown_results[category]:
                breakdown_results[category][model_id] = {}
            if scenario_type not in breakdown_results[category][model_id]:
                breakdown_results[category][model_id][scenario_type] = {}

            if model_id not in aggregated_results[category]:
                aggregated_results[category][model_id] = {}
            if scenario_type not in aggregated_results[category][model_id]:
                aggregated_results[category][model_id][scenario_type] = []

            # Identify potential file paths
            csv_path = os.path.join(model_dir, 'evaluation.csv')
            json_path = os.path.join(model_dir, 'result.json')

            score = None

            # 1) CSV scenarios
            if scenario_type in csv_scenarios:
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        score = df['accuracy'].mean()  # average of 'accuracy' column
                    except Exception as e:
                        print(f"Error reading CSV file {csv_path}: {e}")
                        score = None
                else:
                    score = None

            # 2) JSON with top-level "accuracy"
            elif scenario_type in json_scenarios_accuracy:
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    score = data.get("accuracy", None)
                else:
                    score = None

            # 3) JSON "cooccurrence" scenario
            elif scenario_type in json_scenarios_keyed:
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    # image-to-text: the JSON might look like {"action": 0.2, "high_cooc": 0.4, ...}
                    if category == "image-to-text":
                        score = data.get(task, None)

                    elif category == "text-to-image":
                        cooc_keys = ["high_cooc", "low_cooc", "historical_bias"]
                        overall_vals = []
                        for ck in cooc_keys:
                            if ck in data and "overall" in data[ck]:
                                overall_vals.append(data[ck]["overall"])
                        if overall_vals:
                            score = sum(overall_vals) / len(overall_vals)
                        else:
                            score = None
                else:
                    score = None

            # Store in the breakdown dictionary
            breakdown_results[category][model_id][scenario_type][task] = score

            # Also collect for aggregated results
            if score is not None:
                aggregated_results[category][model_id][scenario_type].append(score)

        # After gathering all tasks, compute average per scenario
        for m_id, scenarios_dict in aggregated_results[category].items():
            for s_type, scores_list in scenarios_dict.items():
                if len(scores_list) == 0:
                    aggregated_results[category][m_id][s_type] = None
                else:
                    aggregated_results[category][m_id][s_type] = sum(scores_list) / len(scores_list)

    # Return breakdown or aggregated
    return breakdown_results if breakdown else aggregated_results
    


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
        print(json.dumps(summarized_results["breakdown_results"], indent=4))

    return summarized_results


if __name__ == "__main__":
    summarize_results(result_dir="./results")
