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
                    if category == "image-to-text":
                        try:
                            df = pd.read_csv(csv_path)
                            score = df['accuracy'].mean()  # average of 'accuracy' column
                        except Exception as e:
                            print(f"Error reading CSV file {csv_path}: {e}")
                            score = None
                    elif category == "text-to-image":
                        try:
                            df = pd.read_csv(csv_path)
                            if task in ["identification", "count"]:
                                score = df['soft_accuracy'].mean()  # average of 'soft_accuracy' for count and identification
                            elif task in ["attribute", "spatial"]:
                                score = df['strict_accuracy'].mean()  # average of 'strict_accuracy' for attribute and spatial
                            else:
                                raise ValueError(f"Unknown task: {task}")
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
    """
    Reads robust accurate rate scores for adv evaluation from two modalities:
    image_to_text and text_to_image. The results are organized as:

      {
         "image_to_text": { model_id: score, ... },
         "text_to_image": { model_id: score, ... }
      }

    Directory structure assumed:
      result_dir/
        image_to_text/
          <model_id with potential subdirectories>/
            object/ (statistic_result.txt)
            attribute/ (statistic_result.txt)
            spatial/ (statistic_result.txt)
        text_to_image/
          <model_id with potential subdirectories>/
            object/ (statistic_result.txt)
            attribute/ (statistic_result.txt)
            spatial/ (statistic_result.txt)

    When breakdown is True, returns a dict mapping each model_id to a dict of available task scores.
    When breakdown is False:
      - For each model, if all three tasks have a score, returns the aggregated average (a float).
      - Otherwise, that model is omitted from the aggregated results.
    """
    import os

    modalities = ["image_to_text", "text_to_image"]
    tasks = ["object", "attribute", "spatial"]
    results = {}

    for modality in modalities:
        modality_path = os.path.join(result_dir, modality, 'adv')
        modality_scores = {}
        if not os.path.exists(modality_path):
            continue
        # Use os.walk to traverse subdirectories (to handle model IDs with '/')
        for root, dirs, files in os.walk(modality_path):
            # Consider 'root' as a model directory if it directly contains at least one task folder.
            if any(task in dirs for task in tasks):
                # Compute model_id as the relative path from modality_path with "/" as separator.
                model_id = os.path.relpath(root, modality_path).replace(os.sep, "/")
                breakdown_avg = {}
                for task in tasks:
                    task_dir = os.path.join(root, task)
                    stat_file = os.path.join(task_dir, "statistic_result.txt")
                    if os.path.exists(stat_file):
                        with open(stat_file, "r") as f:
                            lines = f.readlines()
                        rate = None
                        for line in lines:
                            if "Robust accurate rate:" in line:
                                try:
                                    rate_str = line.split("Robust accurate rate:")[1].strip()
                                    rate = float(rate_str)
                                except Exception as e:
                                    print(f"Error parsing rate in {stat_file}: {e}")
                                break
                        if rate is not None:
                            breakdown_avg[task] = rate
                if breakdown:
                    modality_scores[model_id] = breakdown_avg
                else:
                    # Only report an aggregated score if all three tasks are available.
                    if len(breakdown_avg) == 3:
                        aggregated = sum(breakdown_avg[task] for task in tasks) / 3
                        modality_scores[model_id] = aggregated
        results[modality] = modality_scores
    return results


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
