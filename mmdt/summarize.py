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
    modalities = ["image_to_text", "text_to_image"]
    modality2tasks = {
        "image_to_text": ['typography', 'illustration', 'jailbreak'],
        "text_to_image": ['vanilla', 'transformed', 'jailbreak']
    }
    results = {}

    for modality in modalities:
        modality_path = os.path.join(result_dir, modality, 'safety')
        modality_scores = {}
        if not os.path.exists(modality_path):
            continue
        # Use os.walk to traverse subdirectories (to handle model IDs with '/')
        for root, dirs, files in os.walk(modality_path):
            # Consider 'root' as a model directory if it directly contains at least one task folder.
            if any(task in dirs for task in modality2tasks[modality]):
                # Compute model_id as the relative path from modality_path with "/" as separator.
                model_id = os.path.relpath(root, modality_path).replace(os.sep, "/")
                breakdown_avg = {}
                for task in modality2tasks[modality]:
                    task_dir = os.path.join(root, task)
                    stat_file = os.path.join(task_dir, "statistic_result.txt")
                    if os.path.exists(stat_file):
                        with open(stat_file, "r") as f:
                            result = json.load(f)
                        breakdown_avg[task] = result['HGR']
                if breakdown:
                    modality_scores[model_id] = breakdown_avg
                else:
                    # Only report an aggregated score if all three tasks are available.
                    if len(breakdown_avg) == 3:
                        aggregated = sum(breakdown_avg[task] for task in tasks) / 3
                        modality_scores[model_id] = aggregated
        results[modality] = modality_scores
    return results


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
    """
    Reads OOD (Out of Distribution) scores for both text-to-image and image-to-text modalities.
    
    For text-to-image, evaluates:
    - Shake scenarios (helpfulness, counting, spatial, attribute)
    - Paraphrase/rare scenarios (helpfulness, counting, spatial, attribute)
    
    For image-to-text, evaluates:
    - Corruption scenarios (object, counting, spatial, attribute)
    - Transform scenarios (object, counting, spatial, attribute)
    
    Directory structure assumed:
      result_dir/
        text-to-image/ood/
          <model_id>/
            summary_0.json, summary_1.json, summary_2.json
        image-to-text/ood/
          <model_id>/
            summary.json

    When breakdown is True, returns per-subscenario scores.
    When breakdown is False, returns the overall aggregated score.
    """
    def aggregate_t2i_scores(result_path):
        """Internal function to aggregate text-to-image OOD scores"""
        with open(os.path.join(result_path, "summary_0.json"), "r") as file:
            results = json.load(file)
        model_ids = list(results.keys())
        results_scores = {}
        results_subscenarios = {}
        for model_id in model_ids:
            results_shake = {"fidelity": 0, "counting": 0, "spatial": 0, "color": 0, "size": 0}
            results_paraphrase = {"fidelity": 0, "counting": 0, "spatial": 0, "color": 0, "size": 0}
            
            # Average over 3 trials
            for trial_id in [0, 1, 2]:
                summary_path = os.path.join(result_path, f"summary_{trial_id}.json")
                with open(summary_path, "r") as file:
                    results = json.load(file)
                
                
                # Accumulate scores for shake scenarios
                results_shake["fidelity"] += results[model_id]['helpfulness']['Shake_']
                results_shake["counting"] += results[model_id]['count']['Shake_']
                results_shake["spatial"] += results[model_id]['spatial']['Shake_']
                results_shake["color"] += results[model_id]['color']['Shake_']
                results_shake["size"] += results[model_id]['size']['Shake_']
                
                # Accumulate scores for paraphrase scenarios
                results_paraphrase["fidelity"] += results[model_id]['helpfulness']['Paraphrase_']
                results_paraphrase["counting"] += results[model_id]['count']['Paraphrase_']
                results_paraphrase["spatial"] += results[model_id]['spatial']['Paraphrase_']
                results_paraphrase["color"] += results[model_id]['color']['Paraphrase_']
                results_paraphrase["size"] += results[model_id]['size']['Paraphrase_']

            # Average the accumulated scores
            for key in results_shake:
                results_shake[key] /= 3
                results_paraphrase[key] /= 3
            
            # Convert fidelity scores to percentages
            results_shake["fidelity"] *= 100
            results_paraphrase["fidelity"] *= 100
            
            # Calculate attribute scores (average of color and size)
            shake_attribute = (results_shake["color"] + results_shake["size"]) / 2
            paraphrase_attribute = (results_paraphrase["color"] + results_paraphrase["size"]) / 2
            
            # Calculate final averages
            avg_shake = (results_shake["fidelity"] + results_shake["counting"] + 
                        results_shake["spatial"] + shake_attribute) / 4
            avg_paraphrase = (results_paraphrase["fidelity"] + results_paraphrase["counting"] + 
                            results_paraphrase["spatial"] + paraphrase_attribute) / 4
            
            results_scores[model_id] = (avg_shake + avg_paraphrase) / 2
            results_subscenarios[model_id] = {
                "helpfulness_shake": results_shake["fidelity"],
                "counting_shake": results_shake["counting"],
                "spatial_shake": results_shake["spatial"],
                "attribute_shake": shake_attribute,
                "helpfulness_rare": results_paraphrase["fidelity"],
                "counting_rare": results_paraphrase["counting"],
                "spatial_rare": results_paraphrase["spatial"],
                "attribute_rare": paraphrase_attribute
            }
            
        return {"score": results_scores, "subscenarios": results_subscenarios}

    def aggregate_i2t_scores(result_path):
        """Internal function to aggregate image-to-text OOD scores"""
        transformations = ["Van_Gogh", "oil_painting", "watercolour_painting"]
        corruptions = ["zoom_blur", "gaussian_noise", "pixelate"]
        
        summary_path = os.path.join(result_path, "summary.json")
        with open(summary_path, "r") as file:
            results = json.load(file)
        model_ids = list(results.keys())
        results_scores = {}
        results_subscenarios = {}
        for model_id in model_ids:
        
            # Calculate corruption scores
            identification_corrupt = sum(results[model_id]['identification'][corrupt] for corrupt in corruptions) / 3
            count_corrupt = sum(results[model_id]['count'][corrupt] for corrupt in corruptions) / 3
            spatial_corrupt = sum(results[model_id]['spatial'][corrupt] for corrupt in corruptions) / 3
            attribute_corrupt = sum(results[model_id]['attribute'][corrupt] for corrupt in corruptions) / 3
            avg_corrupt = (identification_corrupt + count_corrupt + spatial_corrupt + attribute_corrupt) / 4
            
            # Calculate transformation scores
            identification_transform = sum(results[model_id]['identification'][transform] for transform in transformations) / 3
            count_transform = sum(results[model_id]['count'][transform] for transform in transformations) / 3
            spatial_transform = sum(results[model_id]['spatial'][transform] for transform in transformations) / 3
            attribute_transform = sum(results[model_id]['attribute'][transform] for transform in transformations) / 3
            avg_transform = (identification_transform + count_transform + spatial_transform + attribute_transform) / 4
            results_scores[model_id] = (avg_corrupt + avg_transform) / 2
            results_subscenarios[model_id] = {
                "object_corrupt": identification_corrupt,
                "counting_corrupt": count_corrupt,
                "spatial_corrupt": spatial_corrupt,
                "attribute_corrupt": attribute_corrupt,
                "object_transform": identification_transform,
                "counting_transform": count_transform,
                "spatial_transform": spatial_transform,
                "attribute_transform": attribute_transform
            }

        return {"score": results_scores, "subscenarios": results_subscenarios}

    modalities = {
        "text-to-image": "text-to-image",
        "image-to-text": "image-to-text"
    }
    results = {}

    for modality_key, modality_path in modalities.items():
        base_path = os.path.join(result_dir, modality_path, "ood")
        modality_scores = {}
        
        if not os.path.exists(base_path):
            continue

        # Use os.walk to traverse subdirectories (to handle model IDs with '/')
        for root, dirs, files in os.walk(base_path):
            # For text-to-image, look for summary_0.json
            # For image-to-text, look for summary.json
            if modality_key == "text-to-image" and "summary_0.json" in files:
                # try:
                if True:
                    scores = aggregate_t2i_scores(root)
                    if breakdown:
                        modality_scores = scores["subscenarios"]
                    else:
                        modality_scores = scores["score"]
                # except Exception as e:
                #     print(f"Error processing text-to-image OOD for {model_id}: {e}")
                #     continue
                    
            elif modality_key == "image-to-text" and "summary.json" in files:
                try:
                    scores = aggregate_i2t_scores(root)
                    if breakdown:
                        modality_scores = scores["subscenarios"]
                    else:
                        modality_scores = scores["score"]
                except Exception as e:
                    print(f"Error processing image-to-text OOD for {model_id}: {e}")
                    continue

        results[modality_key] = modality_scores

    return results


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
