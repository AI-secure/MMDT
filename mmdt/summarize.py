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
    pass


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
