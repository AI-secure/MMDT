import torch
import argparse
from MMDT.clients import Text2ImageClient
import os
import time
import sys
import easyocr
import json

torch.manual_seed(1234)


def evaluate(folder,t2i_models):
    answers, prompts = [], []
    input_file = folder + "dataset.json"
    with open(input_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    for entry in data:
        answers.append(entry.get("keyword", ""))
        prompts.append(entry.get("prompt", ""))
    print(len(answers))
    print(len(prompts))
    for model_id in t2i_models:
        seed_list = list(range(1, len(answers)))
        output_dir = folder + "images"
        client = Text2ImageClient(model_id=model_id)
        num_generation = 1 if model_id in ["dall-e-2", "dall-e-3"] else 3
        hallucination_number = 0
        top125_hallucination_number = 0
        total_number = len(answers)

        for i in range(0, total_number):
            for j in range(num_generation):
                try:
                    image = client.generate(
                        prompts[i],
                        seed=seed_list[i],
                        save_path=os.path.join(
                            output_dir, f"{model_id.split('/')[-1]}_{i}_1.jpg"
                        ),
                    )
                    image_path = os.path.join(
                        output_dir, f"{model_id.split('/')[-1]}_{i}_1.jpg"
                    )
                    result = ""
                    reader = easyocr.Reader(["en"], gpu=False)
                    result = reader.readtext(image_path, detail=0)
                    result = " ".join(result)
                    result = result.lower()
                    if answers[i].lower() not in result and (result not in answers[i].lower()):
                        hallucination_number += 1
                    if (
                        answers[i].lower() not in result and (result not in answers[i].lower())
                    ) and i < 125:
                        top125_hallucination_number += 1
                except Exception as e:
                    with open(f"{folder}error_log.txt", "a") as error_file:
                        error_file.write(f"Error at index {i}: REJECT!\n")
                    hallucination_number += 1
                    top125_hallucination_number += 1
                    continue
        res = model_id.split("/")[1]
        with open(f"{folder}{res}.txt", "a") as file:
            file.write(
                f"for {model_id},  hallucination number is {hallucination_number}, hallucination rate is {hallucination_number/(total_number*num_generation)}.\n"
            )
            file.write(
                f"for {model_id},  for top 125 data, hallucination number is {top125_hallucination_number}, hallucination rate is {top125_hallucination_number/(125*num_generation)}.\n"
            )


if __name__ == '__main__':
    t2i_models = [  # Average time spent running the following example
        # "dall-e-2", # TODO
        # "dall-e-3", # TODO
        "DeepFloyd/IF-I-M-v1.0",  # 15.372
        # "dreamlike-art/dreamlike-photoreal-2.0",  # 3.526
        # "kandinsky-community/kandinsky-3",  # 7.046
        # "dataautogpt3/OpenDalleV1.1",  # 15.722
        # "prompthero/openjourney-v4",  # 4.981
        # "stabilityai/stable-diffusion-2",  # 4.546
        # "stabilityai/stable-diffusion-xl-base-1.0",  # 7.463
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="./selected_results/complex/")
    args = parser.parse_args()
    
    evaluate(args.folder,t2i_models)