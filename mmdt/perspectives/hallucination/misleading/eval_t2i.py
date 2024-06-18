import torch
import argparse
from MMDT.clients import Text2ImageClient
import os
import time
import sys
from utils.image_detector_utils import *
from transformers import (
    LlavaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)
import json

torch.manual_seed(1234)

cache_dir = "/data2/common/peiyang"
processor_llava16 = LlavaNextProcessor.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=cache_dir
)
model_llava16 = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, cache_dir=cache_dir
)
model_llava16.to("cuda:3")


def determine_relative_position(x1, y1, x2, y2):
    if x1 < x2 and abs(x1 - x2) > abs(y1 - y2):
        return ["left"]
    elif x1 > x2 and abs(x1 - x2) > abs(y1 - y2):
        return ["right"]
    elif y1 < y2:
        return ["above", "top", "over"]
    else:
        return ["under"]


def evaluate(folder,type,t2i_models):
    objects = []
    attribute = []
    number = []
    relation_objects = []
    spatial_relation = []
    prompts = []
    detector = ImageDetector()
    input_file = folder + "dataset.json"
    with open(input_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    for entry in data:
        objects.append(entry.get("object", ""))
        attribute.append(entry.get("attribute", ""))
        number.append(entry.get("number", ""))
        relation_objects.append(entry.get("relation_object", ""))
        spatial_relation.append(entry.get("spatial_relation", ""))
        prompts.append(entry.get("prompt", ""))
    length = max(len(objects), len(attribute), len(number), len(relation_objects), len(spatial_relation), len(prompts))
    print(length)
    for model_id in t2i_models:

        seed_list = list(range(1, length))
        output_dir = folder + "images"
        client = Text2ImageClient(model_id=model_id)
        num_generation = 1 if model_id in ["dall-e-2", "dall-e-3"] else 3
        hallucination_number = 0
        top125_hallucination_number = 0
        total_number = len(prompts)

        for i in range(0, length):
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

                    if type == "Attribute":
                        image = Image.open(image_path)
                        question = f"Is the {objects[i]} {attribute[i]}? Please provide the answer with 'Yes' or 'No'"
                        prompt = f"[INST] <image>\n{question} [/INST]"
                        inputs = processor_llava16(prompt, image, return_tensors="pt").to(
                            "cuda:3"
                        )
                        output = model_llava16.generate(**inputs, max_new_tokens=100)
                        answer = processor_llava16.decode(output[0], skip_special_tokens=True)
                        answer = answer.split("[/INST]")[1].strip()
                        print(answer)
                        if "no" in answer.lower():
                            hallucination_number += 1
                        if "no" in answer.lower() and i < 125:
                            top125_hallucination_number += 1

                    elif type == "Existence":
                        res = detector.single_detect(image_path, [f"{objects[i]}"], 0.6)
                        print(res)
                        if res["entity_info"][f"{objects[i]}"]["total_count"] == 0:
                            hallucination_number += 1
                        if res["entity_info"][f"{objects[i]}"]["total_count"] == 0 and i < 125:
                            top125_hallucination_number += 1
                    elif type == "Number":
                        res = detector.single_detect(image_path, [f"{objects[i]}"], 0.6)
                        print(res)
                        if res["entity_info"][f"{objects[i]}"]["total_count"] != int(number[i]):
                            hallucination_number += 1
                        if (
                            res["entity_info"][f"{objects[i]}"]["total_count"] != int(number[i])
                            and i < 125
                        ):
                            top125_hallucination_number += 1
                    elif type == "SpatialRelation":
                        res = detector.single_detect(
                            image_path, [f"{objects[i]}", f"{relation_objects[i]}"]
                        )
                        if (
                            res["entity_info"][f"{objects[i]}"]["total_count"] != 1
                            or res["entity_info"][f"{relation_objects[i]}"]["total_count"] != 1
                        ):
                            hallucination_number += 1
                            if i < 125:
                                top125_hallucination_number += 1
                        else:
                            obj1 = res["entity_info"][f"{objects[i]}"]["bbox"][0]
                            obj2 = res["entity_info"][f"{relation_objects[i]}"]["bbox"][0]
                            x_center1 = (obj1[0] + obj1[2]) / 2
                            x_center2 = (obj2[0] + obj2[2]) / 2
                            y_center1 = (obj1[1] + obj1[3]) / 2
                            y_center2 = (obj2[1] + obj2[3]) / 2
                            result = determine_relative_position(
                                x_center1, y_center1, x_center2, y_center2
                            )
                            if spatial_relation[i] not in result:
                                hallucination_number += 1
                                if i < 125:
                                    top125_hallucination_number += 1

                except Exception as e:
                    with open(folder+"error_log.txt", "a") as error_file:
                        error_file.write(f"Error at index {i}: REJECT!\n")
                    hallucination_number += 1
                    top125_hallucination_number += 1
                    continue

        res = model_id.split("/")[1]
        with open(folder+f"{res}.txt", "a") as file:
            file.write(
                f"for {model_id},  hallucination number is {hallucination_number}, hallucination rate is {hallucination_number/(total_number*num_generation)}.\n"
            )
            file.write(
                f"for {model_id},  for top 125 data, hallucination number is {top125_hallucination_number}, hallucination rate is {top125_hallucination_number/(125*num_generation)}.\n"
            )


if __name__ == '__main__':
    t2i_models = [  # Average time spent running the following example
        # "dall-e-2", #TODO
        # "dall-e-3", #TODO
        "DeepFloyd/IF-I-M-v1.0",  # 15.372
        # "dreamlike-art/dreamlike-photoreal-2.0",  # 3.526
        # "kandinsky-community/kandinsky-3",  # 7.046
        # "dataautogpt3/OpenDalleV1.1",  # 15.722
        # "prompthero/openjourney-v4",  # 4.981
        # "stabilityai/stable-diffusion-2",  # 4.546
        # "stabilityai/stable-diffusion-xl-base-1.0",  # 7.463
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="./selected_results/Existence/")
    parser.add_argument("--type", type=str, default="Existence")
    args = parser.parse_args()
    
    evaluate(args.folder,args.type,t2i_models)