import json, jsonlines
import os, sys
import sys
sys.path.append("./")
sys.path.append("./evaluation")

from clients import Text2ImageClient
import json
from tqdm import tqdm
import base64
import replicate
import copy
from huggingface_hub import login

# login()

# seed = 42

# model_id = "DeepFloyd/IF-I-M-v1.0"
# # model_id = "dataautogpt3/OpenDalleV1.1"
# # model_id = "stabilityai/stable-diffusion-2"
# # model_id = "dataautogpt3/OpenDalleV1.1"


# # client = Text2ImageClient(model_id="/home/czr/.cache/huggingface/hub/sd2/snapshots/1e128c8891e52218b74cde8f26dbfc701cb99d79")
# client = Text2ImageClient(model_id=model_id)


def generate_cooc_text_to_image(model_id, task, client, seed, output_dir):

    response_save_root = "co-occurrence/evaluation/result_t2ichallenging/"

    model_identifier = model_id.split("/")[-1]

    if not os.path.exists(os.path.join(response_save_root, model_identifier)):
        os.makedirs(os.path.join(response_save_root, model_identifier))


    result_dict = {}

    # base_folder = "co-occurrence/final_dataset/text-to-image"
    base_folder = "co-occurrence/final_dataset/challenging_dataset_t2i"

    high_cooc_folder = os.path.join(base_folder, "high_cooc")
    low_cooc_folder = os.path.join(base_folder, "low_cooc")
    historical_foler = os.path.join(base_folder, "historical_bias")

    # all_category_list = ["object_existence", "count", "attribute", "spatial"]
    all_category_list = [task]

    overall_counter = 0


    for folder, type_folder in zip([high_cooc_folder, low_cooc_folder, historical_foler], ["high_cooc", "low_cooc", "historical_bias"]):
        
        # if type_folder != "historical_bias":
        #     continue

        folder_dict = {}

        for category in all_category_list:

            prompt_dir = os.path.join(folder, category) + ".json"

            category_list = []

            print("prompt_dir", prompt_dir)
            with jsonlines.open(prompt_dir, "r") as f:
                data_source = []
                for idx, obj in enumerate(f):

                    data_source.append(obj)

            print(f"Processing {category} prompts")

            for i_idx, item in tqdm(enumerate(data_source), total=len(data_source)):


                new_item = copy.deepcopy(item)
                prompt = item["prompt"]

                response_image = client.generate(prompt, seed)
                # print(response)

                output_dir = os.path.join(response_save_root, model_identifier) + f"/{type_folder}_{category}_{i_idx}.png"

                response_image.save(output_dir)

                new_item["response_image"] = output_dir

                category_list.append(new_item)

                overall_counter += 1

            folder_dict[category] = category_list

            print(f"Overall Counter: {overall_counter}")

        result_dict[type_folder] = folder_dict

        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(result_dict, f, indent=4)


# print(f"Experiment done: {overall_counter}")