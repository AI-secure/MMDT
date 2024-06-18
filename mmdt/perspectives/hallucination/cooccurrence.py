import json, jsonlines
import os, sys
import sys
import json
from tqdm import tqdm
import base64
# import replicate
import copy
# from huggingface_hub import login


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


def generate_cooc_image_to_text(model_id, task, client, generation_configs, output_dir):

    model_identifier = model_id.split("/")[-1]


    result_dict = {}

    base_folder = "co-occurrence/final_dataset/challenging_dataset_75"

    high_cooc_folder = os.path.join(base_folder, "high_cooc")
    low_cooc_folder = os.path.join(base_folder, "low_cooc")
    historical_foler = os.path.join(base_folder, "historical_bias")

    # all_category_list = ["object_existence", "count", "attribute", "spatial", "action"]
    all_category_list = [task]

    overall_counter = 0

    image_base_dir = "co-occurrence/final_dataset/dataset_img"

    for folder, type_folder in zip([high_cooc_folder, low_cooc_folder, historical_foler], ["high_cooc", "low_cooc", "historical_bias"]):
        
        folder_dict = {}

        image_folder = os.path.join(folder, "images")
        prompt_folder = os.path.join(folder, "prompts")

        for category in all_category_list:

            image_dir = os.path.join(image_folder, category)
            prompt_dir = os.path.join(prompt_folder, category) + ".json"

            category_list = []

            print("prompt_dir", prompt_dir)
            with jsonlines.open(prompt_dir, "r") as f:
                data_source = []
                for idx, obj in enumerate(f):

                    data_source.append(obj)

            print(f"Processing {category} prompts")

            for i_idx, item in tqdm(enumerate(data_source), total=len(data_source)):

                new_item = {}
                prompt = item["prompt"]
                img_dir = image_base_dir + "/" + item["image_path"]
                # img_dir = item["image_path"]

                response = client.generate(prompt, img_dir, **generation_configs)
                # response = replicate_api(img_dir, prompt)
                # print(response)

                # check if img_dir exists
                if not os.path.exists(img_dir):
                    print(f"Image not found: {img_dir}")
                    input()

                new_item["target"] = item["target"]
                new_item["label"] = item["label"]
                if "idx" in item:
                    new_item["idx"] = item["idx"]
                else:
                    new_item["idx"] = i_idx
                new_item["image_path"] = img_dir

                new_item["response"] = response

                category_list.append(new_item)

                overall_counter += 1

            folder_dict[category] = category_list

            print(f"Overall Counter: {overall_counter}")

        result_dict[type_folder] = folder_dict

        with open(f"{output_dir}/result.json", "w") as f:
            json.dump(result_dict, f, indent=4)