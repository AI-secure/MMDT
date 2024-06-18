import json, jsonlines
import os, sys
import sys
sys.path.append("./")
sys.path.append("./evaluation")

from clients import Image2TextClient
import json
from tqdm import tqdm
import base64
import replicate

# generation_configs = {'do_sample': False, 'max_new_tokens': 128}

# model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
# # model_id = "Qwen/Qwen-VL-Chat"
# # model_id = "Salesforce/instructblip-vicuna-7b"


# def replicate_api(image_path, prompt):
#     with open(img_dir, 'rb') as file:
#         data = base64.b64encode(file.read()).decode('utf-8')
#         image = f"data:application/octet-stream;base64,{data}"

#     input = {
#         "image": image,
#         "prompt": prompt
#     }

#     # model_id = "yorickvp/llava-v1.6-mistral-7b:19be067b589d0c46689ffa7cc3ff321447a441986a7694c01225973c2eafc874"
#     model_id = "lucataco/qwen-vl-chat:50881b153b4d5f72b3db697e2bbad23bb1277ab741c5b52d80cd6ee17ea660e9"

#     output = replicate.run(
#         model_id,
#         input=input
#     )
#     response = "".join(output)

#     return response

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


    # print(f"Experiment done: {overall_counter}")