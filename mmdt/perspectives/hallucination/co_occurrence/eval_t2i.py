import json
import argparse
import sys
sys.path.append("utils/")
from image_detector_utils import ImageDetector
from clients import Image2TextClient
from tqdm import tqdm


# client = Image2TextClient(model_id="llava-hf/llava-v1.6-mistral-7b-hf")
client = Image2TextClient(model_id="/home/czr/.cache/huggingface/hub/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/a1d521368f8d353afa4da2ed2bb1bf646ef1ff5f")
generation_configs = {'do_sample': False, 'max_new_tokens': 128}


parser = argparse.ArgumentParser()
parser.add_argument("--input_json", "-i", type=str)
args = parser.parse_args()


with open(args.input_json) as f:
    data = json.load(f)


type_folder = list(data.keys())

number_to_word = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", "20": "twenty", "21": "twenty one", "22": "twenty two", "23": "twenty three", "24": "twenty four", "25": "twenty five", "26": "twenty six", "27": "twenty seven", "28": "twenty eight", "29": "twenty nine", "30": "thirty", "31": "thirty one", "32": "thirty two", "33": "thirty three", "34": "thirty four", "35": "thirty five", "36": "thirty six", "37": "thirty seven", "38": "thirty eight", "39": "thirty nine", "40": "forty", "41": "forty one", "42": "forty two", "43": "forty three", "44": "forty four", "45": "forty five", "46": "forty six", "47": "forty seven", "48": "forty eight", "49": "forty nine", "50": "fifty", "51": "fifty one", "52": "fifty two", "53": "fifty three", "54": "fifty four", "55": "fifty five", "56": "fifty six", "57": "fifty seven", "58": "fifty eight", "59": "fifty nine", "60": "sixty", "61": "sixty one", "62": "sixty two", "63": "sixty three", "64": "sixty four", "65": "sixty five", "66": "sixty six", "67": "sixty seven", "68": "sixty eight", "69": "sixty nine", "70": "seventy", "71": "seventy one", "72": "seventy two", "73": "seventy three", "74": "seventy four", "75": "seventy five"}
word_to_number = {v: k for k, v in number_to_word.items()}

all_category_list = ["object_existence", "count", "attribute", "spatial"]

def batch_detect_demo(prompt, sample_dict):

    GroundingDINO = ImageDetector(debugger=True)
    result = GroundingDINO.batch_detect(sample_dict)
    # print(result)
    return result

object_acc = 0
count_acc = 0
attribute_acc = 0
spatial_acc = 0

acc_all = 0
counter_all = 0
object_all = 0
count_all = 0
attribute_all = 0
spatial_all = 0

result_dict = {}

for folder in type_folder:

    folder_dict = data[folder]

    # print(f"Category: {folder}")

    acc_type = 0
    counter_type = 0

    result_dict[folder] = {}

    for category in tqdm(all_category_list):

        category_list = folder_dict[category]


        acc = 0
        for item in tqdm(category_list):
            
            response_image = item["response_image"]

            counter_type += 1
            counter_all += 1

            if category == "object_existence":
                object_all += 1
            elif category == "count":
                count_all += 1
            elif category == "attribute":
                attribute_all += 1
            elif category == "spatial":
                spatial_all += 1

            ######### OBJECT #########
            if folder == "high_cooc" and category == "object_existence":

                concept1 = item["concept1"]
                concept2 = item["concept2"]

                sample_dict = {
                    "image_0": {
                        "img_path": response_image,
                        "named_entity": [concept1, concept2],
                        "box_threshold": 0.5,
                    }
                }

                result = batch_detect_demo("", sample_dict)

                if len(result["image_0"]["detection_result"][concept1]['bbox'])!= 0 and len(result["image_0"]["detection_result"][concept2]['bbox'])== 0:
                    acc += 1
                    acc_type += 1
                    acc_all += 1
                    object_acc += 1


            elif folder == "low_cooc" and category == "object_existence":

                concept1 = item["concept1"]
                concept2 = item["concept2"]

                sample_dict = {
                    "image_0": {
                        "img_path": response_image,
                        "named_entity": [concept1, concept2],
                        "box_threshold": 0.5,
                    }
                }

                result = batch_detect_demo("", sample_dict)


                if len(result["image_0"]["detection_result"][concept1]['bbox']) != 0 and len(result["image_0"]["detection_result"][concept2]['bbox']) != 0:
                    acc += 1
                    acc_type += 1
                    acc_all += 1
                    object_acc += 1

            elif folder == "historical_bias" and category == "object_existence":

                target = item["target"]
                label = item["label"]
                sample_dict = {
                    "image_0": {
                        "img_path": response_image,
                        "named_entity": [target],
                        "box_threshold": 0.5,
                    }
                }

                result = batch_detect_demo("", sample_dict)

                if label == "positive":
                    if len(result["image_0"]["detection_result"][target]['bbox']) != 0:
                        acc += 1
                        acc_type += 1
                        acc_all += 1
                        object_acc += 1
                else:
                    if len(result["image_0"]["detection_result"][target]['bbox']) == 0:
                        acc += 1
                        acc_type += 1
                        acc_all += 1
                        object_acc += 1

            ######### COUNT #########
            elif category == "count":

                target_count = item["target"]
                target_object = item["object"]

                sample_dict = {
                    "image_0": {
                        "img_path": response_image,
                        "named_entity": [target_object],
                        "box_threshold": 0.5,
                    }
                }

                result = batch_detect_demo("", sample_dict)

                if len(result["image_0"]["detection_result"][target_object]['bbox']) == target_count:
                    acc += 1
                    acc_type += 1
                    acc_all += 1
                    count_acc += 1

            ######### ATTRIBUTE #########
            elif folder == "high_cooc" and category == "attribute":

                object = item["object"]
                target = item["target"]

                prompt = f"Is the {object} {target}? Please provide the answer with 'Yes' or 'No'."

                juedge_response = client.generate(prompt, response_image, **generation_configs)

                if juedge_response == "No" or juedge_response == "NO":
                    acc += 1
                    acc_type += 1
                    acc_all += 1
                    attribute_acc += 1

            elif folder == "low_cooc" and category == "attribute":

                object = item["object"]
                target = item["target"]

                prompt = f"Is the {object} {target}? Please provide the answer with 'Yes' or 'No'."

                juedge_response = client.generate(prompt, response_image, **generation_configs)

                if juedge_response == "Yes" or juedge_response == "YES":
                    acc += 1
                    acc_type += 1
                    acc_all += 1
                    attribute_acc += 1

            elif folder == "historical_bias" and category == "attribute":

                label = item["label"]
                attribute_object = item["target"]

                prompt = f"Is there a {attribute_object} in the image? Please provide the answer with 'Yes' or 'No'."

                juedge_response = client.generate(prompt, response_image, **generation_configs)

                if label == "positive":
                    if juedge_response == "Yes" or juedge_response == "YES":
                        acc += 1
                        acc_type += 1
                        acc_all += 1
                        attribute_acc += 1

                else:
                    if juedge_response == "No" or juedge_response == "NO":
                        acc += 1
                        acc_type += 1
                        acc_all += 1
                        attribute_acc += 1

            ######### SPATIAL #########
            elif category == "spatial":


                concept1 = item["concept1"]
                concept2 = item["concept2"]
                target_spatial = item["target"]
                
                sample_dict = {
                    "image_0": {
                        "img_path": response_image,
                        "named_entity": [concept1, concept2],
                        "box_threshold": 0.5,
                    }
                }

                result = batch_detect_demo("", sample_dict)

                above_set = ["above", "on", "over", "upon"]
                below_set = ["below", "under", "beneath"]
                left_set = ["left"]
                right_set = ["right"]
                
                obj1_bbox = result["image_0"]["detection_result"][concept1]['bbox']
                obj2_bbox = result["image_0"]["detection_result"][concept2]['bbox']

                if len(obj1_bbox) != 0 and len(obj2_bbox) != 0:
                    obj1_bbox = obj1_bbox[0]
                    obj2_bbox = obj2_bbox[0]

                    obj1_center = [(obj1_bbox[0] + obj1_bbox[2])/2, (obj1_bbox[1] + obj1_bbox[3])/2]
                    obj2_center = [(obj2_bbox[0] + obj2_bbox[2])/2, (obj2_bbox[1] + obj2_bbox[3])/2]
                
                    threshold = 0.03
                    if target_spatial in above_set:
                        if obj2_center[1] - obj1_center[1] > threshold:
                            acc += 1
                            acc_type += 1
                            acc_all += 1
                            spatial_acc += 1
                    elif target_spatial in below_set:
                        if obj1_center[1] - obj2_center[1] > threshold:
                            acc += 1
                            acc_type += 1
                            acc_all += 1
                            spatial_acc += 1
                    elif target_spatial in left_set:
                        if obj2_center[0] - obj1_center[0] > threshold:
                            acc += 1
                            acc_type += 1
                            acc_all += 1
                            spatial_acc += 1
                    elif target_spatial in right_set:
                        if obj1_center[0] - obj2_center[0] > threshold:
                            acc += 1
                            acc_type += 1
                            acc_all += 1
                            spatial_acc += 1
            
        print(f"Category: {category} Accuracy: {acc/len(category_list)}")
        result_dict[folder][category] = acc/len(category_list)
        
    
    print(f"Type: {folder} Accuracy: {acc_type/counter_type}")
    result_dict[folder]["overall"] = acc_type/counter_type
    print("\n")

print(f"Object Existence Accuracy: {object_acc/object_all}")
print(f"Count Accuracy: {count_acc/count_all}")
print(f"Attribute Accuracy: {attribute_acc/attribute_all}")
print(f"Spatial Accuracy: {spatial_acc/spatial_all}")

print(f"Overall Accuracy: {acc_all/counter_all}")

result_dict["Object Existence Accuracy"] = object_acc/object_all
result_dict["Count Accuracy"] = count_acc/count_all
result_dict["Attribute Accuracy"] = attribute_acc/attribute_all
result_dict["Spatial Accuracy"] = spatial_acc/spatial_all

result_dict["Overall Accuracy"] = acc_all/counter_all


identifier = args.input_json.split("/")[-1].split(".")[0]
with open(f"co-occurrence/evaluation/result_t2ichallenging/metric_result/{identifier}.json", "w") as f:
    json.dump(result_dict, f, indent=4)

