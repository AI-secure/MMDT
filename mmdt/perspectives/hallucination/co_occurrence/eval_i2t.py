import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_json", "-i", type=str, default="co-occurrence/dataset/cooc/high_obj_cooc.json")
args = parser.parse_args()


with open(args.input_json) as f:
    data = json.load(f)


type_folder = list(data.keys())

word_to_number = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", "20": "twenty", "21": "twenty one", "22": "twenty two", "23": "twenty three", "24": "twenty four", "25": "twenty five", "26": "twenty six", "27": "twenty seven", "28": "twenty eight", "29": "twenty nine", "30": "thirty", "31": "thirty one", "32": "thirty two", "33": "thirty three", "34": "thirty four", "35": "thirty five", "36": "thirty six", "37": "thirty seven", "38": "thirty eight", "39": "thirty nine", "40": "forty", "41": "forty one", "42": "forty two", "43": "forty three", "44": "forty four", "45": "forty five", "46": "forty six", "47": "forty seven", "48": "forty eight", "49": "forty nine", "50": "fifty", "51": "fifty one", "52": "fifty two", "53": "fifty three", "54": "fifty four", "55": "fifty five", "56": "fifty six", "57": "fifty seven", "58": "fifty eight", "59": "fifty nine", "60": "sixty", "61": "sixty one", "62": "sixty two", "63": "sixty three", "64": "sixty four", "65": "sixty five", "66": "sixty six", "67": "sixty seven", "68": "sixty eight", "69": "sixty nine", "70": "seventy", "71": "seventy one", "72": "seventy two", "73": "seventy three", "74": "seventy four", "75": "seventy five"}

all_category_list = ["object_existence", "count", "attribute", "spatial", "action"]

object_acc = 0
count_acc = 0
attribute_acc = 0
spatial_acc = 0
action_acc = 0

acc_all = 0
counter_all = 0
object_all = 0
count_all = 0
attribute_all = 0
spatial_all = 0
action_all = 0

for folder in type_folder:
    folder_dict = data[folder]

    acc_type = 0
    counter_type = 0
    for category in all_category_list:

        category_list = folder_dict[category]

        acc = 0
        for item in category_list:
            
            target = item["target"]
            label = item["label"]
            caption = item["response"]

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
            elif category == "action":
                action_all += 1

            if label == "positive":
                if str(target) in list(word_to_number.keys()):
                    target = word_to_number[str(target)]
                # print(target)
                if isinstance(target, list):
                    obj1 = target[0]
                    obj2 = target[1]
                    if obj1 in caption and obj2 in caption:
                        acc += 1
                        acc_type += 1
                        acc_all += 1

                        if category == "object_existence":
                            object_acc += 1
                        elif category == "count":
                            count_acc += 1
                        elif category == "attribute":
                            attribute_acc += 1
                        elif category == "spatial":
                            spatial_acc += 1
                        elif category == "action":
                            action_acc += 1
                else:
                    if target in caption:
                        acc += 1
                        acc_type += 1
                        acc_all += 1

                        if category == "object_existence":
                            object_acc += 1
                        elif category == "count":
                            count_acc += 1
                        elif category == "attribute":
                            attribute_acc += 1
                        elif category == "spatial":
                            spatial_acc += 1
                        elif category == "action":
                            action_acc += 1

            elif label == "negative":
                if target in word_to_number:
                    target = word_to_number[target]

                if isinstance(target, list):
                    obj1 = target[0]
                    obj2 = target[1]
                    if obj1 not in caption and obj2 not in caption:
                        acc += 1
                        acc_type += 1
                        acc_all += 1

                        if category == "object_existence":
                            object_acc += 1
                        elif category == "count":
                            count_acc += 1
                        elif category == "attribute":
                            attribute_acc += 1
                        elif category == "spatial":
                            spatial_acc += 1
                        elif category == "action":
                            action_acc += 1
                else:
                    if target not in caption:
                        acc += 1
                        acc_type += 1
                        acc_all += 1
                
                        if category == "object_existence":
                            object_acc += 1
                        elif category == "count":
                            count_acc += 1
                        elif category == "attribute":
                            attribute_acc += 1
                        elif category == "spatial":
                            spatial_acc += 1
                        elif category == "action":
                            action_acc += 1

            else:
                print("Error: label not found")
                exit()
            
        print(f"Category: {category} Accuracy: {acc/len(category_list)}")
    
    print(f"Type: {folder} Accuracy: {acc_type/counter_type}")
    print("\n")

print(f"Object Existence Accuracy: {object_acc/object_all}")
print(f"Count Accuracy: {count_acc/count_all}")
print(f"Attribute Accuracy: {attribute_acc/attribute_all}")
print(f"Spatial Accuracy: {spatial_acc/spatial_all}")
print(f"Action Accuracy: {action_acc/action_all}")

print(f"Overall Accuracy: {acc_all/counter_all}")

