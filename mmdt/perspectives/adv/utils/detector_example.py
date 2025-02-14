import numpy as np
import json
from image_detector_utils import ImageDetector


def single_detect_demo(prompt, entity, image):

    GroundingDINO = ImageDetector(debugger=True)

    result = GroundingDINO.single_detect(image, entity, box_threshold=0.4)
    print(result)


def batch_detect_demo(prompt, sample_dict):

    GroundingDINO = ImageDetector(debugger=True)
    result = GroundingDINO.batch_detect(sample_dict)
    print(result)

def batch_detect_attribute_demo(prompt, sample_dict):

    GroundingDINO = ImageDetector(debugger=True)
    result = GroundingDINO.detect_attribute(sample_dict)
    print(result)


# compare the performance of our detect_attribute api against the original batch detect api

image_dir_1 = "co-occurrence/dataset/test/material.jpg"  # glass/wooden vas
image_dir_2 = "co-occurrence/dataset/test/blue_bus.jpg"  # blue/red bus

entity_list_1 = [
    {"vase": ["glass"]},
    {"vase": ["wooden"]}
]

entity_list_2 = [
    {"bus": ["blue"]},
    {"bus": ["red"]}
]

sample_dict = {
    "image_0": {
        "img_path": image_dir_1,
        "named_entity": ["glass vase", "wooden vase"],
        "box_threshold": 0.5,
    },
    "image_1": {
        "img_path": image_dir_2,
        "named_entity": ["blue bus", "red bus"],
        "box_threshold": 0.5,
    },
}

batch_detect_demo("demo", sample_dict)  

input("original batch detect done!")

sample_dict_attribute = {
    "image_0": {
        "img_path": image_dir_1,
        "named_entity": entity_list_1,
        "box_threshold": 0.5,
        "attribute_threshold": 0.5,
        "iou_threshold": 0.2,
    },
    "image_1": {
        "img_path": image_dir_2,
        "named_entity": entity_list_2,
        "box_threshold": 0.5,
        "attribute_threshold": 0.5,
        "iou_threshold": 0.2,
    },
}


batch_detect_attribute_demo("demo", sample_dict_attribute)


with open("attribute_detect_demo.json", "w") as f:
    json.dump(sample_dict_attribute, f, indent=4)
