
import argparse
import os
import sys
path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
sys.path.append(path_to_add)
from location.generate_response_location_privacy import *
from pii.generate_response_pii import generate_pii_response
from PIL import Image
from datasets import load_dataset


def generate(kwargs):
    model_id, scenario, task = kwargs.model_id, kwargs.scenario, kwargs.task
    dataset_dir = './mmdt/perspectives/privacy/location/Pri-Street-View'
    if(scenario=="location"):
        if not os.path.exists(dataset_dir):
            # Define task mapping to folder names
            task_to_path = {
                "Pri-SV-with-text": "Dataset-1-single_image_text",
                "Pri-SV-without-text": "Dataset-2-single_image_no_text",
                "Pri-4Loc-SV-with-text": "Dataset-3-group_text_feature",
                "Pri-4Loc-SV-without-text": "Dataset-3-group_no_text_feature"
            }

            task_to_type = {
                "Pri-SV-with-text": "single_image_text",
                "Pri-SV-without-text": "single_image_no_text",
                "Pri-4Loc-SV-with-text": "group_image_text",
                "Pri-4Loc-SV-without-text": "group_image_no_text"
            }
            # Iterate through each task in the mapping
            for task, folder_name in task_to_path.items():
                print(f"Processing task: {task}")

                # Load the dataset for the current task
                ds = load_dataset("AI-Secure/MMDecodingTrust-I2T", "privacy", split="street_view")

                # Filter the dataset based on the task
                filtered_ds = ds.filter(lambda x: x["type_street_view"] == task_to_type[task])
                filtered_length = len(filtered_ds) 
                print(f"filtered_length = {filtered_length}")
                # Create the directory if it doesn't exist
                output_dir = os.path.join('./mmdt/perspectives/privacy/location/Pri-Street-View', folder_name)
                os.makedirs(output_dir, exist_ok=True)

                # Save images for the current task
                for row in filtered_ds:
                    image = row["image"]  # Assuming 'image' is a PIL.Image object
                    image_id = row["id"]  # Get image ID
                    image_path = os.path.join(output_dir, f"{image_id}.jpg")  # Save as ID.jpg
                    image.save(image_path)

                print(f"Saved {len(filtered_ds)} images to {output_dir}")
            
        if task in ["Pri-4Loc-SV-with-text", "Pri-4Loc-SV-without-text"]:
            generate_response_4img(model_id, task)
        elif task in ["Pri-SV-with-text","Pri-SV-without-text"]:
            generate_response_1img(model_id, task)
    elif (scenario == "pii"):
        ds = load_dataset("AI-Secure/MMDecodingTrust-I2T", "privacy", split="selfies")
        dataset_dir = './mmdt/perspectives/privacy/pii/Selfies'
        if not os.path.exists(dataset_dir):
            for row in ds:
                ethnicity = row.get("ethnicity", "").strip().replace(" ", "_")
                label_selfies = row.get("label_selfies", "").strip().replace(" ", "_")
                ethnicity_dir = os.path.join(dataset_dir, ethnicity)
                os.makedirs(ethnicity_dir, exist_ok=True)
                subdir_name = f"{ethnicity}_{label_selfies}"
                output_dir = os.path.join(ethnicity_dir, subdir_name)
                os.makedirs(output_dir, exist_ok=True)
                image_id = str(row.get("id", "")).strip()
                image_path = os.path.join(output_dir, f"{image_id}.jpg")
                image = row.get("image")
                image.save(image_path)
                print(f"Saved: {image_path}")

        generate_pii_response(model_id)
