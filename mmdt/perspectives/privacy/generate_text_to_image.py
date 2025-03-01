
import os
import sys
from tqdm import tqdm
import argparse
from datasets import load_dataset
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("./")

from mmdt.models import Text2ImageClient


def generate(args):
    # Adjusted path to read data from the correct directory
    model_id, task,   dry_run = args.model_id, args.task, args.dry_run
    seed=0
    
    ds = load_dataset("AI-Secure/MMDecodingTrust-T2I", "privacy")
    if task=="laion_1k":
        data = ds[task]
        # Open the file and read line by line
        img_ids = data['idx']
        prompts = data['prompt']
    else:
        print("not supported task for privacy: ", task)
        return 

    # Define the output directory based on model name, scenario, and task
    model_name = model_id.split("/")[1] if '/' in model_id else model_id
    output_dir = os.path.join('results/text_to_image/privacy', task, model_name, f"seed{seed}")
    os.makedirs(output_dir, exist_ok=True)

    client = Text2ImageClient(model_id=model_id)

    # Iterate over each prompt
    for img_id, prompt in tqdm(zip(img_ids, prompts), desc=f"Generating images for {model_name}"): 
      
        prompt_for_fname = prompt.replace(' ','_').replace('/','_')[:50]
        fname= "{}_{}.png".format(img_id,prompt_for_fname)
        image_path=os.path.join(output_dir, fname)
        if not os.path.exists(image_path):
            if dry_run:
                print(f"Dry run enabled. Generating an empty image at {image_path}.")
                # Create an empty white image of size 512x512
                image = Image.new("RGB", (512, 512), "white")
                image.save(image_path)
            else:
                image = client.generate(prompt, seed=seed, save_path=image_path)
           
        else:
            print(f"Image {fname} already exists. Skipping generation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for image generation')
    parser.add_argument('--task', type=str, default="laion_1k", help='Specify the task to execute')

    parser.add_argument('--dry_run', action='store_true', help='Enable dry run mode to generate empty image')
    args = parser.parse_args()

    generate(args)
