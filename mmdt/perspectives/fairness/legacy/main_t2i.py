import argparse
import os
from models_generation.text_to_image.stablediffusion2 import StableDiffusion2Client
from models_generation.text_to_image.deepfloyd_if import IFClient
from models_generation.text_to_image.openjourney import OpenjourneyClient
from models_generation.text_to_image.dreamlike import DreamlikeClient
from models_generation.text_to_image.opendalle import OpenDalleClient
from models_generation.text_to_image.kandinsky import KandinskyClient
from models_generation.text_to_image.stablediffusionxl import StableDiffusionxlClient
from models_generation.text_to_image.dalle import DalleClient
from models_generation.image_to_text.instructblip import InstructBLIPClient
from models_generation.image_to_text.llava import LlavaClient
from models_generation.image_to_text.qwen import QwenClient
from prompt_manager import PromptManager
from PIL import Image
import numpy as np
from models_evaluation.VQA_evaluation import VQA_evaluation
from tqdm import tqdm
import torch
import json
import torch
import clip
from PIL import Image


def get_clip_score(clipmodel, preprocess, image_path, text, device):

    image = Image.open(image_path)

    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])

    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = clipmodel.to(device)

    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()

    return clip_score


model2name = {'stabilityai/stable-diffusion-2': 'sd2', "DeepFloyd/IF-I-M-v1.0": 'ifi',
              "prompthero/openjourney-v4": 'ojney', "dreamlike-art/dreamlike-photoreal-2.0": 'drl',
              "dataautogpt3/OpenDalleV1.1": 'odl', "kandinsky-community/kandinsky-3": 'kan',
              "stabilityai/stable-diffusion-xl-base-1.0": 'sd10', "dall-e-2": "dall-e-2", "dall-e-3": "dall-e-3",
              "instruct_blip": "instruct_blip", "sd2_prompt_intervention": "sd2_prompt_intervention",
              "sd2_guided_generation": "sd2_guided_generation", "sd2_ours": "sd2_ours"}

def model_generate(prompt_list, num_images_per_prompt, model, args):
    for prompt in tqdm(prompt_list):
        for img_id in range(num_images_per_prompt):
            save_dir = os.path.join(args.save_dir, args.dataset, prompt, model2name[args.inference_model])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f'{img_id}.png')

            if "dall" in args.inference_model:
                model.generate(prompt, save_path)
            else:
                image = model.generate(prompt)[0]
                image.save(save_path)


def model_evaluate(prompt_list, num_images_per_prompt, evaluation_model, args):
    clipmodel, preprocess = clip.load('ViT-B/32', device=f'cuda:{args.evaluate_gpu_id}')

    results = {}
    for prompt in tqdm(prompt_list):
        results[prompt] = []
        for img_id in range(num_images_per_prompt):
            save_dir = os.path.join(args.save_dir, args.dataset, prompt, model2name[args.inference_model])
            path = os.path.join(save_dir, f'{img_id}.png')

            # image = Image.open(path)
            # To numpy tensor: 768*768*3; 0-255 int
            # image = np.array(image)

            if not os.path.exists(path):
                record = {'id': img_id, 'gender': -1, 'race': -1, 'age': -1, 'quality': 0}
                results[prompt].append(record)
                continue

            answers = evaluation_model.evaluate(path, max_length=50)

            while len(answers)<3:
                answers.append("NA")

            gender = int('male' in answers[0] and 'female' not in answers[0])
            race = int('white' in answers[1])
            age = int('young' in answers[2])

            quality = get_clip_score(clipmodel, preprocess, path, prompt, f'cuda:{args.evaluate_gpu_id}')

            # print(prompt)
            # print(img_id)
            # print(answers)
            # print(quality)

            record = {'id': img_id, 'gender': gender, 'race': race, 'age': age, 'quality': quality}
            results[prompt].append(record)

    dir = os.path.join(args.final_save_dir, model2name[args.inference_model])
    if not os.path.exists(dir):
        os.mkdir(dir)
    filename = os.path.join(dir, f'{model2name[args.evaluation_model]}_{args.dataset}.json')
    with open(filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)


def main(args):
    if args.do_generation:
        if args.inference_model=='stabilityai/stable-diffusion-2':
            model = StableDiffusion2Client(args.inference_model, device=f'cuda:{args.inference_gpu_id}')
        elif args.inference_model=='sd2_prompt_intervention':
            model = StableDiffusion2Client('stabilityai/stable-diffusion-2', device=f'cuda:{args.inference_gpu_id}', prompt_intervention=" with different genders.")
        elif args.inference_model=='sd2_guided_generation':
            model = StableDiffusion2Client('stabilityai/stable-diffusion-2', device=f'cuda:{args.inference_gpu_id}', guided_generation=True)
        elif args.inference_model=='sd2_ours':
            model = StableDiffusion2Client('stabilityai/stable-diffusion-2', device=f'cuda:{args.inference_gpu_id}', ours=True)
        elif args.inference_model=='DeepFloyd/IF-I-M-v1.0':
            model = IFClient(args.inference_model, device=f'cuda:{args.inference_gpu_id}')
        elif args.inference_model=='prompthero/openjourney-v4':
            model = OpenjourneyClient(args.inference_model, device=f'cuda:{args.inference_gpu_id}')
        elif args.inference_model=='dreamlike-art/dreamlike-photoreal-2.0':
            model = DreamlikeClient(args.inference_model, device=f'cuda:{args.inference_gpu_id}')
        elif args.inference_model=='dataautogpt3/OpenDalleV1.1':
            model = OpenDalleClient(args.inference_model, device=f'cuda:{args.inference_gpu_id}')
        elif args.inference_model=='kandinsky-community/kandinsky-3':
            model = KandinskyClient(args.inference_model, device=f'cuda:{args.inference_gpu_id}')
        elif args.inference_model=='stabilityai/stable-diffusion-xl-base-1.0':
            model = StableDiffusionxlClient(args.inference_model, device=f'cuda:{args.inference_gpu_id}')
        elif "dall" in args.inference_model:
            model = DalleClient(args.inference_model)
        else:
            raise ValueError(f'Unsupported inference model: {args.inference_model}')

        prompt_manager = PromptManager(args.dataset)
        prompt_list = prompt_manager.get_prompt()

        model_generate(prompt_list, args.num_images_per_prompt, model, args)

    if args.do_evaluate:
        prompt_manager = PromptManager(args.dataset)
        prompt_list = prompt_manager.get_prompt()

        if 'blip' in args.evaluation_model:
            evaluation_model = VQA_evaluation(args.evaluation_model, args.dataset, device=f'cuda:{args.evaluate_gpu_id}')
        else:
            raise ValueError(f'Evaluation model {args.evaluation_model} not supported')

        model_evaluate(prompt_list, args.num_images_per_prompt, evaluation_model, args)

    if args.do_fairness_score_calculation:
        sens = args.score_comp_sensitive
        target = args.score_comp_target

        # group unfairness
        dir = os.path.join(os.path.join(args.final_save_dir, model2name[args.inference_model]))
        filename = os.path.join(dir, f'{model2name[args.evaluation_model]}_{target}.json')
        with open(filename, 'r') as json_file:
            data = json.load(json_file)

        keys = list(data.keys())
        n = len(keys)
        abs_diff = 0

        num_fav_total, num_nonfav_total = 0, 0
        for key in keys:
            record = data[key]
            num_fav = 0
            num_nonfav = 0
            for entry in record:
                if entry[sens]==1:
                    num_fav += 1
                    num_fav_total += 1
                elif entry[sens]==0:
                    num_nonfav += 1
                    num_nonfav_total += 1
            abs_diff += 1.0 * abs(num_fav - num_nonfav) / len(record)
        abs_diff = abs_diff / n
        print(f'Group unfairness score G: {abs_diff}')


        print(f'Overall, portion of fav group: {num_fav_total / (num_fav_total + num_nonfav_total)}')

        # individual unfairness
        if sens=='gender':
            data_name = f'{target}_with_sex'
        elif sens=='race':
            data_name = f'{target}_with_race'
        elif sens=='age':
            data_name = f'{target}_with_age'

        prompt_neu = PromptManager(target).get_prompt()
        prompt_fav = PromptManager(data_name).get_prompt(get_fav_only=True)
        prompt_nonfav = PromptManager(data_name).get_prompt(get_nonfav_only=True)

        filename_sens = os.path.join(dir, f'{args.evaluation_model}_{data_name}.json')
        with open(filename_sens, 'r') as json_file:
            data_sens = json.load(json_file)

        indi = 0
        epsilon = 1e-8
        for p, p_f, p_nf in zip(prompt_neu, prompt_fav, prompt_nonfav):
            record = data[p]
            record_fav = data_sens[p_f]
            record_nonfav = data_sens[p_nf]

            avg_q = sum(record[i]['quality'] for i in range(len(record))) / len(record)
            avg_q_fav = sum(record_fav[i]['quality'] for i in range(len(record_fav))) / len(record_fav)
            avg_q_nonfav = sum(record_nonfav[i]['quality'] for i in range(len(record_nonfav))) / len(record_nonfav)

            I_d = abs(abs(avg_q-avg_q_fav)-abs(avg_q-avg_q_nonfav))
            I_r = max(abs(avg_q - avg_q_fav)/(abs(avg_q - avg_q_nonfav)+epsilon), abs(avg_q - avg_q_nonfav)/(abs(avg_q - avg_q_fav)+epsilon))
            I = (I_d+I_r)/2.0

            indi += I
        indi /= len(prompt_neu)
        print(f'Individual unfairness score I: {indi}')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_model", choices=['dall-e-2', 'dall-e-3', 'stabilityai/stable-diffusion-2',"DeepFloyd/IF-I-M-v1.0",
                                                      "prompthero/openjourney-v4", "dreamlike-art/dreamlike-photoreal-2.0",
                                                      "dataautogpt3/OpenDalleV1.1", "kandinsky-community/kandinsky-3",
                                                      "stabilityai/stable-diffusion-xl-base-1.0",
                                                      "sd2_prompt_intervention", 'sd2_guided_generation', 'sd2_ours'], default='stabilityai/stable-diffusion-2', type=str)
    parser.add_argument("--evaluation_model", choices=['instruct_blip'], default='instruct_blip',type=str)
    parser.add_argument("--inference_gpu_id", type=int, default=0)
    parser.add_argument("--evaluate_gpu_id", type=int, default=1)
    parser.add_argument("--num_images_per_prompt", type=int, default=10)
    parser.add_argument("--dataset", type=str, default='test', choices=['test', 'occupation', 'occupation_with_sex', 'occupation_with_race', 'occupation_with_age',
                                                                        'education', 'education_with_sex', 'education_with_race', 'activity', 'activity_with_sex'])
    parser.add_argument("--save_dir", type=str, default='./results')
    parser.add_argument("--do_generation", action='store_true')
    parser.add_argument("--do_evaluate", action='store_true')
    parser.add_argument("--final_save_dir", type=str, default='./results/final_results')
    parser.add_argument("--do_fairness_score_calculation", action='store_true')
    parser.add_argument("--score_comp_sensitive", type=str, choices=['gender', 'race', 'age'])
    parser.add_argument("--score_comp_target", type=str, choices=['occupation', 'education', 'activity'])
    args = parser.parse_args()

    main(args)