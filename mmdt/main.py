import argparse
from importlib import import_module
from mmdt.summarize import summarize_results


PERSPECTIVES = ["safety", "hallucination", "fairness", "privacy", "adv", "ood"]
KNOWN_MODEL_MODALITY = {
    "text_to_image": ["dall-e-2", "dall-e-3", "DeepFloyd/IF-I-M-v1.0", "dreamlike-art/dreamlike-photoreal-2.0",
                      "kandinsky-community/kandinsky-3", "dataautogpt3/OpenDalleV1.1", "prompthero/openjourney-v4",
                      "stabilityai/stable-diffusion-2", "stabilityai/stable-diffusion-xl-base-1.0"],
    "image_to_text": ["models/gemini-1.5-pro-001", "gpt-4-vision-preview", "gpt-4o-2024-05-13",
                      "claude-3-opus-20240229", "llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/llava-v1.6-vicuna-7b-hf",
                      "llava-hf/llava-v1.6-vicuna-13b-hf", "Salesforce/instructblip-vicuna-7b", "Qwen/Qwen-VL-Chat"]
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default="", choices=["", "text_to_image", "image_to_text"])
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for generation')
    parser.add_argument('--perspectives', type=str, default="safety,hallucination,fairness,privacy,adv,ood",
                        help='Perspectives to evaluate')
    parser.add_argument('--scenario', type=str, default="", help='Scenario type')
    parser.add_argument('--task', type=str, default="", help='Task to be executed')
    parser.add_argument('--output_dir', type=str, default="./results", help='Output directory')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Check modality
    if args.modality == "":
        if args.model_id in KNOWN_MODEL_MODALITY["text_to_image"]:
            args.modality = "text_to_image"
        elif args.model_id in KNOWN_MODEL_MODALITY["image_to_text"]:
            args.modality = "image_to_text"
        else:
            raise Exception("Unknown modality")

    # Check perspective
    perspectives = args.perspectives.split(',')
    perspectives = [perspective.strip() for perspective in perspectives]
    for perspective in perspectives:
        assert perspective in PERSPECTIVES, f"Perspective {perspective} not recognized"

    for perspective in perspectives:
        generation_module_name = f'perspectives.{perspective}.generate_{args.modality}'
        perspective_generation_module = import_module(generation_module_name)
        perspective_generation_module.generate(args)

        eval_module_name = f'perspectives.{perspective}.eval_{args.modality}'
        perspective_eval_module = import_module(eval_module_name)
        perspective_eval_module.evaluate(args)

    summarize_results(args.output_dir)


if __name__ == '__main__':
    main()
