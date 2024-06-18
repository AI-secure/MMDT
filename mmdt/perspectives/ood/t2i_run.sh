#!/bin/bash

t2i_models=("dall-e-2" "dall-e-3" "DeepFloyd/IF-I-M-v1.0" "dreamlike-art/dreamlike-photoreal-2.0" "prompthero/openjourney-v4" "stabilityai/stable-diffusion-xl-base-1.0")
tasks=("helpfulness" "count" "spatial" "color" "size")
corruptions=("Shake_" "Paraphrase_")


for model in "${t2i_models[@]}"; do
    for task in "${tasks[@]}"; do
        for corruption in "${corruptions[@]}"; do
            python generate_text_to_image.py --model "$model" --task "$task" --scenario "$corruption"
            python eval_text_to_image.py --model "$model" --task "$task" --scenario "$corruption"

        done
    done
done
