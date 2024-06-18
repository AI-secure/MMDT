#!/bin/bash

i2t_models=("gpt-4-vision-preview" "gpt-4o-2024-05-13" "llava-hf/llava-v1.6-vicuna-7b-hf")
tasks=("attribute" "count" "spatial" "identification")
corruptions=("Van_Gogh" "oil_painting" "watercolour_painting" "gaussian_noise" "zoom_blur" "pixelate")

for model in "${i2t_models[@]}"; do
    for task in "${tasks[@]}"; do
        for corruption in "${corruptions[@]}"; do
            python generate_image_to_text.py --model "$model" --task "$task" --scenario "$corruption"
            python eval_image_to_text.py --model "$model" --task "$task" --scenario "$corruption"

        done
    done
done
