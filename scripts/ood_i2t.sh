#!/bin/bash

modality="image_to_text"
model_id=$1  # e.g., llava-hf/llava-v1.6-vicuna-7b-hf
perspective="ood"
tasks=("attribute" "count" "spatial" "identification")
corruptions=("Van_Gogh" "oil_painting" "watercolour_painting" "gaussian_noise" "zoom_blur" "pixelate")

for task in "${tasks[@]}"; do
    for corruption in "${corruptions[@]}"; do
        python mmdt/main.py --modality ${modality} --model_id ${model_id} --perspectives ${perspective} --task ${task} --scenario ${corruption}
    done
done
