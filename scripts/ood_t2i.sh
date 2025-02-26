#!/bin/bash

modality="text_to_image"
model_id=$1  # e.g., stabilityai/stable-diffusion-2
perspective="adv"
tasks=("helpfulness" "count" "spatial" "color" "size")
corruptions=("Shake_" "Paraphrase_")


for task in "${tasks[@]}"; do
    for corruption in "${corruptions[@]}"; do
        python mmdt/main.py --modality ${modality} --model_id ${model_id} --perspectives ${perspective} --task ${task} --scenario ${corruption}
    done
done
