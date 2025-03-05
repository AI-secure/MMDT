#!/bin/bash

modality="image_to_text"
model_id="llava-hf/llava-v1.6-mistral-7b-hf"
perspective="safety"
scenarios=("typography" "illustration" "jailbreak")

for scenario in "${scenarios[@]}"; do
    python mmdt/main.py --modality ${modality} --model_id ${model_id} --perspectives ${perspective} --scenario ${scenario}
done
