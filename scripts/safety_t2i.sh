#!/bin/bash

modality="text_to_image"
model_id="dall-e-2"
perspective="safety"
scenarios=("vanilla" "transformed" "jailbreak")

for scenario in "${scenarios[@]}"; do
    python mmdt/main.py --modality ${modality} --model_id ${model_id} --perspectives ${perspective} --scenario ${scenario}
done
