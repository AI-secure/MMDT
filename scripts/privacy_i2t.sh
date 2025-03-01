
# python privacy/eval_image_to_text.py --model_id ${model_id} --scenario ${scenario}

# Usage: bash ./scripts/privacy_i2t.sh <model_id>
# e.g., ./scripts/privacy_i2t.sh gpt-4o-2024-05-13

modality="image_to_text"
model_id=$1  # e.g., llava-hf/llava-v1.6-vicuna-7b-hf, gpt-4o-2024-05-13
perspective="privacy"

# for location

scenario="location" # "location", "pii"
task="Pri-SV-with-text" # "Pri-SV-with-text", "Pri-SV-without-text", "Pri-4Loc-SV-with-text", "Pri-4Loc-SV-without-text"
python mmdt/main.py --modality ${modality} --model_id ${model_id} --perspectives ${perspective} --scenario $scenario --task ${task}

# for pii
# scenario="pii"
# python mmdt/main.py --modality ${modality} --model_id ${model_id} --perspectives ${perspective} --scenario $scenario