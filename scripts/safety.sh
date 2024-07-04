# modality="image_to_text"
# model_id="llava-hf/llava-v1.6-mistral-7b-hf"
# scenario="typography"

# python mmdt/main.py --modality ${modality} --model_id ${model_id} --scenario ${scenario}


modality="text_to_image"
model_id="dall-e-2"
scenario="vanilla"

python mmdt/main.py --modality ${modality} --model_id ${model_id} --scenario ${scenario}