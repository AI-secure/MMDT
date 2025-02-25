modality="text_to_image"
model_id=$1  # e.g., stabilityai/stable-diffusion-2
perspective="adv"

python mmdt/main.py --modality ${modality} --model_id ${model_id} --perspectives ${perspective}