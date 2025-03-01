modality="text_to_image"
model_id=$1  # e.g., stabilityai/stable-diffusion-2
perspective="fairness"
scenario="occupation" # 'occupation', 'occupation_with_sex', 'occupation_with_race', 'occupation_with_age', 'education', 'education_with_sex', 'education_with_race', 'activity', 'activity_with_sex'

python mmdt/main.py --modality ${modality} --model_id ${model_id} --perspectives ${perspective} --scenario $scenario