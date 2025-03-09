modality="image_to_text"
model_id=$1  # e.g., llava-hf/llava-v1.6-vicuna-7b-hf
perspective="fairness"
scenario="individual" # "social_stereotype", "decision_making", "overkill", "individual"

python mmdt/main.py --modality ${modality} --model_id ${model_id} --perspectives ${perspective} --scenario $scenario