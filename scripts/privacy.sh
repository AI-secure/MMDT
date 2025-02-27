
cd mmdt/perspectives
scenario="location"
model_id="gpt-4o-2024-05-13"
task="Pri-SV-with-text"

# python privacy/generate_image_to_text.py --model_id ${model_id} --scenario ${scenario} --task ${task}
python privacy/eval_image_to_text.py --model_id ${model_id} --scenario ${scenario} --task ${task}

# scenario="pii"
# python privacy/generate_image_to_text.py --model_id ${model_id} --scenario ${scenario}
# python privacy/eval_image_to_text.py --model_id ${model_id} --scenario ${scenario}


