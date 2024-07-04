model_id="llava-hf/llava-v1.6-mistral-7b-hf"
scenario="location"
task="Pri-SV-with-text"

# cd mmdt/perspectives
# python privacy/generate_image_to_text.py --model_id ${model_id} --scenario ${scenario} --task ${task}
# python privacy/eval_image_to_text.py --model_id ${model_id} --scenario ${scenario} --task ${task}

model_id="gpt-4o-2024-05-13"
task="Pri-4Loc-SV-without-text"
cd mmdt/perspectives

python privacy/generate_image_to_text.py --model_id ${model_id} --scenario ${scenario} --task ${task}
python privacy/eval_image_to_text.py --model_id ${model_id} --scenario ${scenario} --task ${task}

