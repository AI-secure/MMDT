model_id="llava-hf/llava-v1.6-mistral-7b-hf"
scenario="natural"
task="action"

cd mmdt/perspectives
python hallucination/generate_image_to_text.py --model_id ${model_id} --scenario ${scenario} --task ${task}
python hallucination/eval_image_to_text.py --model_id ${model_id} --scenario ${scenario} --task ${task}
