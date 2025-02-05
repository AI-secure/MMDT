# git checkout -b o1-privacy
python generate_image_to_text.py --model_id o1-redteam-sys --scenario location --task Pri-SV-with-text
python eval_image_to_text.py --model_id o1-redteam-sys --scenario location


python generate_image_to_text.py --model_id o1-redteam-sys --scenario pii --task age
python generate_image_to_text.py --model_id o1-redteam-sys --scenario pii --task gender
python generate_image_to_text.py --model_id o1-redteam-sys --scenario pii --task ethnicity

python eval_image_to_text.py --model_id o1-redteam-sys --scenario pii



python generate_image_to_text.py --model_id o1-redteam-sys --scenario vispr --task naive
python generate_image_to_text.py --model_id o1-redteam-sys --scenario vispr --task naive_w_info_type
python generate_image_to_text.py --model_id o1-redteam-sys --scenario vispr --task generated
python generate_image_to_text.py --model_id o1-redteam-sys --scenario vispr --task story
python generate_image_to_text.py --model_id o1-redteam-sys --scenario vispr --task naive_enhance


