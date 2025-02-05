# git checkout -b o1-privacy

source activate mmdt
export MM_DT_PATH="/path/to/your/mmdt"


## PII inference
### download data
cd $MM_DT_PATH/data/image-to-text/privacy
pip install gdown
gdown https://drive.google.com/uc?id=1e3WNC6eIy3fI7ev6jDIqNnl3aypyBOD0
unzip Selfies.zip

cd $MM_DT_PATH/perspectives/privacy/
### generation
python generate_image_to_text.py --model_id o1-redteam-sys --scenario pii --task age
python generate_image_to_text.py --model_id o1-redteam-sys --scenario pii --task gender
python generate_image_to_text.py --model_id o1-redteam-sys --scenario pii --task ethnicity
### eval score
python eval_image_to_text.py --model_id o1-redteam-sys --scenario pii
### eval + reformat response
cd $MM_DT_PATH/results/image-to-text/privacy/pii
python auto_grader.py



## Location inference (omitted for o1)
### download data
# cd $MM_DT_PATH/data/image-to-text/privacy
# gdown https://drive.google.com/uc?id=1e3WNC6eIy3fI7ev6jDIqNnl3aypyBOD0
# unzip MMDT-Privacy.zip
# mv MMDT-Privacy/Pri-Street-View/ .

# cd $MM_DT_PATH/perspectives/privacy/
### generation
# python generate_image_to_text.py --model_id o1-redteam-sys --scenario location --task Pri-SV-with-text
### eval score
# python eval_image_to_text.py --model_id o1-redteam-sys --scenario location


## PII OCR
### download data
cd $MM_DT_PATH/data/image-to-text/privacy
### TODO: need request access permission from Alex
gdown https://drive.google.com/drive/folders/1K6EEzmkMdCEal659eCEwdIXJ-CC779Pb -O vispr --folder

cd $MM_DT_PATH/perspectives/privacy/
python generate_image_to_text.py --model_id o1-redteam-sys --scenario vispr --task naive_enhance
python generate_image_to_text.py --model_id o1-redteam-sys --scenario vispr --task naive
python generate_image_to_text.py --model_id o1-redteam-sys --scenario vispr --task naive_w_info_type
python generate_image_to_text.py --model_id o1-redteam-sys --scenario vispr --task generated
python generate_image_to_text.py --model_id o1-redteam-sys --scenario vispr --task story

### eval + reformat response
cd $MM_DT_PATH/results/image-to-text/privacy/vispr
python auto_grader.py