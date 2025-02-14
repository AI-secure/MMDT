# Adversarial Robustness Perspective

This directory contains scripts for generating and evaluating adversarial robustness perspectives in both text-to-image and image-to-text formats as part of the MMDT project.

There contains three sub tasks for adversarial robustness: 
- Object 
- Attribute
- Spatial

## Directory Structure

- `generate_t2i.py`: Script to generate images from text prompts.
- `generate_it2.py`: Script to generate text responses from images.
- `evaluate_t2i.py`: Script to evaluate the generated iamges.
- `evaluate_i2t.py`: Script to evaluate the generated text responses.
- `utils`: Folder containing the GroundingDINO detector tools.

## Usage
### Install dependencies
Besides installing the requirements for the whole MMDT project, you also need to install requirements for GroundingDINO detector tools.

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Download pre-trained Swin-T model weights for GroundingDINO.
```
cd utils/GroundingDINO
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### Generation
Make sure you are at the adv sub directory.

T2I Generation:
```
python generate_t2i.py --model_id {model_id} --sub_task {sub_task} --image_number {image_number}
```

I2T Generation:
```
python generate_i2t.py --model_id {model_id} --sub_task {sub_task}
```

T2I Evaluation:
```
python evaluate_t2i.py --model_id {model_id} --sub_task {sub_task} --image_number {image_number}
```

I2T Evaluation
```
python evaluate_i2t.py --model_id {model_id} --sub_task {sub_task}
```