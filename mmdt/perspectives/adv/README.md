# Adversarial Robustness Perspective

This directory contains scripts for generating and evaluating adversarial robustness perspectives in both text-to-image and image-to-text formats as part of the MMDT project.

There contains three sub tasks for adversarial robustness: 
- Object 
- Attribute
- Spatial

## Directory Structure

- `generate_text_to_image.py`: Script to generate images from text prompts.
- `generate_image_to_text.py`: Script to generate text responses from images.
- `eval_text_to_image.py`: Script to evaluate the generated iamges.
- `eval_image_to_text.py`: Script to evaluate the generated text responses.
- `utils`: Folder containing the GroundingDINO detector tools.

## Usage
### Download Checkpoint

Download pre-trained Swin-T model weights for GroundingDINO.
```
cd mmdt/detection/GroundingDINO
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### Generation

T2I Generation:
```
python mmdt/perspectives/adv/generate_text_to_image.py --model_id {model_id} --sub_task {sub_task} --image_number {image_number}
```

I2T Generation:
```
python mmdt/perspectives/adv/generate_image_to_text.py --model_id {model_id} --sub_task {sub_task}
```

T2I Evaluation:
```
python mmdt/perspectives/adv/eval_text_to_image.py --model_id {model_id} --sub_task {sub_task} --image_number {image_number}
```

I2T Evaluation
```
python mmdt/perspectives/adv/eval_image_to_text.py --model_id {model_id} --sub_task {sub_task}
```