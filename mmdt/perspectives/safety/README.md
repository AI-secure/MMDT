# Safety Perspective

This directory contains scripts for evaluating safety perspectives in both text-to-image and image-to-text formats as part of the MMDT project.

There contains six scenarios for safety: 
= Image-to-Text
    - Typography
    - Illustration
    - Jailbreak

- Text-to-Image
    - Vanilla
    - Transformed
    = Jailbreak

## Directory Structure

- `generate_text_to_image.py`: Script to generate images from text prompts. This is empty. Data can directly be loaded from HuggingFace.
- `generate_image_to_text.py`: Script to generate text responses from images. This is empty. Data can directly be loaded from HuggingFace.
- `eval_text_to_image.py`: Script to evaluate the generated images.
- `eval_image_to_text.py`: Script to evaluate the generated text responses.