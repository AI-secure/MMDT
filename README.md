# MMDT: Decoding the Trustworthiness and Safety of Multimodal Foundation Models

[[Website 🌐]](https://mmdecodingtrust.github.io/), [[Text-to-Image data 🤗]](https://huggingface.co/datasets/AI-Secure/MMDecodingTrust-T2I), [[Image-to-Text data 🤗]](https://huggingface.co/datasets/AI-Secure/MMDecodingTrust-I2T)

## Overview

This repo contains the source code of MMDT (Multimodal DecodingTrust). This research endeavor is designed to help researchers and practitioners better understand the capabilities, limitations, and potential risks involved in deploying these state-of-the-art Multimodal foundation models (MMFMs). See our paper for details.


This project is organized around the following six primary perspectives of trustworthiness, including:
1. Safety
2. Hallucination
3. Fairness
4. Privacy
5. Adversarial robustness
6. Out-of-Distribution Robustness

## Project Structure
This project is structured around subdirectories dedicated to each area of trustworthiness. Each subdir includes scripts, data, and a dedicated README for easy comprehension.


## Getting Started

### Clone the repository

```bash
git clone https://github.com/AI-secure/MMDT.git && cd MMDT
```

### Install requirements

Create a new environment:

```bash
conda create --name mmdt python=3.9
conda activate mmdt
```

Install PyTorch following [this link](https://pytorch.org/get-started/locally/). Then install the requirements:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Evaluate all perspectives

```bash
bash scripts/t2i.sh {model_id}  # Evaluate a text-to-image model
bash scripts/i2t.sh {model_id}  # Evaluate an image-to-text model
```

### Evaluate each perspective

We also provide off-the-shelf scripts for evaluating each perspective under `./scripts`.
For example, the following script evaluates all scenarios and tasks of image-to-text modality for the hallucination perspective.
```bash
bash scripts/hallucination_i2t.sh gpt-4o
```
An example of the output summarized score can be found [here](./mmdt/perspectives/hallucination/README.md?plain=1#L76).

Moreover, you can customize the evaluation process with specific perspective, scenario, and task by running the following script:
```
python mmdt/main.py --modality {modality} --model_id {model_id} --perspectives {perspective} --scenario {scenario} --task {task}
```

For example, to evaluate `gpt-4o` on hallucination under `natural selection` scenario and `action recognition` task, we can run the following example script.
```bash
python mmdt/main.py --modality image_to_text --model_id gpt-4o --perspectives hallucination --scenario natural --task action
```

Our framework includes the following perspectives, scenarios, and tasks:
```
Text-to-image models
├── safety
│   ├── vanilla
│   ├── transformed
│   └── jailbreak
├── hallucination
│   ├── natural
│   │   ├── identification
│   │   ├── attribute
│   │   ├── spatial
│   │   └── count
│   ├── counterfactual
│   │   ├── identification
│   │   ├── attribute
│   │   ├── spatial
│   │   └── count
│   ├── misleading
│   │   ├── identification
│   │   ├── attribute
│   │   ├── spatial
│   │   └── count
│   ├── distraction
│   │   ├── identification
│   │   ├── attribute
│   │   ├── spatial
│   │   └── count
│   ├── ocr
│   │   ├── complex
│   │   ├── contradictory
│   │   ├── distortion
│   │   └── misleading
│   └── cooccurrence
│       ├── identification
│       ├── attribute
│       ├── spatial
│       └── count
├── fairness
│   ├── stereotype
│   ├── decision_making
│   ├── overkill
│   └── individual
├── privacy
│   └── training
│       └── laion_1k
├── adv
│   └── adv
│       ├── object
│       ├── attribute
│       └── spatial
└── ood
    ├── Shake_
    │   ├── helpfulness
    │   ├── count
    │   ├── spatial
    │   ├── color
    │   └── size
    └── Paraphrase_
        ├── helpfulness
        ├── count
        ├── spatial
        ├── color
        └── size

Image-to-text models
├── safety
│   ├── typography
│   ├── illustration
│   └── jailbreak
├── hallucination
│   ├── natural
│   │   ├── identification
│   │   ├── attribute
│   │   ├── spatial
│   │   ├── count
│   │   └── action
│   ├── counterfactual
│   │   ├── identification
│   │   ├── attribute
│   │   ├── spatial
│   │   └── count
│   ├── misleading
│   │   ├── identification
│   │   ├── attribute
│   │   ├── spatial
│   │   ├── count
│   │   └── action
│   ├── distraction
│   │   ├── identification
│   │   ├── attribute
│   │   ├── spatial
│   │   ├── count
│   │   └── action
│   ├── ocr
│   │   ├── contradictory
│   │   ├── cooccur
│   │   ├── doc
│   │   └── scene
│   └── cooccurrence
│       ├── identification
│       ├── attribute
│       ├── spatial
│       ├── count
│       └── action
├── fairness
│   ├── stereotype
│   ├── decision_making
│   ├── overkill
│   └── individual
├── privacy
│   ├── location
│   │   ├── Pri-SV-with-text
│   │   ├── Pri-SV-without-text
│   │   ├── Pri-4Loc-SV-with-text
│   │   └── Pri-4Loc-SV-without-text
│   └── pii
├── adv
│   └── adv
│       ├── object
│       ├── attribute
│       └── spatial
└── ood
    ├── Van_Gogh
    │   ├── attribute
    │   ├── count
    │   ├── spatial
    │   └── identification
    ├── oil_painting
    │   ├── attribute
    │   ├── count
    │   ├── spatial
    │   └── identification
    ├── watercolour_painting
    │   ├── attribute
    │   ├── count
    │   ├── spatial
    │   └── identification
    ├── gaussian_noise
    │   ├── attribute
    │   ├── count
    │   ├── spatial
    │   └── identification
    ├── zoom_blur
    │   ├── attribute
    │   ├── count
    │   ├── spatial
    │   └── identification
    └── pixelate
        ├── attribute
        ├── count
        ├── spatial
        └── identification
```


### Notes
+ Each of the six perspectives has its subdirectory containing the respective code and README.

+ Follow the specific `README`: Every subdirectory has its own README. Refer to these documents for information on how to run the scripts and interpret the results.

## License
This project is licensed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)  - see the LICENSE file for details.

## Contact
Please reach out to us if you have any questions or suggestions. You can submit an issue or pull request, or send an email to chejian2@illinois.edu.

Thank you for your interest in MMDT. We hope our work will contribute to a more trustworthy, fair, and robust AI future.
