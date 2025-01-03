# MMDT: Decoding the Trustworthiness and Safety of Multimodal Foundation Models

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

```bash
conda create --name mmdt python=3.9
conda activate mmdt
pip install -r requirements.txt
```

### Load MMDT dataset

```python
from datasets import load_dataset

ds = load_dataset("AI-Secure/MMDecodingTrust-T2I", "hallucination")  # Load Hallucination subset of Text-to-Image dataset

ds = load_dataset("AI-Secure/MMDecodingTrust-I2T", "fairness")  # Load Fairness subset of Image-to-Text dataset
```

Please refer to our HF datasets for more details ([Text-to-Image](https://huggingface.co/datasets/AI-Secure/MMDecodingTrust-T2I), [Image-to-Text](https://huggingface.co/datasets/AI-Secure/MMDecodingTrust-I2T)).

### Evaluate all perspectives

```bash
python mmdt/main.py --model_id {model_id}
```

### Evaluate each perspective

To evaluate MMFMs with MMDT, we provide script examples for different perspectives under `./scripts`. For example, to evaluate `llava-hf/llava-v1.6-mistral-7b-hf` on hallucination under `natural selection` scenario and `action recognition` task, we can run the following example script.
```bash
bash script/hallucination.sh
```

### Notes
+ Each of the six areas has its subdirectory containing the respective code and README.

+ Follow the specific `README`: Every subdirectory has its own README. Refer to these documents for information on how to run the scripts and interpret the results.

## License
This project is licensed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)  - see the LICENSE file for details.

## Contact
Please reach out to us if you have any questions or suggestions. You can submit an issue or pull request, or send an email to chejian2@illinois.edu.

Thank you for your interest in MMDT. We hope our work will contribute to a more trustworthy, fair, and robust AI future.
