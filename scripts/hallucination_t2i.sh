#!/bin/bash

# Usage: ./hallucination_t2i.sh <model_id>


MODEL_ID=$1

# Make sure the user supplied a model_id
if [ -z "$MODEL_ID" ]; then
  echo "Error: You must provide a model_id as the first argument."
  echo "Usage: $0 <model_id>"
  exit 1
fi

# Arrays for scenarios and tasks
SCENARIOS=("natural" "counterfactual" "misleading" "distraction" "ocr" "cooccurrence")
TASKS=("identification" "attribute" "spatial" "count")

# Loop over each scenario and task
for scenario in "${SCENARIOS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "Running with scenario=${scenario}, task=${task}, model_id=${MODEL_ID}"
    python ./mmdt/main.py \
      --modality text_to_image \
      --model_id "${MODEL_ID}" \
      --perspectives hallucination \
      --scenario "${scenario}" \
      --task "${task}"
  done
done