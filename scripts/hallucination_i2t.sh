#!/bin/bash

# Usage: ./hallucination_i2t.sh <model_id>


MODEL_ID=$1

# Make sure the user supplied a model_id
if [ -z "$MODEL_ID" ]; then
  echo "Error: You must provide a model_id as the first argument."
  echo "Usage: $0 <model_id>"
  exit 1
fi

# Arrays for scenarios and tasks
SCENARIOS=("natural" "counterfactual" "misleading" "distraction" "ocr" "cooccurrence")
TASKS=("identification" "attribute" "spatial" "count" "action")

# Loop over each scenario and task
for scenario in "${SCENARIOS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "Running with scenario=${scenario}, task=${task}, model_id=${MODEL_ID}"
    python ./mmdt/main.py \
      --modality image_to_text \
      --model_id "${MODEL_ID}" \
      --perspectives hallucination \
      --scenario "${scenario}" \
      --task "${task}"
  done
done