#!/bin/bash

# Usage: bash ./scripts/hallucination_i2t.sh <model_id>

MODEL_ID=$1

# Make sure the user supplied a model_id
if [ -z "$MODEL_ID" ]; then
  echo "Error: You must provide a model_id as the first argument."
  echo "Usage: $0 <model_id>"
  exit 1
fi

# Scenarios
SCENARIOS=("natural" "counterfactual" "misleading" "distraction" "ocr" "cooccurrence")

# Tasks for most scenarios
DEFAULT_TASKS=("identification" "attribute" "spatial" "count" "action")

# Tasks only for the "ocr" scenario
OCR_TASKS=("contradictory" "cooccur" "doc" "scene")

COUNTERFACTUAL_TASKS=("identification" "attribute" "spatial" "count")


# Loop over each scenario
for scenario in "${SCENARIOS[@]}"; do
  
  # If the scenario is "ocr", use OCR_TASKS; otherwise, use DEFAULT_TASKS
  if [ "$scenario" == "ocr" ]; then
    tasks=("${OCR_TASKS[@]}")
  elif [ "$scenario" == "counterfactual" ]; then
    tasks=("${COUNTERFACTUAL_TASKS[@]}")
  else
    tasks=("${DEFAULT_TASKS[@]}")
  fi

  # Loop over tasks for the current scenario
  for task in "${tasks[@]}"; do
    echo "Running with scenario=${scenario}, task=${task}, model_id=${MODEL_ID}"
    python ./mmdt/main.py \
      --modality image_to_text \
      --model_id "${MODEL_ID}" \
      --perspectives hallucination \
      --scenario "${scenario}" \
      --task "${task}"
  done
done