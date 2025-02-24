#!/bin/bash

# Usage: bash ./scripts/hallucination_t2i.sh <model_id>

MODEL_ID=$1

# Make sure the user supplied a model_id
if [ -z "$MODEL_ID" ]; then
  echo "Error: You must provide a model_id as the first argument."
  echo "Usage: $0 <model_id>"
  exit 1
fi

# Scenarios
SCENARIOS=("natural" "counterfactual" "misleading" "distraction" "ocr" "cooccurrence")


# Default tasks (for all scenarios except "ocr")
DEFAULT_TASKS=("identification" "attribute" "spatial" "count")

# Tasks only for the "ocr" scenario
OCR_TASKS=("complex" "contradictory" "distortion" "misleading")

# Loop over each scenario
for scenario in "${SCENARIOS[@]}"; do
  
  # Decide which tasks to run based on the scenario
  if [ "$scenario" == "ocr" ]; then
    tasks=("${OCR_TASKS[@]}")
  else
    tasks=("${DEFAULT_TASKS[@]}")
  fi

  # Loop over tasks for the current scenario
  for task in "${tasks[@]}"; do
    echo "Running with scenario=${scenario}, task=${task}, model_id=${MODEL_ID}"
    python ./mmdt/main.py \
      --modality text_to_image \
      --model_id "${MODEL_ID}" \
      --perspectives hallucination \
      --scenario "${scenario}" \
      --task "${task}"
  done
done