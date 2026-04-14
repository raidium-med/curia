#!/bin/bash
set -e

# Default paths for Docker environment
INPUT_DIR="${INPUT_DIR:-/workspace/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs}"
MASKS_DIR="${MASKS_DIR:-}"
NUM_CLASSES="${NUM_CLASSES:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MODEL_PATH="${MODEL_PATH:-/opt/app/model}"

# Build command with optional arguments
CMD="python3 extract_feat_LP.py -i \"$INPUT_DIR\" -o \"$OUTPUT_DIR\" --model_path \"$MODEL_PATH\""

if [ -n "$MASKS_DIR" ]; then
    CMD="$CMD --masks_path \"$MASKS_DIR\""
fi

if [ -n "$NUM_CLASSES" ]; then
    CMD="$CMD --num_classes $NUM_CLASSES"
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch_size $BATCH_SIZE"
fi

# Run feature extraction
eval $CMD
