#!/bin/bash

# Exit on any error
set -e

INPUT=ailia-models-tflite/image_classification/resnet50/clock.jpg
SCRIPT=ailia-models-tflite/image_classification/resnet50/resnet50.py
TFLITE_FILE="${SCRIPT%.py}_quant_recalib.tflite"
MLIR_BYTECODE="${SCRIPT%.py}.bc.mlir"
MLIR_19="${SCRIPT%.py}.old.mlir"
OUTPUT="outputs/resnet50_int8.mlir"
OUTPUT_DIR=$(dirname "$OUTPUT")
JSON_DATA="${OUTPUT%.mlir}_sample_data.json"

# Run python script
python ailia-models-tflite/image_classification/resnet50/resnet50.py -i ailia-models-tflite/image_classification/resnet50/clock.jpg

# Create mlir bytecode from tflite flatbuffer
python tflite_to_tosa.py -c "$TFLITE_FILE" -o "$MLIR_BYTECODE"

# print in generic format
mlir-opt "$MLIR_BYTECODE" -o "$MLIR_19" --mlir-print-op-generic --mlir-print-local-scope

# convert to mlir 20
python mlir_19_to_20.py -i "$MLIR_19" -o "$OUTPUT"

# remove intermediate files
rm "$MLIR_BYTECODE" "$MLIR_19"

python tflite_generate_data.py "$TFLITE_FILE" "$JSON_DATA"

echo ""
