#!/bin/bash

# Exit on any error
set -e

# List of input/output pairs
declare -A MODELS=(
  ["mlperf_tiny/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite"]="outputs/mlperf_tiny_resnet18_int8.mlir"
  ["mlperf_tiny/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite"]="outputs/mlperf_tiny_mobilenet_int8.mlir"
  ["mlperf_tiny/benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite"]="outputs/mlperf_tiny_anomaly_int8.mlir"
  ["mlperf_tiny/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite"]="outputs/mlperf_tiny_kws_int8.mlir"
)

# Iterate over models and run the converter
for INPUT in "${!MODELS[@]}"; do
  OUTPUT="${MODELS[$INPUT]}"
  MLIR_BYTECODE="${OUTPUT%.mlir}.bc.mlir"
  MLIR_19="${OUTPUT%.mlir}.old.mlir"
  OUTPUT_DIR=$(dirname "$OUTPUT")

  echo "Processing: $INPUT →  $OUTPUT"
  mkdir -p "$OUTPUT_DIR"

  # create mlir bytecode from tflite flatbuffer
  python tflite_to_tosa.py -c "$INPUT" -o "$MLIR_BYTECODE"

  # print in generic format
  mlir-opt "$MLIR_BYTECODE" -o "$MLIR_19" --mlir-print-op-generic --mlir-print-local-scope

  # convert to mlir 20
  python mlir_19_to_20.py -i "$MLIR_19" -o "$OUTPUT"

  # remove intermediate files
  rm "$MLIR_BYTECODE" "$MLIR_19"

  echo ""

done
