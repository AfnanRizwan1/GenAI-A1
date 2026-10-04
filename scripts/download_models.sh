#!/usr/bin/env bash
# Downloads the trained ONNX models into ./models (default: the models-v1 release of this repository).
#   bash scripts/download_models.sh
#   MODELS_BASE_URL=<other folder holding the .onnx files> bash scripts/download_models.sh
set -euo pipefail
MODELS_BASE_URL="${MODELS_BASE_URL:-https://github.com/AfnanRizwan1/GenAI-A1/releases/download/models-v1}"
cd "$(dirname "$0")/.."
mkdir -p models
for f in universal_restoration classifier specialist_salt specialist_blur specialist_occlusion soft_moe sketch_generator; do
  if [ -f "models/$f.onnx" ]; then echo "have models/$f.onnx"; else
    echo "downloading $f.onnx"; curl -fL --retry 3 -o "models/$f.onnx" "$MODELS_BASE_URL/$f.onnx"
  fi
done
echo "done: $(ls models/*.onnx | wc -l) models in ./models"
