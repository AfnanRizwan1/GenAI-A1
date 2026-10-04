#!/usr/bin/env bash
# Downloads the trained ONNX models into ./models.
#   MODELS_BASE_URL=https://github.com/<user>/<repo>/releases/download/<tag> bash scripts/download_models.sh
set -euo pipefail
: "${MODELS_BASE_URL:?set MODELS_BASE_URL to the folder / release that holds the .onnx files (see README)}"
cd "$(dirname "$0")/.."
mkdir -p models
for f in universal_restoration classifier specialist_salt specialist_blur specialist_occlusion soft_moe sketch_generator; do
  if [ -f "models/$f.onnx" ]; then echo "have models/$f.onnx"; else
    echo "downloading $f.onnx"; curl -fL --retry 3 -o "models/$f.onnx" "$MODELS_BASE_URL/$f.onnx"
  fi
done
echo "done: $(ls models/*.onnx | wc -l) models in ./models"
