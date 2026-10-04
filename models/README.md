# Trained models (ONNX)

The application expects these seven files in this folder (they are not committed, see the top-level README):

| File | Task | Input → output |
|---|---|---|
| `universal_restoration.onnx` | 1 | image `[0,1]` NCHW 128×128 → restored image |
| `classifier.onnx` | 2 | image → 4 class probabilities (clean, salt-and-pepper, blur, occlusion) |
| `specialist_salt.onnx`, `specialist_blur.onnx`, `specialist_occlusion.onnx` | 2 | image → restored image |
| `soft_moe.onnx` | 3 | image → (restored image, 4 routing weights: identity, salt, blur, occlusion) |
| `sketch_generator.onnx` | 4 | (photo `[-1,1]`, style index int64 0..2) → sketch `[-1,1]` |

Download: `bash scripts/download_models.sh` (fetches them from the [models-v1 release](https://github.com/AfnanRizwan1/GenAI-A1/releases/tag/models-v1)), or produce them yourself with
`python -m src.export.export_onnx` (see the top-level README).
