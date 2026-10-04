"""Backend settings, read from environment variables (set in docker-compose.yml)."""
import os
from pathlib import Path

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", Path(__file__).resolve().parents[1] / "samples"))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "10"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(40_000_000)))
CORS_ORIGINS = [o for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o]
IMAGE_SIZE = 128
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# ONNX files written by src.export.export_onnx
MODEL_FILES = {
    "universal": "universal_restoration.onnx",
    "classifier": "classifier.onnx",
    "specialist_salt": "specialist_salt.onnx",
    "specialist_blur": "specialist_blur.onnx",
    "specialist_occlusion": "specialist_occlusion.onnx",
    "soft_moe": "soft_moe.onnx",
    "sketch": "sketch_generator.onnx",
}
CLASS_NAMES = ["clean", "salt-and-pepper", "gaussian blur", "occlusion"]
BRANCH_NAMES = ["identity (clean)", "salt-and-pepper expert", "blur expert", "occlusion expert"]
STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]
