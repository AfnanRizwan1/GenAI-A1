"""Upload validation, preprocessing, corruption settings, image encoding and quality metrics."""
import base64
import io

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError

from . import config

try:  # inside the repo: the very module the models were trained with (numpy only, no torch)
    from src.data.corruption_spec import apply_spec, make_spec, occlusion_mask, sample_rects
except ImportError:  # inside the Docker image: copied next to the app
    from corruption_spec import apply_spec, make_spec, occlusion_mask, sample_rects  # type: ignore  # noqa: F401

Image.MAX_IMAGE_PIXELS = None  # we enforce our own limit below, before decoding


def decode_image(data: bytes, content_type: str | None = None) -> np.ndarray:
    """Validate an uploaded image and return float32 HxWx3 in [0,1] resized to 128x128 (like in training)."""
    if content_type is not None and content_type not in config.ALLOWED_TYPES:
        raise HTTPException(415, f"unsupported file type '{content_type}', use JPEG, PNG, WebP or BMP")
    if len(data) == 0:
        raise HTTPException(400, "empty file")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file larger than {config.MAX_UPLOAD_MB:g} MB")
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > config.MAX_PIXELS:
            raise HTTPException(413, "image has too many pixels")
        if min(img.size) < 16:
            raise HTTPException(400, "image is too small (minimum 16x16 pixels)")
        img = ImageOps.exif_transpose(img).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise HTTPException(400, "the file is not a readable image") from None
    img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def to_data_url(img01: np.ndarray) -> str:
    arr = (np.clip(img01, 0, 1) * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def to_nchw(img01: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img01.transpose(2, 0, 1)[None], dtype=np.float32)


def from_nchw(x: np.ndarray) -> np.ndarray:
    return np.clip(x[0].transpose(1, 2, 0), 0, 1)


# ---------------------------------------------------------------- corruption settings
SEVERITIES = ("low", "medium", "high", "custom")
KINDS = ("none", "salt", "blur", "occlusion")


def build_spec(kind: str, severity: str, seed: int, salt_p=0.08, blur_kernel=5, blur_sigma=1.5,
               occ_coverage=0.2, occ_rects=2) -> dict:
    """Corruption spec for one image: a preset severity (the assignment's fixed test levels) or custom parameters."""
    if kind not in KINDS:
        raise HTTPException(422, f"corruption must be one of {KINDS}")
    if severity not in SEVERITIES:
        raise HTTPException(422, f"severity must be one of {SEVERITIES}")
    if kind == "none":
        return {"type": "clean", "level": None, "seed": seed}
    rng = np.random.default_rng(seed)
    if severity != "custom":
        spec = make_spec(kind, rng, seed, level=severity, fixed=True)
    else:
        spec = {"type": kind, "level": "custom", "seed": seed}
        if kind == "salt":
            spec["p"] = float(salt_p)
        elif kind == "blur":
            if blur_kernel % 2 == 0:
                raise HTTPException(422, "blur_kernel must be odd")
            spec["k"], spec["sigma"] = int(blur_kernel), float(blur_sigma)
        else:
            spec["rects"] = sample_rects(rng, int(occ_rects), float(occ_coverage))
            spec["coverage"] = float(occlusion_mask(spec["rects"]).mean())
    return spec


def describe_spec(spec: dict) -> dict:
    """JSON-friendly settings shown in the UI."""
    d = {"type": spec["type"], "severity": spec.get("level"), "seed": spec["seed"]}
    for k in ("p", "k", "sigma", "coverage"):
        if k in spec:
            d[{"p": "probability", "k": "kernel_size", "sigma": "sigma", "coverage": "area_covered"}[k]] = round(spec[k], 4)
    if "rects" in spec:
        d["rectangles"] = spec["rects"]
    return d


# ---------------------------------------------------------------- metrics
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b) ** 2))
    return 99.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray, data_range=1.0) -> float:
    """Mean SSIM, Gaussian 11x11 window (sigma 1.5), valid region only -- the same definition as the training loss."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)[5:-5, 5:-5]
    mu_a, mu_b = blur(a), blur(b)
    s_aa, s_bb, s_ab = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return float(m.mean())


def quality(clean: np.ndarray, corrupted: np.ndarray, restored: np.ndarray) -> dict:
    return {"input_psnr": round(psnr(corrupted, clean), 2), "restored_psnr": round(psnr(restored, clean), 2),
            "input_ssim": round(ssim(corrupted, clean), 4), "restored_ssim": round(ssim(restored, clean), 4)}


DIFF_FULL_SCALE = 0.25  # absolute error (in [0,1] pixel units) that maps to the top of the colour scale


def difference_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Heat map of |a - b| averaged over the colour channels (inferno colormap, DIFF_FULL_SCALE = brightest)."""
    err = np.abs(a.astype(np.float64) - b).mean(axis=2)
    gray = (np.clip(err / DIFF_FULL_SCALE, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def routing_entropy(weights) -> dict:
    """Entropy of the routing weights: 0 = one branch takes everything, 1 = perfectly uniform (normalised by ln K)."""
    w = np.clip(np.asarray(weights, dtype=np.float64), 1e-12, 1.0)
    nats = float(-(w * np.log(w)).sum())
    return {"nats": round(nats, 4), "normalized": round(nats / float(np.log(len(w))), 4)}
