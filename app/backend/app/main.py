"""FastAPI backend for the four workspaces.

    GET  /api/health      loaded models, onnxruntime version
    GET  /api/meta        class / style names, corruption presets
    GET  /api/samples     bundled clean sample images  (GET /api/samples/{id} returns one)
    POST /api/universal   Task 1  Universal Restoration
    POST /api/hard        Task 2  Hard-Routed Restoration
    POST /api/soft        Task 3  Soft Mixture-of-Experts Restoration
    POST /api/sketch      Task 4  Face-to-Sketch Generator
"""
import secrets
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import config, imaging
from .models import ModelStore

SAMPLE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def create_app(models_dir: Path | str | None = None, samples_dir: Path | str | None = None) -> FastAPI:
    app = FastAPI(title="Generative AI Assignment 1 - restoration and sketch API", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    store = ModelStore(models_dir or config.MODELS_DIR)
    samples = Path(samples_dir or config.SAMPLES_DIR)

    def list_samples():
        return sorted(p for p in samples.glob("*") if p.suffix.lower() in SAMPLE_EXT) if samples.exists() else []

    async def load_input(file: UploadFile | None, sample_id: str | None) -> np.ndarray:
        """The image to work on: an upload, or one of the bundled samples."""
        if file is not None and file.filename:
            data = await file.read(int(config.MAX_UPLOAD_MB * 1024 * 1024) + 1)  # +1 byte so oversize is detected
            return imaging.decode_image(data, file.content_type)
        if sample_id:
            match = [p for p in list_samples() if p.stem == sample_id]
            if not match:
                raise HTTPException(404, f"unknown sample '{sample_id}'")
            return imaging.decode_image(match[0].read_bytes())
        raise HTTPException(422, "send an image file or a sample_id")

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        status = store.status()
        return {"status": "ok", "ready": all(v["loaded"] for v in status.values()), "models": status,
                "onnxruntime": ort.__version__, "providers": ort.get_available_providers()}

    @app.get("/api/meta")
    def meta():
        return {"image_size": config.IMAGE_SIZE, "classes": config.CLASS_NAMES, "branches": config.BRANCH_NAMES,
                "styles": config.STYLE_NAMES, "corruptions": list(imaging.KINDS), "severities": list(imaging.SEVERITIES),
                "max_upload_mb": config.MAX_UPLOAD_MB}

    @app.get("/api/samples")
    def samples_list():
        return [{"id": p.stem, "name": p.stem.replace("_", " "), "url": f"/api/samples/{p.stem}"} for p in list_samples()]

    @app.get("/api/samples/{sample_id}")
    def sample_file(sample_id: str):
        match = [p for p in list_samples() if p.stem == sample_id]
        if not match:
            raise HTTPException(404, "unknown sample")
        return FileResponse(match[0])

    # ------------------------------------------------------------------ shared request handling
    async def prepare(file, sample_id, corruption, severity, seed, salt_p, blur_kernel, blur_sigma, occ_coverage, occ_rects):
        clean = await load_input(file, sample_id)
        seed = secrets.randbelow(2 ** 31) if seed is None else seed
        spec = imaging.build_spec(corruption, severity, seed, salt_p, blur_kernel, blur_sigma, occ_coverage, occ_rects)
        applied = corruption != "none"
        corrupted = imaging.apply_spec(clean, spec) if applied else clean
        return clean, corrupted, spec, applied

    def response(clean, corrupted, spec, applied, restored, ms, extra):
        return {"original": imaging.to_data_url(clean) if applied else None, "input": imaging.to_data_url(corrupted),
                "restored": imaging.to_data_url(restored), "settings": imaging.describe_spec(spec),
                "difference": imaging.to_data_url(imaging.difference_map(restored, clean)) if applied else None,
                "difference_full_scale": imaging.DIFF_FULL_SCALE,
                "metrics": imaging.quality(clean, corrupted, restored) if applied else None,
                "inference_ms": round(ms, 2), "image_size": config.IMAGE_SIZE, **extra}

    # ------------------------------------------------------------------ Task 1
    @app.post("/api/universal")
    async def universal(file: UploadFile | None = File(None), sample_id: str | None = Form(None),
                        corruption: str = Form("none"), severity: str = Form("medium"), seed: int | None = Form(None),
                        salt_p: float = Form(0.08, ge=0.0, le=0.5), blur_kernel: int = Form(5, ge=3, le=15),
                        blur_sigma: float = Form(1.5, gt=0.0, le=6.0), occ_coverage: float = Form(0.2, ge=0.02, le=0.5),
                        occ_rects: int = Form(2, ge=1, le=4)):
        t0 = time.perf_counter()
        clean, corrupted, spec, applied = await prepare(file, sample_id, corruption, severity, seed, salt_p, blur_kernel,
                                                        blur_sigma, occ_coverage, occ_rects)
        y, ms = store.universal(imaging.to_nchw(corrupted))
        return response(clean, corrupted, spec, applied, imaging.from_nchw(y), ms,
                        {"total_ms": round((time.perf_counter() - t0) * 1000, 2)})

    # ------------------------------------------------------------------ Task 2
    @app.post("/api/hard")
    async def hard(file: UploadFile | None = File(None), sample_id: str | None = Form(None),
                   corruption: str = Form("none"), severity: str = Form("medium"), seed: int | None = Form(None),
                   salt_p: float = Form(0.08, ge=0.0, le=0.5), blur_kernel: int = Form(5, ge=3, le=15),
                   blur_sigma: float = Form(1.5, gt=0.0, le=6.0), occ_coverage: float = Form(0.2, ge=0.02, le=0.5),
                   occ_rects: int = Form(2, ge=1, le=4)):
        t0 = time.perf_counter()
        clean, corrupted, spec, applied = await prepare(file, sample_id, corruption, severity, seed, salt_p, blur_kernel,
                                                        blur_sigma, occ_coverage, occ_rects)
        r = store.hard_routed(imaging.to_nchw(corrupted))
        probs = [float(p) for p in r["probabilities"]]
        expert = {None: "none (identity bypass)", "specialist_salt": "salt-and-pepper specialist",
                  "specialist_blur": "blur specialist", "specialist_occlusion": "occlusion specialist"}[r["expert"]]
        extra = {"probabilities": dict(zip(config.CLASS_NAMES, probs)), "predicted_class": config.CLASS_NAMES[r["predicted"]],
                 "selected_expert": expert, "classifier_ms": round(r["classifier_ms"], 2), "expert_ms": round(r["expert_ms"], 2),
                 "total_ms": round((time.perf_counter() - t0) * 1000, 2)}
        return response(clean, corrupted, spec, applied, imaging.from_nchw(r["restored"]),
                        r["classifier_ms"] + r["expert_ms"], extra)

    # ------------------------------------------------------------------ Task 3
    @app.post("/api/soft")
    async def soft(file: UploadFile | None = File(None), sample_id: str | None = Form(None),
                   corruption: str = Form("none"), severity: str = Form("medium"), seed: int | None = Form(None),
                   salt_p: float = Form(0.08, ge=0.0, le=0.5), blur_kernel: int = Form(5, ge=3, le=15),
                   blur_sigma: float = Form(1.5, gt=0.0, le=6.0), occ_coverage: float = Form(0.2, ge=0.02, le=0.5),
                   occ_rects: int = Form(2, ge=1, le=4)):
        t0 = time.perf_counter()
        clean, corrupted, spec, applied = await prepare(file, sample_id, corruption, severity, seed, salt_p, blur_kernel,
                                                        blur_sigma, occ_coverage, occ_rects)
        y, w, ms = store.soft_moe(imaging.to_nchw(corrupted))
        weights = [float(v) for v in w]
        order = np.argsort(weights)[::-1]
        extra = {"weights": dict(zip(config.BRANCH_NAMES, weights)),
                 "dominant_branch": config.BRANCH_NAMES[int(order[0])],
                 "routing_entropy": imaging.routing_entropy(weights),
                 "contribution_ranking": [{"branch": config.BRANCH_NAMES[int(i)], "weight": weights[int(i)]} for i in order],
                 "total_ms": round((time.perf_counter() - t0) * 1000, 2)}
        return response(clean, corrupted, spec, applied, imaging.from_nchw(y), ms, extra)

    # ------------------------------------------------------------------ Task 4
    @app.post("/api/sketch")
    async def sketch(file: UploadFile | None = File(None), sample_id: str | None = Form(None),
                     style: int = Form(1, ge=1, le=3)):
        t0 = time.perf_counter()
        photo = await load_input(file, sample_id)
        y, ms = store.sketch(imaging.to_nchw(photo * 2 - 1), style - 1)
        sketch_img = np.clip((y[0].transpose(1, 2, 0) + 1) / 2, 0, 1)
        return {"photo": imaging.to_data_url(photo), "sketch": imaging.to_data_url(sketch_img),
                "style": config.STYLE_NAMES[style - 1], "inference_ms": round(ms, 2),
                "total_ms": round((time.perf_counter() - t0) * 1000, 2), "image_size": config.IMAGE_SIZE}

    return app


app = create_app()
