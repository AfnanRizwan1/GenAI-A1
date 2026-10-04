"""ONNX model store: loads the exported models once and runs inference with timing."""
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import HTTPException

from . import config


class ModelStore:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.sessions: dict[str, ort.InferenceSession] = {}
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        for name, fname in config.MODEL_FILES.items():
            path = self.models_dir / fname
            if path.exists():
                self.sessions[name] = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])

    def status(self) -> dict:
        return {name: {"file": fname, "loaded": name in self.sessions} for name, fname in config.MODEL_FILES.items()}

    def _session(self, name: str) -> ort.InferenceSession:
        if name not in self.sessions:
            raise HTTPException(503, f"model '{name}' is not available (expected {config.MODEL_FILES[name]} in {self.models_dir})")
        return self.sessions[name]

    def _run(self, name: str, feed: dict):
        sess = self._session(name)
        t0 = time.perf_counter()
        out = sess.run(None, feed)
        return out, (time.perf_counter() - t0) * 1000.0

    # Task 1
    def universal(self, x: np.ndarray):
        (y,), ms = self._run("universal", {"image": x})
        return y, ms

    # Task 2: classifier probabilities -> argmax -> expert (clean: identity bypass, no expert call)
    def hard_routed(self, x: np.ndarray):
        (probs,), cls_ms = self._run("classifier", {"image": x})
        probs = probs[0]
        pred = int(np.argmax(probs))
        if pred == 0:
            return {"probabilities": probs, "predicted": 0, "expert": None, "restored": x,
                    "classifier_ms": cls_ms, "expert_ms": 0.0}
        expert = ("specialist_salt", "specialist_blur", "specialist_occlusion")[pred - 1]
        (y,), exp_ms = self._run(expert, {"image": x})
        return {"probabilities": probs, "predicted": pred, "expert": expert, "restored": y,
                "classifier_ms": cls_ms, "expert_ms": exp_ms}

    # Task 3
    def soft_moe(self, x: np.ndarray):
        (y, w), ms = self._run("soft_moe", {"image": x})
        return y, w[0], ms

    # Task 4: photo in [-1,1], style index 0..2 -> sketch in [-1,1]
    def sketch(self, photo: np.ndarray, style: int):
        (y,), ms = self._run("sketch", {"photo": photo, "style": np.array([style], dtype=np.int64)})
        return y, ms
