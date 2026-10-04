# Generative AI — Assignment 1: restoration, mixture-of-experts and face-to-sketch

Four generative models behind one web application (**Restoration Studio**):

| Workspace | Task | Model |
|---|---|---|
| Universal Restoration | 1 | One convolutional denoising autoencoder with a compressed bottleneck, restores clean / salt-and-pepper / blur / occlusion images |
| Hard-Routed Restoration | 2 | Corruption classifier → one of three specialist autoencoders (clean images bypass the experts) |
| Soft Mixture-of-Experts Restoration | 3 | Gate + identity branch + three experts, trained jointly; every branch gets a continuous weight |
| Face-to-Sketch Generator | 4 | Style-conditioned cGAN (U-Net generator with a learned style embedding, PatchGAN discriminator) on FS2K |

Stack: PyTorch, Optuna, MLflow, ONNX / ONNX Runtime, FastAPI, React + Tailwind CSS, Docker Compose.
Datasets: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (Tasks 1–3), [FS2K](https://github.com/DengPingFan/FS2K) (Task 4).
Datasets and trained models are **not** in the repository (download instructions below).

---

## 1. Run the application (evaluator quick start)

Requirements: Docker with Compose v2. No Python or Node is needed.

```bash
git clone https://github.com/AfnanRizwan1/GenAI-A1.git && cd GenAI-A1

# 1. get the trained ONNX models into ./models  (7 files, ~200 MB, from the models-v1 release; see models/README.md)
bash scripts/download_models.sh

# 2. start the whole application
docker compose up --build
```

Open **http://localhost:8080**. The sidebar shows `Backend online · 7/7 models loaded` when everything is ready.
API documentation: http://localhost:8000/docs.

Using the app: upload an image (or pick a bundled sample), choose a corruption (none / salt-and-pepper / Gaussian blur /
occlusion) and severity (low / medium / high / custom, optional seed), then run. Choose *None* to restore an image that is
**already corrupted** (unseen images). Each workspace shows the input, the output, the corruption settings, the inference time
and, where relevant, the classifier probabilities or routing weights; results can be downloaded. Task 4 also supports a webcam.

The backend (FastAPI) exposes `GET /api/health`, `POST /api/universal`, `POST /api/hard`, `POST /api/soft`, `POST /api/sketch`
(plus `/api/meta` and `/api/samples`). It validates uploads (type, size, pixel count, decodability), preprocesses to 128×128 and
runs the ONNX models with ONNX Runtime on CPU.

### Without Docker (development)

```bash
pip install -r app/backend/requirements.txt
cd app/backend && MODELS_DIR=../../models PYTHONPATH=../.. uvicorn app.main:app --port 8000
# second terminal
cd app/frontend && npm install && npm run dev        # http://localhost:5173 (proxies /api to :8000)
```

---

## 2. Repository layout

```
src/
  data/       corruption_spec.py (numpy corruptions), corruptions.py (batched GPU corruptions), manifest.py,
              make_manifests.py, pets.py (80/20 split, seed 42), fs2k.py (stratified split, paired augmentation)
  models/     autoencoder.py, classifier.py, soft_moe.py, cgan.py, loading.py
  train/      ae.py, classifier.py, soft_moe.py, cgan.py
  optuna_studies/   ae_study.py, cls_study.py, moe_study.py, cgan_study.py
  eval/       evaluate.py (Tasks 1-3 test evaluation), moe_analysis.py (routing analysis)
  export/     export_onnx.py (export + PyTorch-vs-ONNX verification)
  common/     losses, metrics, MLflow helpers
manifests/    fixed validation / test corruption manifests (committed, small)
app/backend   FastAPI service + Dockerfile       app/frontend   React + Tailwind (Vite) + Dockerfile + nginx.conf
scripts/      run_t1_t2.sh, run_t3.sh, run_t4.sh, download_models.sh
notebooks/    Kaggle notebooks used for training (clone this repo and call the scripts)
design/       Google Stitch brief and design screenshots
tests/        Python tests (pytest);  app/frontend/src/test  frontend tests (vitest)
docker-compose.yml
```

---

## 3. Data

* **Oxford-IIIT Pet** is downloaded by torchvision on first use. The official *trainval* set is split 80 / 20 with seed 42
  (train 2,944 / validation 736); the official *test* set (3,669) is untouched until the final evaluation. Images are converted to RGB
  and resized to 128×128 once and cached.
* **Runtime corruption**: every training sample picks one of {clean, salt-and-pepper (p ∈ [0.02, 0.15]), Gaussian blur (kernel 3/5/7,
  σ ∈ [0.5, 2.5]), 1–3 black rectangles covering 10–35 %} with equal probability, on the GPU, every time it is loaded.
* **Manifests**: `manifests/val.json.gz` and `manifests/test.json.gz` store type, severity, parameters, rectangle coordinates and a seed for
  every entry (regenerate with `python -m src.data.make_manifests`; output is deterministic). Test severities: salt-and-pepper
  0.03 / 0.08 / 0.15, blur (3, 0.7) / (5, 1.5) / (7, 2.5), occlusion ≈ 10 / 20 / 35 % with 1 / 2 / 3 rectangles.
* **FS2K** (1,058 train / 1,046 test pairs, three styles): download from the link in the FS2K repository (Google Drive) and unzip to
  `data/fs2k`. 15 % of the training pairs, stratified by style (seed 42), form the validation set; the official test set is never used for
  training or tuning. Augmentation (flip, small rotation, zoom, shift) is applied with one affine warp to photo and sketch together.

## 4. Training and experiments

Everything below was run on Kaggle GPUs through the notebooks in `notebooks/` (they clone this repository and call the scripts).
Optuna (SQLite storage, median pruner) is used in every task and MLflow (SQLite backend in `outputs/mlflow`) records parameters,
losses, metrics, checkpoints and sample images.

```bash
pip install -r requirements.txt
bash scripts/run_t1_t2.sh      # Task 1 search + training, Task 2 classifier, shared specialist search, 3 specialists
bash scripts/run_t3.sh         # Task 3 (needs the Task 2 outputs)
bash scripts/run_t4.sh         # Task 4 (needs FS2K in data/fs2k)
```

Useful environment variables: `TAG=_v2` (separate output folders), `TRIALS`, `TRIAL_EPOCHS`, `FINAL_EPOCHS`, `GPU`.
View the tracking UI with `mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db`.

After training and evaluation, two small scripts finish the record:

```bash
python scripts/log_results_to_mlflow.py --outputs outputs --models models --mlflow-dir outputs/mlflow   # checkpoints, test evaluation, ONNX verification into MLflow
python scripts/collect_results.py --outputs outputs --models models --out .                              # configs/*.json and results/ (small files only)
```

`configs/` holds the hyper-parameters selected by each Optuna study (with the search budget that was used) and `results/` the Optuna trial
tables, training histories, evaluation summaries, tables and figures and the ONNX verification. Checkpoints and ONNX files are too large for git.

Evaluation on the fixed test manifest (tables per corruption and severity, no-restoration baseline, oracle vs predicted routing,
confusion matrix, example / failure grids with error maps, routing heatmap and expert-activity checks):

```bash
python -m src.eval.evaluate --t1 outputs/t1_v2/best.pt --cls outputs/t2_cls_v2/best.pt \
    --specs outputs/spec_salt_v2/best.pt outputs/spec_blur_v2/best.pt outputs/spec_occlusion_v2/best.pt \
    --moe outputs/t3_v2/best.pt --out outputs/eval_v2
```

## 5. ONNX export

```bash
python -m src.export.export_onnx --t1 ... --cls ... --specs ... --moe ... --gen outputs/t4/best.pt --out models
```

writes the seven `.onnx` files and `models/onnx_verification.json`, comparing ONNX Runtime with PyTorch on several inputs and
batch sizes (tolerance 1e-4; the command fails otherwise). The soft mixture-of-experts is exported as one graph (image → restored image
+ routing weights); for Task 4 only the generator is exported.

Trained models are too large for a normal git commit (one file can exceed 100 MB). They are distributed as release assets / download
links (see `models/README.md`) and fetched with `scripts/download_models.sh`.

## 6. Tests

```bash
pip install -r requirements.txt pytest httpx && python -m pytest tests -q      # Python (data, models, training, Optuna, evaluation, ONNX, backend)
cd app/frontend && npm install && npm test                                      # frontend (vitest)
```

## 7. Notes and limitations

* `docker compose up --build` was run with the real trained ONNX models: the backend reports 7/7 models loaded and all four workspaces return results through the nginx-served frontend.
* FS2K's official test split is style-skewed (619 / 381 / 46 images for styles 1 / 2 / 3), which matters when reading per-style results.
* The assignment was completed under a tight deadline; the exact Optuna budgets and epochs used for each task are stated in the report.
* AI tools were used for research, coding and debugging; see the AI-use appendix of the report.
