# GenAI Assignment 1 — Implementation Plan

Stack: PyTorch, Optuna (SQLite storage), MLflow, ONNX / ONNX Runtime, FastAPI, React + Tailwind, Docker Compose.
Training happens on the NVIDIA DGX Spark (ARM64) over SSH. The web app must also run on a normal x86 laptop.

> The deadline printed in the PDF is March 16, 2024, which is almost certainly a typo. Confirm the real date before scheduling.
> Some formulas in the PDF came out garbled in text extraction (the balance loss, loss-weight symbols). Check them visually in the PDF before implementing.

---

## 0. Key design decisions

| Topic | Decision | Reason |
|---|---|---|
| Framework | PyTorch | Simple ONNX export, easy custom losses |
| SSIM | `pytorch-msssim` (or own implementation) | Differentiable, only needed at training time |
| Tracking | MLflow (local `sqlite`/file backend) | No account needed, works offline on the Spark, screenshots for video/report |
| Optuna storage | SQLite file per task | Resume interrupted studies, pruners work, one study per task |
| Data caching | Pre-resize clean images to 128x128 uint8 tensors, cache to `.pt`/`.npy` | Corruption stays at runtime, I/O cost goes away |
| Corruption code | One shared module (`corruptions.py`) used by training, manifests, evaluation **and** backend | Guarantees the app matches what the models were trained on |
| Backend inference | `onnxruntime` (CPU) | Runs anywhere, 128x128 models are tiny |
| Frontend serving | nginx container serving the built React app and proxying `/api` to FastAPI | One origin, no CORS trouble |
| Models in Docker | Mounted volume `./models`, filled by `scripts/download_models.sh` | Complies with "no large files on GitHub" |

---

## 1. Repository layout

```
GenAI-A1/
├── README.md                  # full run instructions (required)
├── requirements.txt           # training deps
├── docker-compose.yml
├── configs/                   # YAML per task (final selected hyperparameters)
├── data/                      # gitignored; download + cache here
├── manifests/                 # val/test corruption manifests (small JSON, committed)
├── src/
│   ├── common/                # seed, logging, mlflow helpers, metrics (PSNR/SSIM/LPIPS), ssim loss
│   ├── data/
│   │   ├── pets.py            # dataset, 80/20 split (seed 42), runtime corruption
│   │   ├── corruptions.py     # salt-pepper, blur, occlusion (+ deterministic param-driven versions)
│   │   ├── make_manifests.py  # builds val + test manifests
│   │   └── fs2k.py            # FS2K dataset, stratified 15% val split, paired augmentation
│   ├── models/
│   │   ├── autoencoder.py     # Task 1 / specialists
│   │   ├── classifier.py      # Task 2
│   │   ├── soft_moe.py        # Task 3
│   │   └── cgan.py            # Task 4 generator (U-Net + style embed) and PatchGAN
│   ├── train/                 # train_t1.py, train_t2_cls.py, train_t2_spec.py, train_t3.py, train_t4.py
│   ├── optuna_studies/        # one script per study
│   ├── eval/                  # evaluate_t1..t4, figures (error maps, confusion matrix, routing heatmap)
│   └── export/                # export_onnx.py + verify_onnx.py
├── app/
│   ├── backend/               # FastAPI + Dockerfile
│   └── frontend/              # React + Tailwind + Dockerfile (+ nginx.conf)
├── models/                    # gitignored or Git LFS: .onnx files
├── scripts/                   # download_data.sh, download_models.sh, run_all.sh
├── report/                    # IEEEtran LaTeX, figures/
├── design/                    # Google Stitch exports/screenshots (report evidence)
└── docs/                      # assignment PDF, this plan, AI-use log
```

---

## 2. Phases

Order matters: T1 -> T2 -> T3 is a dependency chain. T4 is independent. The app skeleton can start early with fake models.

### Phase 0 — Setup (about 1 day)
- [ ] Confirm permission to use the Spark. Set up Remote-SSH in VS Code.
- [ ] Run the environment check: `nvidia-smi`, Docker + GPU, free disk, CPU count.
- [ ] Choose the training environment: NGC PyTorch container (preferred on ARM64) or a verified aarch64 CUDA wheel.
- [ ] Verify `onnx`, `onnxruntime`, `optuna`, `mlflow`, `pytorch-msssim`, `lpips` install on aarch64.
- [ ] Repo skeleton, `.gitignore` (data, models, mlruns, \*.db), first commit and push.
- [ ] Start `docs/AI_USE_LOG.md` now: tool, task, how it was verified. It is cheaper than reconstructing it at the end.
- [ ] Download datasets (Oxford-IIIT Pet via torchvision, FS2K from the official source). **Check FS2K access now.** The official download links can be Google Drive or Baidu and sometimes break.
- [ ] Start Google Stitch design work in parallel (see Phase 6).

Exit criteria: a toy PyTorch script runs on the Spark GPU inside the chosen environment, and MLflow records a dummy run.

### Phase 1 — Data pipeline for Tasks 1–3 (about 2 days)
1. `pets.py`: use official `trainval` (3,680 images) and `test` (3,669). Convert to RGB, resize to 128x128 and cache. Split trainval 80/20 with seed 42, saved to `splits.json` so every task reuses the identical split.
2. `corruptions.py`, each corruption as a pure function of (image, params):
   - Salt-and-pepper: p ~ U(0.02, 0.15), each corrupted pixel black or white with equal probability.
   - Blur: kernel in {3, 5, 7}, sigma ~ U(0.5, 2.5).
   - Occlusion: 1–3 black rectangles, total area 10–35%, random locations. Write the area-sampling carefully (rectangle sizes must jointly hit the target area, overlaps handled deliberately). Unit-test the achieved coverage.
3. Training dataset: pick one of the four conditions with equal probability on each `__getitem__`, return `(corrupted, clean, label)`. Use `worker_init_fn` seeding so workers don't repeat random draws.
4. Manifests (`make_manifests.py`), written once and committed:
   - Validation manifest: one condition per image, or all four per image (decide once. I suggest all four per image so metrics are stable and comparable across epochs).
   - Test manifest: clean + 3 corruptions x 3 fixed severities. That is 10 entries per image. Store type, severity, blur (k, sigma), rectangle coordinates and a per-entry seed.
   - Test severities: salt-and-pepper 0.03/0.08/0.15, blur (3,0.7)/(5,1.5)/(7,2.5), occlusion about 10/20/35% with 1/2/3 rectangles.
5. Verify with tests: manifest replay gives identical output twice, coverage is within tolerance, no test image leaks into training. Plot a sample grid for the report.

Exit criteria: `make_manifests.py` is deterministic, a visual check of all 10 conditions looks right, and the unit tests pass.

### Phase 2 — Task 1: Universal DAE (about 3 days)
- **Architecture:** conv encoder 128 -> 64 -> 32 -> 16 -> 8 (stride-2 conv + norm + activation, channels growing), flatten to a linear **bottleneck of dimension d** (Optuna), then a mirrored decoder (ConvTranspose or upsample+conv), sigmoid output. No skip connections in the main variant.
- **Skip-connection study (the doc asks for justification if any are used):** compare no-skip vs. one limited skip at the lowest-resolution scale. Report PSNR/SSIM and show that the heavy-skip case degenerates into copying. Keep the no-skip model as the main one unless the evidence says otherwise.
- **Loss:** alpha·L1 + (1−alpha)·(1−SSIM), alpha starting at 0.8.
- **Optuna search space (>= the required):** learning rate (log), batch size, bottleneck dim, base encoder channels, dropout, alpha. Objective: validation combination such as `(1 − SSIM) + L1` or `PSNR + SSIM`, fixed in advance. Use `MedianPruner`. About 25 trials x about 15 epochs. Record the space, completed and pruned trial counts, and the best trial.
- **Final training:** the best config, longer (about 100 epochs), cosine or plateau LR schedule, early stopping, checkpoints and sample images to MLflow.
- **Evaluation on the test manifest:** PSNR, SSIM, L1 per corruption type x low/medium/high severity, plus clean. Figures: 12+ examples (clean / corrupted / restored / absolute error map) and 4 analysed failure cases.
- **Baseline:** the input itself (no restoration) for context.

### Phase 3 — Task 2: Classifier and hard routing (about 3 days)
- **Classifier:** small CNN with 4 outputs. Batches are explicitly balanced (equal counts per class, enforced by the sampler, not just by luck). Cross-entropy. Optuna: lr, batch size, channel config, dropout, weight decay. Evaluate: accuracy, macro P/R/F1, per-class metrics, normalized confusion matrix (also by severity, since low severity is where confusion will hide).
- **Specialists:** one shared Optuna search on one architecture (lr, bottleneck, channels, batch size, L1/SSIM weight), then train **three independent** models. Each trains only on its corruption, with the same clean targets.
- **Hard-routed inference:**
  - clean -> identity bypass (no model call);
  - otherwise the argmax class selects the specialist.
- **Two evaluation modes** on the test manifest:
  - oracle routing (true label from the manifest);
  - predicted routing (classifier output).
- **Analysis:** the gap between oracle and predicted, a list of misrouting cases (e.g. low-severity blur called clean, small occlusion vs. salt noise) and the resulting PSNR drop. Compare also with Task 1.

### Phase 4 — Task 3: Soft mixture-of-experts (about 3 days)
- **Model:** gate (initialized from the Task 2 classifier weights) -> logits -> `softmax(z / T)` over [identity, salt, blur, occlusion]. The three experts are initialised from the Task 2 specialists. Output = sum of weights x branch outputs.
- **Training stages:**
  1. Warm-up: freeze experts, train only the gate for a few epochs.
  2. Joint fine-tune: unfreeze everything at a smaller LR.
- **Loss:** a·L1 + b·(1−SSIM) + g·CE(gate, true label) + d·L_balance. Starting weights 0.8 / 0.2 / 0.1 / 0.01 (verify against the PDF). L_balance uses the batch-mean routing weight per branch pushed toward uniform (1/K). **Check the exact PDF formula**; if the Optuna-found collapse behaviour suggests it, cite and justify an alternative (e.g. the Switch-Transformer load-balancing loss or an entropy regulariser).
- **Optuna:** joint LR, temperature T, CE weight, balance weight, L1/SSIM weighting. Prune on routing collapse (e.g. max average branch weight above 0.9 for several epochs, or poor validation).
- **Gate analysis (required):**
  - mean weights per true corruption x severity (table + heatmap);
  - examples with one dominant expert and examples with spread weights;
  - checks for a dead expert (mean weight near 0 everywhere) and an expert dominating unrelated inputs;
  - compare with hard routing and with Task 1 on the same test manifest.
- **Extra (cheap, valuable):** a few test images with *combined* corruptions (e.g. blur + salt) to show where soft routing beats hard routing. This makes the task's motivation visible in the report and video.

### Phase 5 — Task 4: Face-to-sketch cGAN (about 3–4 days, can overlap Phases 2–4)
- **Data:** FS2K official train/test. 15% of train held out for validation, **stratified by style**, seed 42. 128x128. Verify photo/sketch pairing and the style labels by eye on a sample grid. Paired augmentation (flip, small crop/resize) applies the **same random parameters to both images**. Unit-test this.
- **Generator:** U-Net (6–7 down/up levels at 128x128). `nn.Embedding(3, e)` for style. The embedding is injected into the network itself, e.g. broadcast and concatenated with the input and/or FiLM / conditional norm at the bottleneck.
- **Discriminator:** PatchGAN on concat(photo, sketch, style-embedding map) so the style genuinely conditions it.
- **Loss:** BCE-with-logits adversarial + lambda·L1 (lambda starts at 100). D loss is logged as real and fake parts separately, and G loss as adversarial and L1 parts.
- **Optuna:** G and D learning rates, batch size, base channels, dropout, embedding dim, lambda. Short trials (10–15 epochs), then retrain the best config for the full schedule (about 150–200 epochs).
- **Logging to MLflow:** the five loss curves above, validation L1/SSIM/PSNR (plus LPIPS or FID if time permits), and generated samples for the **same fixed validation photos** every N epochs.
- **Check:** the same photo with Style 1/2/3 should produce visibly different sketches. If it doesn't, the conditioning is too weak (a common failure; fix by stronger injection).
- **Test evaluation:** official test set, once, at the end.

### Phase 6 — ONNX export and verification (about 2 days, start per model as each finishes)
- Export with `torch.onnx.export`, opset 17, dynamic batch axis, fixed 128x128.
- Exports needed:
  - T1 autoencoder;
  - T2 classifier + 3 specialists (4 files);
  - T3 whole soft-MoE pipeline as **one graph** with outputs (restored image, 4 gate weights, optionally the expert outputs);
  - T4 generator, inputs (photo, style index).
- `verify_onnx.py` compares the ONNX Runtime and PyTorch outputs on real inputs and records the max absolute difference (target below about 1e-4). Save the numbers for the report table.
- Watch for ops that export poorly (some resize/interpolation modes, `pytorch-msssim` is not exported because it is training-only).
- Upload the `.onnx` files to a hosting link (GitHub release / Drive / Hugging Face) or Git LFS, and write `download_models.sh`.

### Phase 7 — Application (about 5 days, begins in parallel)

**7a. Google Stitch design first**
- Design 4 workspaces plus a landing/nav: Universal Restoration, Hard-Routed Restoration, Soft MoE Restoration, Face-to-Sketch Generator.
- Save screenshots and exports to `design/` with timestamps. The report needs evidence of the **original** design.

**7b. Backend (FastAPI)**
- Endpoints:
  - `GET /health`
  - `POST /api/universal`
  - `POST /api/hard`
  - `POST /api/soft`
  - `POST /api/sketch`
  - a corruption-preview endpoint, or a corruption parameter on each restoration call.
- Validation: allowed content types, max file size, decodable with PIL, sane dimensions. Return clean 4xx errors.
- Preprocess: RGB, resize to 128x128. Apply the selected corruption using the shared `corruptions.py` with user-chosen type and severity (or accept an already-corrupted upload).
- Load the ONNX sessions once at startup. Return JSON with base64 images, the corruption settings, class probabilities / gate weights, the selected expert, and the inference time in ms.
- Run models at 128x128 and upscale only for display, and say so in the UI.

**7c. Frontend (React + Tailwind, Vite)**
- Four workspace pages with upload, sample-image picker, corruption type/severity controls.
- Side-by-side input/output panels, probability bars, expert-contribution visualisation, timing.
- Face-to-Sketch: webcam capture (`getUserMedia`), style selector, side-by-side view, download button.
- Result download for the restoration workspaces too (the demo must show downloading).
- Responsive layout. Loading and error states.

**7d. Docker**
- Backend Dockerfile (python-slim, `onnxruntime`, uvicorn). Frontend multi-stage Dockerfile (node build -> nginx).
- `docker-compose.yml`: `frontend` + `backend`, `./models` volume, health checks. One documented command: `docker compose up --build`.
- Test from a **fresh clone** on a different machine, preferably x86, with no VS Code.

### Phase 8 — Report, repo polish and video (about 4–5 days)
- **Report (LaTeX, IEEEtran, Overleaf or local):** introduction, related work, datasets, per-task methodology (architecture diagram, losses, training, Optuna design and results, results tables, visual grids, error maps, failure cases), routing heatmaps, application architecture with Stitch evidence and screenshots, limitations, conclusion, GitHub and YouTube links, **AI-use appendix**. Interpret every figure/table in text. Write figure-generation scripts under `src/eval/` so figures are reproducible.
- **README:** setup, data download, how to train/evaluate/export, how to download models, one-command run, repo map.
- **Video (5–7 min):** a script and a rehearsal first. Cover startup via Docker Compose, image upload, runtime corruption, universal restoration, hard routing with probabilities, soft expert weights, face-to-sketch with a webcam or upload, result download and the MLflow records. Upload to YouTube and put only the link in the report.
- **Final checks:** requirements checklist (section 4), fresh-clone test, test on unseen images, a rehearsal of "explain/modify any component".

---

## 3. Schedule at a glance (effort, not dates)

| Phase | Effort | Depends on |
|---|---|---|
| 0 Setup | 1 d | |
| 1 Data pipeline | 2 d | 0 |
| 2 Task 1 | 3 d | 1 |
| 3 Task 2 | 3 d | 2 (reuses blocks) |
| 4 Task 3 | 3 d | 3 |
| 5 Task 4 | 3–4 d | 0 (parallel with 2–4) |
| 6 ONNX | 2 d | each model |
| 7 App | 5 d | Stitch design; real models arrive from 6 |
| 8 Report + video | 4–5 d | everything |
| **Total** | **about 25 working days** with overlap, roughly 4–5 calendar weeks | |

Suggested overlap: Task 4 runs on the Spark while Task 1–3 results are analysed; the app skeleton is built against dummy ONNX models while training finishes; the report skeleton and figure scripts are written while experiments run.

---

## 4. Requirements checklist (traceability)

- [ ] Split 80/20 seed 42, official test untouched until final evaluation, RGB 128x128
- [ ] Runtime corruption, equal probability of the four conditions, nothing saved to disk for training
- [ ] Deterministic val/test manifests with all required fields
- [ ] T1: bottleneck AE, no unrestricted skips, L1+SSIM, Optuna (>= required params), per-corruption and per-severity results, >= 12 examples and 4 failures, error maps
- [ ] T2: balanced classifier + Optuna + full metrics + normalized confusion matrix; 3 independent specialists; identity bypass; oracle and predicted modes
- [ ] T3: gate and experts initialised from T2; warm-up then joint fine-tune; 4-term loss; Optuna with pruning; gate analysis and heatmap; dead/dominant expert check
- [ ] T4: FS2K official split, 15% stratified val seed 42; U-Net + PatchGAN; learned style embedding in G and D; BCE + lambda·L1; Optuna; paired augmentation; separate loss logging; fixed validation samples over time
- [ ] Optuna in all four tasks; MLflow (or W&B) logging of params, losses, metrics, checkpoints, images
- [ ] ONNX for all inference models with PyTorch-vs-ONNX consistency numbers
- [ ] Stitch design evidence; React + Tailwind frontend; FastAPI backend with all endpoints; four named workspaces
- [ ] Dockerfiles + Compose; fresh-clone one-command start
- [ ] GitHub repo with all required contents, no big datasets or models (links or LFS)
- [ ] IEEE LaTeX report including GitHub link, YouTube link, AI-use appendix
- [ ] 5–7 min YouTube demo showing every required item
- [ ] Submitted through Google Classroom before the deadline

---

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Company policy on using the Spark | Confirm early. Fallback: Kaggle (about 30 GPU-h/week) |
| ARM64 package issues (onnxruntime-gpu, lpips, etc.) | Use the NGC container; run ONNX verification on CPU; test installs in Phase 0 |
| FS2K download/links unavailable | Check in Phase 0; contact classmates or the faculty if broken |
| Gate collapse in Task 3 | Warm-up stage, balance loss, temperature tuning, pruning on collapse |
| Weak style conditioning in the cGAN | Inject the embedding in several places; visual check of the same photo across styles |
| ONNX export of the full MoE graph | Write the MoE as plain tensor ops, export early with random weights |
| Docker works on Spark/ARM but not x86 | Test the compose build on an x86 machine; keep the app images separate from the training environment |
| Report left to the end | Figure scripts and the LaTeX skeleton from Phase 2 onward; keep the AI-use log current |
| Not understanding generated code (viva) | Read and run every module as it is written; keep short notes on design choices as you go |

---

## 6. Open questions

1. What is the actual deadline?
2. Has use of the Spark been approved?
3. How should the trained models be hosted: Git LFS, GitHub release or Google Drive?
4. Is there an x86 machine available for the fresh-clone test?
5. Validation manifest: one random condition per image, or all four per image?
