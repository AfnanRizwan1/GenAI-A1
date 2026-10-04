# Project guide: what the assignment asked, what was built, and why

This guide explains **Generative AI Assignment 1** from zero. You do not need to have seen the assignment PDF or the code. Read it top to bottom once; later use the table of contents to jump to a part. Every number in it comes from the real results in this repository.

**Contents**

1. [The assignment in plain words](#1-the-assignment-in-plain-words)
2. [Ideas you need first (a mini course)](#2-ideas-you-need-first-a-mini-course)
3. [The data](#3-the-data)
4. [Task 1: one universal restoration model](#4-task-1-one-universal-restoration-model)
5. [Task 2: classify, then route to a specialist](#5-task-2-classify-then-route-to-a-specialist)
6. [Task 3: soft mixture of experts](#6-task-3-soft-mixture-of-experts)
7. [Task 4: face-to-sketch with a style control](#7-task-4-face-to-sketch-with-a-style-control)
8. [Hyper-parameter search with Optuna](#8-hyper-parameter-search-with-optuna)
9. [Experiment tracking with MLflow](#9-experiment-tracking-with-mlflow)
10. [ONNX export](#10-onnx-export)
11. [The web application](#11-the-web-application)
12. [Docker and one-command deployment](#12-docker-and-one-command-deployment)
13. [Testing and quality checks](#13-testing-and-quality-checks)
14. [The report, the repository and the video](#14-the-report-the-repository-and-the-video)
15. [How the work actually went (and what went wrong)](#15-how-the-work-actually-went-and-what-went-wrong)
16. [Honest limitations](#16-honest-limitations)
17. [Questions an evaluator may ask, with answers](#17-questions-an-evaluator-may-ask-with-answers)
18. [Glossary](#18-glossary)

---

## 1. The assignment in plain words

### The story

Imagine a photo-repair shop. Customers bring damaged photos: some covered in speckles (like TV static), some blurry, some with black patches covering part of the picture. The assignment asks you to build the repair machinery **and** the shop counter (a website) where a customer uploads a photo and sees it repaired.

There are four "machines" (called **tasks**):

| Task | Idea in one sentence | Analogy |
|---|---|---|
| 1. Universal restoration | One neural network repairs every kind of damage. | One mechanic who fixes everything. |
| 2. Hard-routed restoration | A classifier first recognises the damage type, then sends the photo to one specialist network. | A receptionist who sends you to exactly one specialist. |
| 3. Soft mixture of experts | A "gate" gives each specialist a share (a percentage), and the final photo is a weighted blend. Gate and specialists are trained **together**. | A committee where every member votes with a different weight. |
| 4. Face-to-sketch | A network turns a face photo into a pencil sketch, in one of three art styles that the user picks. | An artist who can draw in three styles on request. |

### The three kinds of damage (called "corruptions")

- **Salt-and-pepper noise:** random pixels become pure black or pure white.
- **Gaussian blur:** the image is smoothed so details disappear.
- **Occlusion:** one to three black rectangles cover parts of the picture.

The damage is applied **at run time**: you start with a clean photo and the program damages it on the fly, so the correct answer (the clean photo) is always known and the repair can be measured.

### Everything that had to be delivered

1. Four trained models (Tasks 1 to 4), trained with PyTorch.
2. **Optuna** (an automatic "try many settings and keep the best" tool) used in **all four** tasks.
3. **MLflow** (a lab notebook for experiments) recording settings, losses, metrics, saved models and sample images.
4. A **website** with four workspaces (React + Tailwind in the browser, FastAPI on the server) whose look was first designed in **Google Stitch** (an AI interface-design tool).
5. The models exported to **ONNX** (a portable model file format) and checked to give the same outputs as PyTorch.
6. **Docker Compose**: the whole app starts with one command.
7. A **GitHub repository** with all code, configs, scripts, Dockerfiles and a README. Large files (datasets, big model files) are not uploaded directly; a download link is given instead.
8. A **technical report** in IEEE research-paper format written in LaTeX, with an **AI-use appendix**.
9. A **5 to 7 minute YouTube demo video**.
10. It is an **individual** assignment, due on Google Classroom (deadline: 11:59 PM, Sunday 4 October 2026).

### How you will probably be judged

The assignment says evaluators may ask you to explain or modify any part and to run the app on unseen images. So understanding matters as much as the code. Section 17 prepares you for that.

---

## 2. Ideas you need first (a mini course)

### 2.1 Images as numbers
A colour image is a grid of pixels; each pixel has three numbers (red, green, blue). In this project every image is resized to **128 x 128** pixels, with values between 0 and 1. A neural network just takes this block of numbers in and gives a block of numbers out.

### 2.2 Neural network, training, loss
A neural network is a big formula with adjustable numbers (called **weights**). **Training** means: show it an example, compare its answer with the correct answer, measure the mistake with a **loss** number, and nudge the weights so the mistake gets smaller. Repeat thousands of times. The loop of "see all training images once" is called an **epoch**.

### 2.3 Autoencoder and "bottleneck"
An **autoencoder** has two halves:
- the **encoder** squeezes the image into a small summary (the **code**),
- the **decoder** rebuilds an image from that summary.

The narrow middle is the **bottleneck**. Because the code is small, the network cannot memorise every pixel; it must keep the important structure. That is why it can remove noise: noise is not important structure, so it does not fit through the bottleneck.

A **denoising autoencoder** gets a *damaged* image as input but is graded against the *clean* image. It learns "undo the damage".

**Strict bottleneck, no skip connections:** many image networks (like U-Net) pass details around the bottleneck with "skip connections". The assignment asks for a compressed bottleneck, so in Tasks 1 to 3 **all information has to pass through the code**. This is a design rule here, and it explains many results (Section 4.5).

### 2.4 Measuring quality
- **L1 error:** the average absolute difference between two images' pixels. Lower is better.
- **PSNR (in dB):** a score derived from the squared error; higher is better. About 20 dB is clearly noisy, 30 dB is quite good. Identical images would be infinite, so this project caps it at **60 dB**.
- **SSIM (0 to 1):** compares structure (edges, contrast) the way humans see it. 1 means identical.

### 2.5 Classifier and confusion matrix
A **classifier** looks at an image and says which of several classes it belongs to (here: clean, salt-and-pepper, blur, occlusion). A **confusion matrix** is a table: rows = truth, columns = prediction. A perfect classifier has numbers only on the diagonal. **Accuracy** = fraction right. **Macro F1** = an average over classes of a score combining precision and recall (so a rare class counts as much as a common one).

### 2.6 Mixture of experts (MoE)
Several small networks (**experts**) each specialise in something. A **gate** network looks at the input and gives each expert a weight; the weights sum to 1 (this is called **softmax**). The output is the weighted blend of the experts' outputs. **Hard routing** picks one expert (weight 1, others 0); **soft routing** blends.

### 2.7 GAN, conditional GAN, U-Net, PatchGAN, FiLM
- A **GAN** (generative adversarial network) has a **generator** (the artist) and a **discriminator** (the critic). The critic learns to tell real from fake; the artist learns to fool the critic.
- A **conditional GAN (cGAN)** also receives a condition (here: the face photo and the chosen style).
- **U-Net:** an encoder-decoder with skip connections, good at image-to-image translation (used for the generator in Task 4, where the output must line up with the input photo).
- **PatchGAN:** a critic that judges small patches of the image instead of the whole image, which encourages sharp local texture.
- **FiLM (feature-wise modulation):** a way to feed a condition into a network: the style embedding produces a scale and a shift that adjust the network's internal features. The style choice therefore steers every decoder block.
- **Style embedding:** each style (1, 2, 3) has a learned vector of numbers, like a "style fingerprint" the network learns during training.

### 2.8 Optuna, MLflow, ONNX
- **Optuna** tries many hyper-parameter combinations (learning rate, layer sizes, ...) automatically, using a smart sampler (TPE) and a **pruner** that stops hopeless trials early to save time.
- **MLflow** records every run: settings, loss curves, results, checkpoints (saved models) and pictures.
- **ONNX** is a standard model file that runs without PyTorch (here with **ONNX Runtime** on the CPU), which makes the app small and portable.

### 2.9 Web and deployment words
- **FastAPI:** a Python web server that offers URLs like `/api/universal`.
- **React:** the JavaScript library that builds the web page; **Tailwind CSS** styles it.
- **Docker:** packages a program with everything it needs into a **container**. **Docker Compose** starts several containers (backend + frontend) with one command.
- **Google Stitch:** an AI tool that generates interface designs from a description.

---

## 3. The data

### 3.1 Tasks 1 to 3: Oxford-IIIT Pet dataset
About 7,400 photos of cats and dogs. The breed labels are not used. The official split is respected:
- **trainval** (3,680 images) is split 80/20 with a fixed random seed (42) into **2,944 for training** and **736 for validation**;
- the official **test** set (**3,669 images**) is not touched until the final evaluation.

Why three sets? *Training* teaches the model, *validation* is used to choose settings and detect overfitting, and *test* is a final honest exam that was never used to make decisions.

### 3.2 How the damage is made
During training each image gets a random corruption each time it is loaded, on the GPU for speed:

| Type | Training range |
|---|---|
| Salt-and-pepper | 2% to 15% of pixels flipped |
| Blur | kernel size 3, 5 or 7, sigma 0.5 to 2.5 |
| Occlusion | 1 to 3 black rectangles covering 10% to 35% |
| Clean | left untouched (so the model also sees undamaged images) |

### 3.3 Fixed validation and test "manifests"
For fair comparison, the **same** damage must be applied to every model. So two files, `manifests/val.json.gz` (2,944 entries) and `manifests/test.json.gz` (**36,690 entries**), list for every image exactly which corruption, which severity and which random seed. Test severities: salt-and-pepper 3%, 8%, 15%; blur (kernel 3, sigma 0.7), (5, 1.5), (7, 2.5); occlusion about 10%, 20%, 35% with 1, 2, 3 rectangles. Every test image appears once clean and nine times damaged (3 types x 3 levels): 3,669 x 10 = 36,690.

A numpy-only copy of the corruption code (`src/data/corruption_spec.py`) is shared by training, evaluation and the web backend, so the website damages images with **exactly** the same code the models were trained with.

### 3.4 Task 4: FS2K
A dataset of face photos paired with hand-drawn sketches in three styles. Official split: **1,058 training pairs, 1,046 test pairs**. 15% of the training pairs (stratified by style, seed 42) become validation: **899 train / 159 validation**. The official test set has a skewed style mix: style 1 = 619, style 2 = 381, style 3 = only 46 images. The test set is never used to tune anything.

**Augmentation for Task 4:** flips, small rotations, zoom and shifts. The photo and its sketch must move together, so one single geometric warp is applied to both stacked as a 6-channel image.

---

## 4. Task 1: one universal restoration model

### 4.1 What was asked
One denoising autoencoder with a compressed bottleneck that restores clean, salt-and-pepper, blurred and occluded images, trained with a combined loss, with Optuna tuning.

### 4.2 What was built
`src/models/autoencoder.py` (class `ConvAE`): a convolutional encoder (stride-2 convolutions, each halving the image: 128, 64, 32, 16, 8), a **convolutional bottleneck** (code of size 8 x 8 x c_z), and a mirrored decoder. With c_z = 32 the code is only 2,048 numbers, against 49,152 input values (a 24x compression). **No skip connections** (a test checks that decoding only the code reproduces the output).

### 4.3 The loss
`loss = alpha * L1 + (1 - alpha) * (1 - SSIM)`.
L1 keeps the colours and brightness right; (1 - SSIM) keeps edges and structure right; `alpha` balances them (tuned by Optuna). Optimiser: AdamW with a one-cycle learning-rate schedule and mixed precision (faster on GPU).

### 4.4 Optuna search and final training
30 trials of 10 epochs each (13 completed, 17 pruned). Search space included bottleneck type (linear vs convolutional), code size, base width, learning rate, batch size, dropout and alpha. Best trial (#15): **convolutional bottleneck, c_z = 32, base channels 48, learning rate 7.7e-4, batch size 32, dropout 0.10, alpha = 0.51**. The final model was then trained for **150 epochs** (validation PSNR about 24.6 dB, SSIM 0.79).

### 4.5 Results on the 36,690-entry test manifest (mean over all entries)

| | PSNR (dB) | SSIM |
|---|---|---|
| No restoration (damaged input) | 23.3 | 0.672 |
| **Task 1 universal AE** | **24.0** | **0.776** |

By damage type it clearly repairs **salt-and-pepper** (about 16 dB to 25 dB) and **occlusion** (about 13.6 dB to 21.5 dB). It does **not** beat the input on **mild blur** or **clean** images.

### 4.6 Why (the honest explanation)
Because the bottleneck is strict, even a perfectly clean image comes back at only about **25.3 dB** (the model cannot reproduce every pixel through the small code). So whenever the damaged input is already better than ~25 dB (clean images, low blur), "restoring" makes it worse. The failure figure in the report also shows colour drift (a pink background coming back beige) and a brown glow where a black rectangle sat on an already-black background. This is a property of the required design, not a bug.

---

## 5. Task 2: classify, then route to a specialist

### 5.1 What was asked
A classifier that recognises the corruption type, then **three specialist autoencoders** (one per damage type); each image goes to exactly one specialist ("hard routing"). Optuna in both parts.

### 5.2 What was built
- **Classifier** (`src/models/classifier.py`): a small convolutional network with four outputs (clean, salt, blur, occlusion). Trained with cross-entropy on class-balanced batches. Optuna: 30 trials (6 completed, 24 pruned).
- **Specialists:** the same autoencoder architecture, searched once by Optuna (30 trials, shared search so the three are comparable) and then **trained three separate times**, each on only its own damage type, 150 epochs.
- **Identity bypass for clean images:** if the classifier says "clean", the image is returned untouched. Otherwise the matching specialist runs.

### 5.3 Results
**Classifier on the test manifest:** accuracy **99.79%**, macro F1 **0.9965**. All salt-and-pepper and blur entries were classified correctly at every severity; the only weak spot is low-severity occlusion (98.3%).

**Specialists (oracle routing = always the correct specialist) vs the universal model:** the specialists are better in **all nine damaged conditions**, by 0.8 to 1.6 dB, because each has an easier job.

**Oracle vs predicted routing:** practically identical (within 0.01 dB in every damaged condition). The only visible cost of classifier mistakes is on clean images (60.0 dB with perfect routing, 59.85 dB with the real classifier). The misrouted examples are all clean photos with wide **black borders**, which the classifier mistakes for occlusion. This is shown in a figure in the report.

**Overall:** hard routing reaches **28.6 dB PSNR / 0.811 SSIM**, better than the universal model (24.0 / 0.776).

---

## 6. Task 3: soft mixture of experts

### 6.1 What was asked
A gate plus experts trained **jointly**, producing a weighted blend; then analyse the gate's behaviour (which expert handles which damage, any dead or dominating expert).

### 6.2 What was built (`src/models/soft_moe.py`)
Four branches: an **identity branch** (the input unchanged) and **three experts**. The gate (a classifier) outputs a score per branch; softmax with a temperature T turns scores into weights g. The output is:

`restored = g0 * input + g1 * salt_expert(input) + g2 * blur_expert(input) + g3 * occlusion_expert(input)`

### 6.3 How it was trained
1. **Smart start:** the gate begins from the trained Task 2 classifier and the experts from the three trained specialists (so training does not start from scratch).
2. **Warm-up** (3 epochs): experts frozen, only the gate adapts, so the experts are not damaged by a gate that is still bad.
3. **Joint fine-tuning** (to 40 epochs): everything is trained together.
4. **Loss:** `a * L1 + (1 - a) * (1 - SSIM)` (image quality) `+ gamma * CE` (a small push for the gate to recognise the damage type) `+ delta * balance` (a penalty if the average usage of the four branches is far from equal, to avoid "one expert does everything").
5. **Collapse guard:** a trial is pruned if one branch gets an average weight above 0.9.

Optuna searched learning rate, temperature, gamma, delta and a: 20 trials, all completed. Best: lr 2.0e-4, T = 2.18, gamma = 0.012, delta = 0.050, a = 0.53.

### 6.4 Routing analysis (the interesting part)
- Salt-and-pepper goes to the salt expert at every severity (weight 0.90 to 0.99).
- Blur goes to the blur expert and the weight grows with severity (0.38, 0.82, 0.95).
- Occlusion goes to the occlusion expert, also growing with severity (0.40, 0.79, 0.96).
- Clean images mostly use the identity branch (0.94).
- Mild blur/occlusion give much weight to the identity branch (about 0.6): sensible, because little needs repairing.
- **No expert is inactive** (mean weights 0.22 to 0.29) and **none dominates** unrelated inputs.
- The gate's top choice matches the true damage type for 83.3% of test entries (lower than the classifier because mild cases deliberately go to "identity").

### 6.5 Soft vs hard
Soft: **27.98 dB / 0.837 SSIM / L1 0.0323**. Hard (predicted): 28.61 dB / 0.811 / 0.0343. Soft has better SSIM and L1 and is much better on low blur (30.4 vs 26.5 dB). Hard is better on PSNR for clean images and low occlusion. Neither wins everywhere, which is a legitimate finding.

---

## 7. Task 4: face-to-sketch with a style control

### 7.1 What was asked
A conditional GAN that turns a face photo into a sketch in a **user-chosen style**, with the style implemented as an embedding that is part of the model, plus Optuna, evaluation and a web workspace (including webcam).

### 7.2 What was built (`src/models/cgan.py`)
- **Generator:** a 6-level U-Net. The style number becomes a learned **embedding**. It is used in two ways: concatenated to the input as extra channels, and injected through **FiLM** in every decoder block.
- **Discriminator:** a PatchGAN that sees the photo, the (real or generated) sketch and the style embedding map, and judges whether each patch is real.
- **Loss:** adversarial loss (BCE) + lambda x L1 (pix2pix-style): the adversarial term makes the strokes look real, the L1 term keeps the sketch aligned with the real one.
- Optuna: 20 trials of 15 epochs (14 completed, 6 pruned); final training 150 epochs.

### 7.3 Results (1,046 test pairs)
| | L1 | SSIM | PSNR |
|---|---|---|---|
| Generator | 0.106 | 0.481 | 15.6 dB |
| Baseline: just the grayscale photo | 0.410 | 0.268 | 6.8 dB |

By style: style 1 = 17.4 dB, style 2 = 12.4 dB (heavy dark strokes are hard to match exactly), style 3 = 18.7 dB (only 46 images, so uncertain).

**Does the style control work?** A **style-conditioning test** feeds every photo each of the three styles and measures the error against each style's true sketch. In every row of the table the lowest error is on the diagonal (the right style gives the closest result). Visually: style 1 = thin light lines, style 2 = heavy dark strokes with shading, style 3 = in between.

**Weakness:** hair comes out as smooth masses or short scribbles, with less contrast than the artists' strokes (an L1-dominated loss and only 899 training pairs).

---

## 8. Hyper-parameter search with Optuna

**What it is for:** instead of guessing learning rates and layer sizes, Optuna runs many short trials, learns which regions work, and **prunes** (stops) trials that are clearly worse than the median.

**Settings used everywhere:** TPE sampler (seed 42 so it is repeatable), median pruner, results stored in a SQLite file per study. The assignment requires the number of completed trials, the best trial and the final configuration to be reported; the report does this in a table.

| Study | Trials | Completed | Pruned | Best value |
|---|---|---|---|---|
| Task 1 autoencoder | 30 | 13 | 17 | 0.4143 |
| Task 2 classifier | 30 | 6 | 24 | 0.9915 (macro F1) |
| Task 2 specialists (shared) | 30 | 10 | 20 | 0.4205 |
| Task 3 soft MoE | 20 | 20 | 0 | 0.1661 |
| Task 4 cGAN | 20 | 14 | 6 | 0.6099 |

The selected hyper-parameters are saved in `configs/*.json`; the trial tables are in `results/`. Some best values sit at the edge of their search range (stated in the report as a limitation: a wider search might find better values).

---

## 9. Experiment tracking with MLflow

**Purpose:** a permanent, browsable record of what was run and what came out. Three MLflow stores exist because training ran in three separate Kaggle sessions:

| Folder | Contains | Command to open it |
|---|---|---|
| `outputs/mlflow_t12` | Tasks 1-2: 93 Optuna trials, training runs | `mlflow ui --backend-store-uri sqlite:///outputs/mlflow_t12/mlflow.db` |
| `outputs/mlflow` | Task 3, plus the **final record of every model**: parameters, final metrics, checkpoint, Optuna summary, test evaluation, ONNX verification | `mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db` |
| `outputs/mlflow_t4` | Task 4: 21 Optuna runs, GAN training | `mlflow ui --backend-store-uri sqlite:///outputs/mlflow_t4/mlflow.db` |

Then open http://localhost:5000. (Run one at a time. Some image previews may look broken because the images were logged on Kaggle with Kaggle paths; the numbers and curves are fine.)

---

## 10. ONNX export

**Purpose:** the website must not need PyTorch. Each model is converted to an `.onnx` file (opset 17, dynamic batch size) and **verified**: the same inputs are fed to PyTorch and to ONNX Runtime and the largest difference is measured; the command fails if it exceeds 1e-4.

| File | Task | Size | Max difference vs PyTorch |
|---|---|---|---|
| universal_restoration.onnx | 1 | 21.9 MB | 2.4e-07 |
| classifier.onnx (with softmax) | 2 | 1.2 MB | 1.2e-07 |
| specialist_salt / blur / occlusion.onnx | 2 | 21.9 MB each | up to 4.2e-07 |
| soft_moe.onnx (image + 4 weights in one graph) | 3 | 67.0 MB | 1.8e-07 |
| sketch_generator.onnx (generator only) | 4 | 65.9 MB | 4.2e-06 |

All are far below the 1e-4 tolerance. Only the **generator** is exported for Task 4 (the discriminator is only needed for training).

---

## 11. The web application

### 11.1 Design with Google Stitch
A design brief and a written design system (colours, fonts, spacing) were given to Stitch, which produced mock-ups for the four workspaces. Screenshots of the first (v1) and improved (v2) designs are in the report as evidence. The React app was then built to match, with fixes where Stitch invented content.

### 11.2 Frontend (React + Tailwind, `app/frontend`)
One page with a sidebar and four workspaces:
1. **Universal Restoration** (Task 1)
2. **Hard-Routed Restoration** (Task 2): shows the four classifier probabilities, predicted class, chosen expert and timing.
3. **Soft Mixture-of-Experts Restoration** (Task 3): shows the four weights, a stacked bar, the dominant expert and routing entropy.
4. **Face-to-Sketch Generator** (Task 4): upload or **webcam**, choose style 1, 2 or 3.

Common features: upload an image or pick a bundled sample; choose corruption (none / salt-and-pepper / blur / occlusion) and severity (low / medium / high / custom) with an optional seed (same seed = same damage); side-by-side view, before/after slider and difference map; PSNR/SSIM cards with the gain; inference time; **download** the result; **copy the metrics as JSON**; dark mode. Choosing "None" restores an **already damaged** photo you uploaded (unseen images).

### 11.3 Backend (FastAPI, `app/backend`)
| URL | Purpose |
|---|---|
| `GET /api/health` | which of the 7 models are loaded |
| `POST /api/universal` | Task 1 |
| `POST /api/hard` | Task 2 |
| `POST /api/soft` | Task 3 |
| `POST /api/sketch` | Task 4 |
| `GET /api/samples`, `/api/meta` | sample pictures, app info |

It validates every upload (type, size, number of pixels, readability), resizes to 128 x 128, applies the chosen corruption with the shared corruption code, runs ONNX Runtime on the CPU, and returns images (as base64) plus metrics.

---

## 12. Docker and one-command deployment

**Purpose:** the evaluator must be able to clone the repository, get the models and open the app **without VS Code or running scripts by hand**.

- `app/backend/Dockerfile`: Python + FastAPI + ONNX Runtime (no PyTorch, so the image stays small).
- `app/frontend/Dockerfile`: builds the React app, serves it with **nginx**, which also forwards `/api` calls to the backend.
- `docker-compose.yml`: starts both; the models folder is mounted read-only into the backend. The backend must report healthy before the frontend starts.

**The evaluator's steps:**
```bash
git clone https://github.com/AfnanRizwan1/GenAI-A1.git && cd GenAI-A1
bash scripts/download_models.sh        # fetches the 7 .onnx files into ./models
docker compose up --build
# open http://localhost:8080  (API docs at http://localhost:8000/docs)
```
The sidebar should say "Backend online - 7 models loaded".

---

## 13. Testing and quality checks

**Automated tests: 69 Python (pytest) and 33 frontend (vitest).** Examples of what they check:
- the GPU blur matches OpenCV's blur;
- the autoencoder has no skip path (decoding the code reproduces the output);
- experts are frozen during warm-up;
- augmentation transforms photo and sketch identically;
- ONNX output equals PyTorch output;
- the backend's SSIM equals the definition used in training;
- upload validation rejects bad files; the UI shows the right result panels.
A "mutation check" confirmed the frontend tests fail if the code is deliberately broken (so the tests are not empty).

**End-to-end checks:** the Docker stack was run with the real models and every endpoint was called; the interface was driven in a real browser (Chrome automation) and screenshots were examined, which found and fixed layout bugs.

---

## 14. The report, the repository and the video

### Report (`report/main.tex`, IEEEtran, 16 pages)
Covers: introduction, related work, data pipeline, each of the four tasks (architecture, loss, training, Optuna, results with tables and figures, failure cases), Optuna summary, application and deployment (Stitch evidence, screenshots, ONNX table), limitations, conclusion, an AI-use appendix and a reproduction appendix. Tables and figures are **generated by a script** (`scripts/make_report_assets.py`) from the real result files, so numbers cannot be mistyped. Build with `pdflatex`/`bibtex` or upload the `report/` folder to Overleaf. The only item still to fill is the YouTube link.

### Repository layout (the important parts)
```
src/data        corruption code, manifests, datasets
src/models      autoencoder, classifier, soft MoE, cGAN
src/train       training loops for each task
src/optuna_studies   the four Optuna searches
src/eval        test evaluation, routing analysis
src/export      ONNX export + verification
app/backend     FastAPI + Dockerfile       app/frontend   React + Tailwind + Dockerfile
configs/        selected hyper-parameters  results/       small result files (tables, figures)
scripts/        run scripts, download_models.sh, report/MLflow helper scripts
notebooks/      the Kaggle notebooks used for training
tests/          pytest tests    docs/    this guide, demo script
```

### Demo video (5 to 7 minutes)
`docs/DEMO_SCRIPT.md` has a timed plan: start-up with one command, upload, run-time corruption, universal restoration, hard routing, soft weights, face-to-sketch, downloading results, and the MLflow records.

---

## 15. How the work actually went (and what went wrong)

1. **Compute plan:** a shared company GPU server was rejected so other people's work would not be affected; **Kaggle GPUs** were used instead.
2. **Built and tested the code first** (data, models, training, Optuna, evaluation, ONNX, backend, frontend) with unit tests, then ran the long training jobs in three Kaggle notebooks.
3. **Problems found and fixed:**
   - a `.gitignore` rule hid the `src/data` folder, which broke the first cloud run (fixed and verified with a fresh clone);
   - MLflow's plain-file store was rejected, so the SQLite store was used;
   - the app's first look differed from the Stitch design, so the gaps were closed and a new Stitch round was made;
   - layout bugs (clipped metric cards, wrapped labels) were found with real-browser screenshots;
   - **at the very end the ONNX files in `models/` turned out to be untrained placeholders** (identical output for every input). Testing the running app with real images caught it; the real ONNX files were downloaded from the final Kaggle run, verified and re-tested.
4. **Honest process point:** the results in the report are read from the evaluation output files, not typed by hand.

---

## 16. Honest limitations

- **One training run per model**, so there are no error bars; small differences (such as 0.01 dB) should not be over-interpreted.
- Some **best hyper-parameters lie at the edge of the searched range**, so a wider search might do better.
- The **strict bottleneck** caps quality at about 25 dB even for clean images; no experiment with skip connections was run to measure how much quality that costs.
- **Task 4 style 3** has only 46 test images, so its numbers are uncertain; sketches are smoother and lighter than the artists' strokes, especially hair.
- The first-run vs final-run improvement for Task 1 (+5.9 dB) mixes several changes (bottleneck type, longer training, bigger search), so it cannot be credited to one cause.
- During training the occlusion rectangles may overlap, while the fixed test manifests use non-overlapping ones (a small mismatch).

---

## 17. Questions an evaluator may ask, with answers

**Why does restoring a lightly blurred image sometimes make it worse?**
The autoencoder squeezes everything through a small code, so it cannot reproduce all fine detail even for a clean image (about 25 dB). If the damaged input is already better than that, the output is worse. The soft mixture handles this by giving weight to the identity branch.

**Why is the soft mixture's PSNR lower than hard routing's overall, but its SSIM higher?**
It loses PSNR on clean images (52.4 vs 60 dB, since identity gets weight 0.94 rather than 1) and on low occlusion, but keeps better structure (SSIM) everywhere damaged.

**What stops one expert from taking over the mixture?**
The balance loss term (delta), the warm-up with frozen experts, and a collapse check that prunes any trial where one branch averages above 0.9.

**Why is Optuna used with a pruner?**
Most settings are bad; the pruner stops them after a few epochs, which let 20 to 30 trials per task fit on limited Kaggle time.

**Why ONNX?**
So the web backend runs without PyTorch, in a small CPU-only container, with outputs verified to match PyTorch within 1e-4 (actually within 5e-6).

**How does the style condition actually work?**
A learned embedding per style is fed to the generator (concatenated at the input and through FiLM scale/shift in every decoder block) and to the discriminator. The conditioning test shows the matching style always gives the lowest error.

**How do you know the validation and test sets are not leaking?**
Pets: fixed 80/20 split of the official trainval with seed 42, the official test set only used at the end. FS2K: validation is carved out of the official training set; the official test set is never used for training or tuning. The corruption manifests are fixed files.

**How is damage applied in the app, and is it the same as training?**
With the same numpy corruption module the training and evaluation use; a seed makes it reproducible.

**What would you do next?**
Repeat runs with several seeds, widen the Optuna ranges that hit their edges, test skip connections, give style 3 more data, and add perceptual losses for sharper sketches.

**Can you modify something live?**
Good candidates to rehearse: change a corruption severity preset (`src/data/corruption_spec.py`), change the gate temperature (`src/models/soft_moe.py`), add a sample image to `app/backend/samples/`, or point the app at a different model file by changing the file name in `app/backend/app/config.py`.

---

## 18. Glossary

| Term | Meaning in plain words |
|---|---|
| Autoencoder | Network that squeezes an image into a small code and rebuilds it |
| Bottleneck | The narrow middle of the autoencoder |
| Checkpoint | A saved copy of a trained model |
| Classifier | A model that picks one label from several |
| Conditional GAN | A GAN that also receives a condition (here photo + style) |
| Corruption | The artificial damage added to an image |
| Docker / container | A packaged, isolated program environment |
| Embedding | A learned vector of numbers that stands for something (e.g. a style) |
| Epoch | One full pass over the training images |
| Expert | A network that specialises in one kind of input |
| FiLM | Scale-and-shift of features controlled by a condition |
| Gate | The network that gives each expert a weight |
| GAN | Generator vs discriminator game for making realistic images |
| Hyper-parameter | A setting chosen before training (learning rate, layer width, ...) |
| L1 / PSNR / SSIM | Image-difference measures (lower / higher / higher is better) |
| Loss | The number that training tries to make small |
| Manifest | A fixed list describing exactly which damage each test image gets |
| MLflow | Tool that records experiments |
| MoE | Mixture of experts |
| ONNX | Portable model file format |
| Optuna | Tool that searches hyper-parameters automatically |
| Oracle routing | Always choosing the correct specialist (an upper bound for hard routing) |
| Pruning | Stopping a clearly bad trial early |
| Seed | A number that makes randomness repeatable |
| Skip connection | A shortcut that bypasses the bottleneck (not used in Tasks 1 to 3) |
| Softmax | Turns scores into weights that sum to 1 |
| Stitch | Google's AI interface-design tool |
| U-Net | Encoder-decoder with skip connections |
| Validation set | Data used to choose settings during development |
