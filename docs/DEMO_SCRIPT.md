# Demo video script (target 6 minutes, limit 5 to 7)

The assignment requires the video to show: application start-up, image upload, run-time corruption, universal restoration, hard routing,
soft expert weights, face-to-sketch generation, result downloading, and experiment-tracking records.
Upload it to YouTube and put **only the link** in the report. Do not upload it to Google Classroom.

## Before recording (10 minutes)

- [ ] Real ONNX models are in `models/` (7 files) and the sample pictures are in `app/backend/samples/`.
- [ ] Stop any running containers: `docker compose down`. Close other windows and notifications. Browser zoom 100 %, full screen.
- [ ] Have these ready in separate tabs / windows: the Linux terminal in the repo folder, the browser, and a second terminal for MLflow.
- [ ] Download the Kaggle `outputs/` folder (at least `outputs/mlflow/`, the `trials.csv` files, `eval_v2/`, `eval_t4/`) to the repo so the tracking records can be shown.
- [ ] Prepare one **clean pet photo** (not from the dataset), one **already corrupted image** (for example a blurred or noisy copy), and a **face photo of yourself or the webcam**.
- [ ] Rehearse once with a timer. Speak in short sentences; the script below is a guide, not a text to read.

## Timeline

| Time | What to show | What to say (one or two sentences) |
|---|---|---|
| 0:00 | Title card or the report title; the repository page | "This is Restoration Studio: four generative models behind one web application." |
| 0:15 | Terminal: `docker compose up --build` (or `up -d` if already built); show both containers healthy; open `http://localhost:8080` | "The whole system starts with one command. The sidebar shows the backend is online with seven models loaded." |
| 0:50 | **Universal Restoration.** Pick a sample, choose *Salt-and-pepper*, *Medium*, set a seed, press *Apply corruption and restore* | "The corruption is applied at run time with the same code used in training; a seed makes it reproducible." |
| 1:30 | Point at the three panels, the PSNR / SSIM cards with the gain, inference time; switch to **Compare** (drag the slider) and **Difference map** | "The input, the restored output, the settings, the inference time and the quality metrics are shown." |
| 2:00 | Change to *Gaussian blur / High*, then *Occlusion / High*; run each | "One universal autoencoder handles all three corruptions; the baseline shows how much each one is repaired." |
| 2:30 | Upload your **already corrupted** image, set corruption *None*, restore | "Unseen, already damaged images can be uploaded too." |
| 3:00 | **Hard-Routed Restoration.** Run a blur image; show the four probabilities, predicted class, selected expert, timing split; then a clean image to show the identity bypass | "A classifier chooses exactly one specialist; clean images skip the experts." |
| 3:50 | **Soft Mixture-of-Experts.** Run the same image; show the four routing weights, the stacked bar, the dominant expert sentence and the entropy | "Here every branch gets a continuous weight and the experts and the gate were trained jointly." |
| 4:40 | Click *Download restored image* (show the file); *Copy metrics as JSON* | "Results can be downloaded, and the metrics exported as JSON." |
| 4:55 | **Face-to-Sketch.** Use the webcam or upload a face; capture; generate with Style 1, 2 and 3; show the original and sketch side by side; click *Download sketch* | "A conditional GAN generates the sketch; the style embedding is part of both networks." |
| 5:50 | **Experiment tracking.** In the second terminal: `mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db`; open `http://localhost:5000`; show an Optuna study with its trials and one final training run with its curves | "Every run, trial, loss curve and sample image was logged with MLflow, and every task was tuned with Optuna." |
| 6:30 | Back to the application or the report's first page; close | "The code, Docker setup, ONNX export, tests and the report are in the repository." |

## Tips

- If something fails live, stop, say so, and re-run that step; a short retake is better than a confusing recording.
- Keep the mouse movements slow and deliberate; pause two seconds on every result so it can be read.
- The MLflow artifacts folder may show broken image links because they were logged on Kaggle with absolute paths; the parameters, metrics and
  curves still display correctly, which is what the video needs.
- Record with Windows Game Bar (Win + G) or OBS; export at 1080p; upload as *Unlisted* and test the link in a private window before submitting.
