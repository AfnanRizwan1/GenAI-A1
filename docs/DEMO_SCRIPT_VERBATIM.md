# Demo video: word-for-word script (about 6.5 minutes)

Lines in quotes are what you say. Lines marked **YOU DO** are actions on screen. The assignment requires 5 to 7 minutes and
must show: start-up, image upload, run-time corruption, universal restoration, hard routing, soft expert weights, face-to-sketch,
result downloading, and experiment-tracking records. Upload the video to YouTube (Unlisted) and put only the link in the report.

## Files you will upload during the video

Folder `demo_assets/` in the repository:

- `already_noisy.png`, `already_blurred.png`, `already_occluded.png`: images that are already damaged (for the "unseen image" step).
- `face_demo.png`: fallback face photo. Use your own face photo if you prefer.

## Before you press record (5 minutes)

1. Open **three windows**: a PowerShell for Docker, a second PowerShell in the repository folder for MLflow, and Chrome (zoom 100 %, full screen).
2. In the Docker PowerShell run `wsl -d Docker -u root`, then
   `cd "/mnt/d/FAST/Semester 7/Gen AI/Assignments/Assignment 1/GenAI-A1"`, then `docker compose down`.
   Leave the prompt ready. Do **not** run `up` yet.
3. In the MLflow PowerShell, `cd` to the repository folder and type the command below **without pressing Enter**:
   `mlflow ui --backend-store-uri sqlite:///outputs/mlflow_t12/mlflow.db`
4. Turn off notifications. Open the repository's GitHub page in a Chrome tab.
5. Do one practice run with a timer.

## Script

### 0:00 to 0:20. Introduction
**SCREEN:** the GitHub repository page.

> "Hello, I am Afnan Rizwan. This is my Generative AI Assignment 1, called Restoration Studio. It has four generative models behind one web application: three that repair damaged images, and one that turns a face photo into a pencil sketch."

### 0:20 to 0:50. Start-up
**SCREEN:** the Docker terminal.
**YOU DO:** type `docker compose up -d` and press Enter. When it finishes, type `docker compose ps` and press Enter. Then switch to Chrome and open `http://localhost:8080`.

> "The whole system starts with one command, docker compose up. The terminal shows two containers, the backend and the frontend, both running. In the browser, the sidebar at the bottom says: backend online, seven models loaded. That means the application is ready."

### 0:50 to 2:00. Universal restoration (Task 1)
**SCREEN:** the Universal Restoration workspace.
**YOU DO:** click the **third sample** photo. Under Corruption click **Salt-and-pepper**. Under Severity click **Medium**. In the Seed box type **5**. Click **Apply corruption and restore**.

> "This is Workspace 1, Task 1, universal restoration. One autoencoder with a compressed bottleneck repairs every kind of damage. I pick a sample image, choose salt-and-pepper noise at medium severity, and type a seed, so the same damage can be repeated. The damage is applied at run time with the same code used in training."

**YOU DO:** wait for the result and slowly move the mouse over the three images and then the cards.

> "Here are the original, the corrupted input, and the restored output. The cards show the inference time, and the PSNR and SSIM compared with the clean original. The quality goes up by several decibels, and the structure score goes up a lot."

**YOU DO:** click **Compare** and drag the slider once. Then click **Difference map**.

> "The compare slider shows the change directly, and the difference map shows where the remaining error is."

### 2:00 to 2:40. Other corruptions and an already-damaged image
**YOU DO:** click **Occlusion**, then **High**, then **Apply**. Wait two seconds. Then click **Gaussian blur**, **Medium**, **Apply**. Point at the red numbers.

> "The same model handles occlusion, where black rectangles hide part of the image. For blur, the numbers turn red. That is expected and honest: the bottleneck limits quality to about twenty-five decibels, so on mild blur the model cannot beat the input. The report explains this limit."

**YOU DO:** under Corruption click **None**. Click the upload box and choose `demo_assets/already_noisy.png`. Click **Restore**.

> "I can also upload an image that is already damaged. With corruption set to none, the model restores it as it is. This is an unseen image that was not part of the dataset."

### 2:40 to 3:30. Hard-routed restoration (Task 2)
**YOU DO:** click **Hard-Routed Restoration** in the sidebar. Click a sample photo. Choose **Gaussian blur**, **High**, **Apply**.

> "Workspace 2, Task 2, hard routing. A classifier first recognises the type of damage, then the image goes to exactly one specialist autoencoder. The bars show the classifier probabilities, blur is close to one hundred percent, and the chip shows which expert was chosen. The timing is split between the classifier and the expert."

**YOU DO:** choose **Salt-and-pepper**, **Apply**. Then choose **None** with a clean sample and click **Restore**.

> "For salt-and-pepper, the salt specialist is chosen. For a clean image, the classifier says clean and the image bypasses the experts, so the output is identical to the input. On the test set the classifier is correct for ninety-nine point eight percent of cases."

### 3:30 to 4:20. Soft mixture of experts (Task 3)
**YOU DO:** click **Soft Mixture-of-Experts Restoration**. Choose a sample, **Gaussian blur**, **Medium**, **Apply**.

> "Workspace 3, Task 3, the soft mixture of experts. Here the gate gives every branch a continuous weight, and the output is the weighted blend of the identity branch and the three experts. The gate and the experts were trained together. The bars show the four weights, here mostly the blur expert, and the sentence names the strongest contribution. The entropy tells how concentrated the routing is."

**YOU DO:** switch Severity to **Low** and click **Apply**. Point at the weights.

> "At low severity, the weight moves toward the identity branch, because little needs repairing. This is why the soft mixture is better than the other models on mild blur."

### 4:20 to 4:40. Downloads
**YOU DO:** click **Download restored image**, then **Copy metrics as JSON**. Open the Downloads folder briefly to show the file.

> "Results can be downloaded as an image, and the metrics can be copied as JSON."

### 4:40 to 5:30. Face-to-sketch (Task 4)
**YOU DO:** click **Face-to-Sketch Generator**. Click the upload box and choose your face photo, or `demo_assets/face_demo.png`. Select **Style 1**, click **Generate sketch**. Then **Style 2**, generate. Then **Style 3**, generate.

> "Workspace 4, Task 4, face-to-sketch. A conditional GAN turns a face photo into a sketch in a style chosen by the user. I upload a photo, pick Style 1, and generate. The original and the sketch appear side by side. Style 2 gives heavier strokes, and Style 3 sits in between. The style is a learned embedding that is part of both the generator and the discriminator."

**YOU DO:** click **Download sketch**.

> "The sketch can be downloaded here."

(The webcam is optional. The assignment allows "upload or capture"; uploading is enough for the video. If you want to show it, click **Use webcam**, allow the camera and capture one frame.)

### 5:30 to 6:20. Experiment tracking
**SCREEN:** the MLflow PowerShell.
**YOU DO:** press Enter on the prepared command. Open Chrome at `http://localhost:5000`. At the top left click **Model training** (not the GenAI view, which is empty) and close the MLflow Assistant panel with its X. Click **optuna-autoencoders**, open the **Runs** tab, then open one run and point at its parameters, metrics and charts.

> "Every experiment was tracked with MLflow, and every task was tuned with Optuna. Here is the Optuna study for the autoencoders, with each trial's parameters and result. Opening a run shows its settings, losses and metrics. The Task 3 records and the final checkpoints and evaluation are in a second store, and the Task 4 records are in a third."

### 6:20 to 6:35. Closing
**SCREEN:** the GitHub repository page, or the first page of the report.

> "The code, Docker setup, ONNX export, tests and the report are all in the repository. Thank you for watching."

## Tips

- Pause two seconds on each result and move the mouse slowly.
- If something fails, say "let me retry" and redo that step. A short retake is better than a confusing recording.
- Total target is 6:30; the limit is 5 to 7 minutes.
- The MLflow stores are three folders; open one at a time (stop one with Ctrl+C before starting the next):
  - `outputs/mlflow_t12`: Tasks 1-2 (Optuna trials and training runs)
  - `outputs/mlflow`: Task 3 and the final record of every model
  - `outputs/mlflow_t4`: Task 4
  Command: `mlflow ui --backend-store-uri sqlite:///<folder>/mlflow.db`, then open `http://localhost:5000`.
- Image previews inside MLflow may look broken (they were logged on Kaggle with Kaggle paths); the parameters, metrics and curves are fine.
- Record with Windows Game Bar (Win + G) or OBS at 1080p. Upload as **Unlisted** and test the link in a private window.
