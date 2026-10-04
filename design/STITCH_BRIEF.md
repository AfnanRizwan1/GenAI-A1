# Google Stitch brief

Paste the prompt below into Google Stitch (stitch.withgoogle.com), generate the screens, tweak if you like,
then **save screenshots of the original generated designs** (and the Stitch project link / export) into this folder.
They are the evidence the assignment asks for in the report. Suggested file names: `stitch_1_universal.png`,
`stitch_2_hard.png`, `stitch_3_soft.png`, `stitch_4_sketch.png`, `stitch_mobile.png`, plus a screenshot showing the
Stitch project with its date.

## Prompt

Design a responsive web app called "Restoration Studio" for a generative-AI course project. It is one coherent
application with a left sidebar that links to four workspaces; on mobile the sidebar collapses to a top bar
with a menu button. Clean, modern, light theme with a dark-mode variant; indigo primary colour, soft gray surfaces,
rounded cards, subtle shadows, generous spacing. Show a small status chip in the sidebar footer
("Backend online · 7 models loaded").

Workspace 1 - "Universal Restoration": a controls card (upload dropzone with drag and drop, or a row of clean
sample thumbnails to pick from; a corruption selector with segmented buttons: None / Salt-and-pepper / Gaussian blur
/ Occlusion; a severity selector Low / Medium / High / Custom with sliders that appear for Custom; an "Apply and
restore" primary button). Below: three image panels side by side labelled Original, Input to model, Restored output, each
128x128 shown enlarged with a caption. A details card shows the selected corruption settings (type, severity,
parameters, seed), inference time in milliseconds, and PSNR / SSIM of input versus restored. A "Download result" button.

Workspace 2 - "Hard-Routed Restoration": same controls and image panels, plus a card with four horizontal probability
bars (Clean, Salt-and-pepper, Gaussian blur, Occlusion) with percentages, a highlighted "Predicted corruption" badge,
a "Selected expert" badge (or "identity bypass" for clean), and timing split into classifier time and expert time.

Workspace 3 - "Soft Mixture-of-Experts": same controls and image panels, plus a card with four routing-weight bars
(Identity, Salt-and-pepper expert, Blur expert, Occlusion expert) sorted by contribution with the strongest highlighted,
a short "dominant expert" sentence, and inference time.

Workspace 4 - "Face-to-Sketch Generator": a photo uploader plus a "Use webcam" button that opens a camera preview with
a capture button; a segmented selector Style 1 / Style 2 / Style 3; a "Generate sketch" button. Two large panels side by
side: Original photo and Generated sketch, with inference time and a "Download sketch" button.

All screens need friendly empty states, a loading state with a spinner on the primary button, and an inline error banner
for failed uploads. Keep the style consistent across the four workspaces.
