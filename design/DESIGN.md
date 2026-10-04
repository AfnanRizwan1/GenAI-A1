---
name: Restoration Studio Precision System
colors:
  surface: '#f6f7fb'
  surface-dim: '#e3e6ef'
  surface-bright: '#ffffff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f1f3f9'
  surface-container: '#eaedf6'
  surface-container-high: '#e1e5f1'
  surface-container-highest: '#d7dcec'
  on-surface: '#0f172a'
  on-surface-variant: '#475569'
  inverse-surface: '#0f172a'
  inverse-on-surface: '#f1f5f9'
  outline: '#94a3b8'
  outline-variant: '#e2e8f0'
  surface-tint: '#4f46e5'
  primary: '#4f46e5'
  on-primary: '#ffffff'
  primary-container: '#eef2ff'
  on-primary-container: '#3730a3'
  inverse-primary: '#a5b4fc'
  secondary: '#0284c7'
  on-secondary: '#ffffff'
  secondary-container: '#e0f2fe'
  on-secondary-container: '#075985'
  tertiary: '#059669'
  on-tertiary: '#ffffff'
  tertiary-container: '#d1fae5'
  on-tertiary-container: '#065f46'
  error: '#dc2626'
  on-error: '#ffffff'
  error-container: '#fee2e2'
  on-error-container: '#991b1b'
  primary-fixed: '#e0e7ff'
  primary-fixed-dim: '#c7d2fe'
  on-primary-fixed: '#1e1b4b'
  on-primary-fixed-variant: '#3730a3'
  secondary-fixed: '#bae6fd'
  secondary-fixed-dim: '#7dd3fc'
  on-secondary-fixed: '#082f49'
  on-secondary-fixed-variant: '#075985'
  tertiary-fixed: '#a7f3d0'
  tertiary-fixed-dim: '#6ee7b7'
  on-tertiary-fixed: '#022c22'
  on-tertiary-fixed-variant: '#065f46'
  background: '#f6f7fb'
  on-background: '#0f172a'
  surface-variant: '#e1e5f1'
typography:
  display-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 36px
    fontWeight: '800'
    lineHeight: 42px
    letterSpacing: -0.03em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 30px
    fontWeight: '700'
    lineHeight: 38px
    letterSpacing: -0.025em
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 20px
    fontWeight: '700'
    lineHeight: 28px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 16px
    fontWeight: '700'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 26px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 22px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0em
  label-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '600'
    lineHeight: 18px
    letterSpacing: 0.01em
  label-caps:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.08em
  metric-lg:
    fontFamily: JetBrains Mono
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.03em
  metric-md:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: -0.01em
  metric-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
rounded:
  sm: 0.375rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1.5rem
  gutter-sm: 1rem
  gutter-lg: 2rem
  margin: 2rem
  margin-sm: 1rem
  margin-lg: 3rem
  space-2xs: 0.125rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
  space-2xl: 3rem
---

## Brand & Style

**Restoration Studio** is a calm, precise workbench for four generative-AI image models: three image-restoration
workspaces (universal autoencoder, hard-routed specialists, soft mixture-of-experts) and a face-to-sketch generator.
The interface should feel like a premium professional tool: quiet, confident, uncluttered and trustworthy.
The photographs are the hero; the interface recedes around them.

The style is **Modern Technical Minimalism with tonal layering**: pale cool-gray canvas, pure white cards with hairline
borders and very soft shadows, one confident indigo accent, and monospaced numerals for every measurement.

### Principles
1. **One clear action per screen.** A single filled indigo primary button; everything else is secondary or ghost.
2. **Images first.** Image panels get the most area and visual weight, on a neutral recessed background so colors read truthfully.
3. **Honest data only.** Show only information the system really computes (see *Content integrity*). Never invent model names, hardware, scores or specs.
4. **Calm density.** Generous white space, aligned to an 8 px grid, no decorative badges that carry no information.
5. **Visible state.** Empty, loading, success and error states are always designed, never left blank.

## Content integrity (hard rules)

The application is a research demo, not a marketing page. Do **not** show: GPU or hardware names (CUDA, RTX, TensorRT, FP16),
network architecture names (ResNet, U-Net, diffusion, GAN-SD, MedianNet), version numbers, fake confidence scores, stroke counts,
LPIPS, VRAM, hashes, or model graphs. Inputs are always **128 x 128 RGB**; uploads are **JPEG, PNG, WebP or BMP up to 10 MB**; inference runs
on **ONNX Runtime (CPU)**; the status chip reads **"Backend online · 7 models loaded"** with a second chip **"ONNX Runtime · CPU"**.

Data that may be displayed, per workspace:
- **All restoration workspaces:** corruption type; severity (Low / Medium / High / Custom); parameters (salt-and-pepper fraction, blur kernel size and sigma, occlusion area and rectangle count); seed; inference time in ms; PSNR and SSIM of input vs restored; a difference map.
- **Hard-Routed:** four class probabilities (Clean, Salt-and-pepper, Gaussian blur, Occlusion); predicted corruption; selected expert or **Identity bypass** for clean; classifier time and expert time.
- **Soft Mixture-of-Experts:** four routing weights (Identity, Salt-and-pepper expert, Blur expert, Occlusion expert) summing to 100 %; the dominant expert in plain language; routing entropy.
- **Face-to-Sketch:** exactly three styles named **Style 1, Style 2, Style 3**; inference time. No sliders, scores or extra settings.

## Colors

A restrained palette; color always carries meaning.

| Role | Light | Dark | Use |
|---|---|---|---|
| Canvas | `#F6F7FB` | `#0B0F19` | Page background |
| Surface (cards, sidebar) | `#FFFFFF` | `#131A2A` | Elevated containers |
| Surface subdued (wells, tracks, inputs) | `#F1F3F9` | `#1A2338` | Recessed areas, segmented tracks |
| Image stage | `#EAEDF6` | `#0F1524` | Neutral frame behind photographs |
| Border | `#E2E8F0` | `#25304A` | 1 px hairlines |
| Text primary | `#0F172A` | `#F1F5F9` | Titles, values |
| Text secondary | `#475569` | `#94A3B8` | Descriptions, labels |
| Text muted | `#94A3B8` | `#64748B` | Hints, units |
| **Primary indigo** | `#4F46E5` (hover `#4338CA`) | `#6366F1` (hover `#818CF8`) | The one primary action, active nav item, selected state, highlighted bar |
| Primary tint | `#EEF2FF` | `#1E1F4B` | Active nav background, soft badges |
| **Emerald** | `#059669` on `#D1FAE5` | `#34D399` on `#0B2E24` | Improvement only: positive PSNR / SSIM change, "ready" dots |
| **Sky** | `#0284C7` on `#E0F2FE` | `#38BDF8` on `#0B2A3D` | Selection and focus rings, info banners |
| Amber | `#D97706` on `#FEF3C7` | `#FBBF24` on `#33260A` | Warnings, partially loaded models |
| Red | `#DC2626` on `#FEE2E2` | `#F87171` on `#3A1212` | Errors, degraded input labels |

**Accent treatment.** A 3 px indigo-to-violet gradient line (`#4F46E5` to `#7C3AED`) sits under each page header, and the
primary button carries a faint indigo glow (`0 4px 14px rgba(79,70,229,0.28)`). No other gradients.

## Typography

Three families, each with one job.
- **Plus Jakarta Sans** (700 to 800): page titles, card titles, brand name. Tight negative letter-spacing.
- **Inter** (400 to 600): all interface copy, buttons, labels and descriptions.
- **JetBrains Mono** (500 to 600): every number with a unit or index: ms, dB, SSIM, percentages, seeds, parameters. Tabular figures so values do not jitter.

Hierarchy on a workspace page: eyebrow label (`label-caps`, indigo, e.g. "WORKSPACE 01") above a `headline-lg` title, then a
single-sentence `body-md` description in secondary text, capped at 70 characters per line. Card titles use `headline-sm`.
Field labels use `label-caps` in muted text.

## Layout & Spacing

Persistent left sidebar plus a fluid content area; no right panel.
- **Sidebar (desktop, 280 px):** brand mark (36 px indigo rounded square with a white glyph) and the name "Restoration Studio" with the small caption "Generative AI · Assignment 1"; four navigation items; at the bottom the status chips and the light/dark toggle.
- **Content (max width 1152 px, centered):** page header, then a two-column grid at 1280 px and above: a **controls card of 384 px** on the left and the **results area** (fluid) on the right.
- **Results area:** three equal image panels in a row (Original, Input to model, Restored output), then a row of information cards (task-specific card plus run details).
- **Sketch workspace:** two large equal panels side by side (Original photo, Generated sketch) with the controls card on the left.
- **Spacing:** 8 px grid. 24 px between cards, 24 px padding inside cards (20 px on small screens), 16 px between related controls, 8 px between a label and its field.
- **Breakpoints:** below 768 px: single column, a top bar with a menu button replaces the sidebar, controls stack above the results, image panels stack one per row. 768 to 1279 px: sidebar collapses to the top bar, results in one column of three panels. 1280 px and above: full layout.

## Elevation & Depth

Hairline borders do most of the work; shadows are soft and diffused.
- **Level 0 canvas:** flat.
- **Level 1 sidebar, top bar:** white, 1 px border, `0 1px 2px rgba(15,23,42,0.04)`.
- **Level 2 cards:** white, 1 px border, `0 1px 2px rgba(15,23,42,0.04), 0 8px 24px rgba(15,23,42,0.06)`.
- **Level 3 floating labels, toasts:** 12 px backdrop blur, 85 % opaque surface, `0 10px 24px rgba(15,23,42,0.12)`.

## Shapes

- Cards and image panels: 24 px radius (`xl`).
- Inner elements (inputs, thumbnails, metric wells, segmented tracks): 12 px (`md`).
- Buttons: 12 px. Small segments inside a track: 8 px.
- Pills, chips and dots: fully round.

## Components

### Buttons
- **Primary:** filled indigo, white Inter 600 text, 44 px tall, 12 px radius, indigo glow; hover darkens and lifts 1 px; pressed removes the lift; disabled at 50 % opacity; **loading** replaces the icon with a 16 px spinner and the label changes to a verb in progress ("Restoring…").
- **Secondary:** white surface, 1 px border, slate text, hover adds a soft shadow.
- **Ghost:** no border, muted text turning primary on hover (used for small viewport actions).
- Focus: a 2 px sky ring with a 2 px offset on every interactive element.

### Navigation item
48 px tall, 12 px radius, icon (20 px) and label, with a small muted "Task 1 to 4" caption on the right. Active: indigo tint background, indigo text and a 3 px indigo bar on the left edge. Hover: subdued background.

### Image panel (the key component)
A white card with a title row (title in `headline-sm`, a mono caption on the right such as "128 x 128"), a **square** image stage on the recessed neutral background, and floating frosted labels in the top-left corner of the image (for example "ORIGINAL", "CORRUPTED INPUT" in red tint, "RESTORED" in emerald tint) in `metric-sm`. Under the image a one-line caption. Images are shown at 3x native size with smooth scaling. Empty state: a dashed outline with a muted icon and a short instruction. Loading state: a soft shimmer skeleton. The Restored panel has a **view toggle** (Side by side / Difference map) and, in the Universal workspace, an optional **before / after slider** with a 2 px white divider and a round dual-arrow handle.

### Segmented control
Recessed 40 px track (`surface subdued`), active segment is a white pill with a soft shadow and indigo text; the active pill glides to the new position in 200 ms. Used for Corruption, Severity, View and Style.

### Upload drop zone
Dashed 2 px border in `outline-variant`, 12 px radius, an upload icon in an indigo-tint circle, the text "Drop an image here or browse", and the hint "JPEG, PNG, WebP or BMP, up to 10 MB. Resized to 128 x 128". Dragging over turns the border indigo and tints the background. Below it a row of four 56 px sample thumbnails with a 2 px indigo ring on the selected one.

### Sliders and fields
4 px track, filled portion in indigo, 18 px white thumb with a 1 px border and a visible focus ring. The current value is shown on the right in mono. The Seed field is a 112 px mono input with a "random" placeholder and the hint "same seed = same corruption".

### Probability and weight bars
Rows with the label on the left and the percentage in mono on the right; a 10 px rounded track below. The highest row is indigo and bold, all others are slate. Widths animate in 500 ms with ease-out. Routing weights are sorted by value. A dominant-expert sentence follows in plain language.

### Badges and status pills
Fully round, 28 px tall, `label-md`. Predicted corruption: indigo tint. Selected expert: emerald tint (slate tint with the text "Identity bypass" for clean). Status chip with a 6 px dot: green for ready, amber for partially loaded, red for offline.

### Metric cards
A white inner well with an uppercase `label-caps` title, a large `metric-lg` value, and a unit in muted text. Positive change is shown as an emerald pill with an up arrow and the delta (for example "+4.8 dB"). Used for inference time, PSNR (input to restored) and SSIM (input to restored).

### Run details card
A definition list with hairline separators: corruption, severity, parameters, seed, inference time, classifier and expert time where relevant. Buttons beneath: Download restored image, Download input, Copy metrics as JSON.

### Banners and toasts
Inline error banner with a red tint, an alert icon, the message in plain words ("File is larger than 10 MB") and a Retry and Dismiss action. Info banner in sky tint for hints. No modal dialogs.

## Motion

Subtle and purposeful, never decorative. All transitions use `cubic-bezier(0.2, 0, 0, 1)`.
- Hover and focus: 150 ms. Segmented pill glide: 200 ms. Bars: 500 ms.
- Results fade in and rise 8 px over 300 ms, staggered by 60 ms across the three image panels.
- While a model runs: the button shows a spinner and the Restored panel shows a shimmer.
- Page change: a 150 ms fade. Respect `prefers-reduced-motion` by disabling all of the above except opacity.

## Dark mode

Same structure, same accents. The canvas becomes `#0B0F19`, cards `#131A2A` with `#25304A` hairlines, shadows are replaced by slightly
lighter borders, image stages become `#0F1524`, and indigo shifts to `#6366F1`. Text contrast stays at least 4.5:1.

## Accessibility

Text contrast at least 4.5:1 and interactive controls at least 3:1. Every control has a visible focus ring. Touch targets are at least 44 px.
Color is never the only signal: bars carry percentages, badges carry words. Images have descriptive alt text and the segmented controls use radio semantics.

## Iconography

Outline icons only (Lucide / Material Symbols Outlined style), 20 px in navigation and buttons, 16 px inside chips, 1.75 px stroke,
`on-surface-variant` color, turning indigo when active. Suggested glyphs: Universal Restoration = sparkles, Hard-Routed = git-branch
(one path chosen), Soft Mixture-of-Experts = sliders or layers, Face-to-Sketch = pencil. No filled or multicolor icons, no emoji.

## Imagery

- **Sample images** in the restoration workspaces are close-up photographs of cats and dogs on simple backgrounds; in the sketch workspace they are front-facing portrait photographs. All are square-cropped, with consistent brightness.
- **Never** use screenshots, charts or illustrations as samples, and never show a person in the restoration workspaces.
- A clean original, its corrupted version and the restored version must visibly be **the same picture** (same subject and framing); corruptions are salt-and-pepper speckles, Gaussian blur, or black rectangles placed over the subject.
- The sketch panel shows a grayscale pencil-style rendering of the same face as the photo, at the same size.
- Image stages use a neutral recessed background and never a tinted or patterned one, so photographs are judged on their own colors.

## States and feedback

Design all of these explicitly for every workspace:
- **Empty (first visit):** controls enabled, primary button disabled with the hint "Choose an image or a sample to begin", result panels showing dashed placeholders with an instruction.
- **Ready:** an image is chosen, its thumbnail is shown in the drop zone, the primary button is active.
- **Running:** the primary button shows a spinner and "Restoring…" (or "Generating…"), controls are locked, the Restored panel shows a shimmer skeleton.
- **Success:** results fade in, the gain pills appear (emerald), the download buttons become available.
- **Error:** a red-tint inline banner above the results with a clear sentence and Retry / Dismiss. Examples: "File is larger than 10 MB", "The file is not a readable image", "Cannot reach the backend. Is it running?".
- **Partial backend:** an amber status chip "Backend online · 5 of 7 models loaded".
- **Offline:** a red status chip "Backend offline" and the primary buttons disabled.

## Screen blueprints

All screens share the sidebar, the page header (eyebrow label, title, one-sentence description, gradient accent line) and the status chips.

1. **Universal Restoration.** Left controls card: drop zone, sample thumbnails, Corruption selector (None, Salt-and-pepper, Gaussian blur, Occlusion), Severity selector (Low, Medium, High, Custom, with sliders appearing for Custom), Seed field, primary button "Apply corruption and restore". Right: three image panels in a row (Original, Input to model, Restored output) with the view toggle (Side by side / Difference map); below, two cards side by side: Run details and a metrics card with inference time, PSNR input to restored and SSIM input to restored with emerald gain pills; then the buttons Download restored image, Download input and Copy metrics as JSON.
2. **Hard-Routed Restoration.** Same as 1, plus a card "Classifier probabilities": four bars, the winner highlighted in indigo, a "Predicted corruption" badge and a "Selected expert" badge (the clean case reads "Identity bypass"), and a two-part timing line (classifier ms, expert ms).
3. **Soft Mixture-of-Experts.** Same as 1, plus a card "Routing weights": four bars sorted by weight and summing to 100 %, a one-line plain-language sentence naming the dominant expert (or saying the weights are spread out), a small stacked bar of the four weights, and the routing entropy as a mono value.
4. **Face-to-Sketch Generator.** Left controls card: drop zone, "Use webcam" button (opening a live preview with a Capture photo button), three style options named Style 1, Style 2, Style 3, each with a small thumbnail, primary button "Generate sketch". Right: two large panels side by side (Original photo, Generated sketch) with the style name and inference time; Download sketch button; an example inline error banner for an unsupported file.
5. **Mobile (Universal Restoration).** Top bar with the brand and a menu button; controls card stacked above; the three image panels stacked one per row; the primary button full width and sticky at the bottom of the viewport.

## Do and don't

**Do** keep copy short and plain, align everything to the grid, let the photographs dominate, and use mono numerals for all measurements.
**Don't** add hardware or architecture claims, extra settings that the models do not have, multiple accent colors in one card, heavy gradients, glossy effects, or stock marketing language.
