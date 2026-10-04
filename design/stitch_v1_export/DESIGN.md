---
name: Restoration Studio Precision System
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#464555'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#777587'
  outline-variant: '#c7c4d8'
  surface-tint: '#4d44e3'
  primary: '#3525cd'
  on-primary: '#ffffff'
  primary-container: '#4f46e5'
  on-primary-container: '#dad7ff'
  inverse-primary: '#c3c0ff'
  secondary: '#006591'
  on-secondary: '#ffffff'
  secondary-container: '#39b8fd'
  on-secondary-container: '#004666'
  tertiary: '#005338'
  on-tertiary: '#ffffff'
  tertiary-container: '#006e4b'
  on-tertiary-container: '#67f4b7'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e2dfff'
  primary-fixed-dim: '#c3c0ff'
  on-primary-fixed: '#0f0069'
  on-primary-fixed-variant: '#3323cc'
  secondary-fixed: '#c9e6ff'
  secondary-fixed-dim: '#89ceff'
  on-secondary-fixed: '#001e2f'
  on-secondary-fixed-variant: '#004c6e'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  display-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.025em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0em
  label-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: 0.01em
  metric-lg:
    fontFamily: JetBrains Mono
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.02em
  metric-md:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: -0.01em
  metric-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.02em
rounded:
  sm: 0.25rem
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

This design system embodies high-precision optical engineering combined with generative AI intelligence. Crafted specifically for advanced image restoration workflows (denoising, super-resolution, face restoration, inpainting), the visual identity balances surgical accuracy with refined, fluid digital craft.

The design movement combines **Modern Technical Minimalism** with subtle **Tonal Layering**. It delivers an interface that recedes into the background to prioritize photographic pixels, while elevating control panels and quantitative benchmarks with sharp, structured hierarchy.

### Core Tenets
- **Instrument-Grade Precision:** Every element signals mathematical rigor and confidence. Layouts are strictly aligned to a 4px/8px structural lattice.
- **Content Paramountcy:** Canvas surfaces and preview areas are pristine, quiet, and unobtrusive, preventing color contamination when evaluating processed imagery.
- **Deterministic Transparency:** Algorithmic actions are never opaque. The interface communicates computational states, model parameters, and quality metrics with real-time clarity.

## Colors

The palette leverages pure indigo as a radiant computational beacon, anchored by neutral slates that provide optical separation between tooling and visual assets.

### Palette Architecture
- **Primary Indigo (`#4F46E5` / Light Mode, `#6366F1` / Dark Mode):** Represents active processing, affirmative model checkpoints, and primary generative actions.
- **Secondary Sky (`#0EA5E9`):** Reserved for spatial tooling, viewport indicators, focus reticles, and selection marquees.
- **Tertiary Emerald (`#10B981`):** Applied exclusively to fidelity benchmarks (positive PSNR/SSIM deltas, pipeline success, model convergence).
- **Neutral Slate Range:**
  - `Canvas Neutral` (`#F8FAFC` light / `#0B0F17` dark): Base atmospheric foundation.
  - `Surface Elevated` (`#FFFFFF` light / `#131926` dark): Component cards, docked panels, inspection sheets.
  - `Surface Subdued` (`#F1F5F9` light / `#1A2234` dark): Recessed wells, input backgrounds, slider tracks.
  - `Border Subtle` (`#E2E8F0` light / `#232E47` dark): Structural dividers and container outlines.
  - `Text Primary` (`#0F172A` light / `#F8FAFC` dark): High-contrast titles, values, and primary instructions.
  - `Text Muted` (`#64748B` light / `#94A3B8` dark): Auxiliary labels, tool descriptions, and units.

## Typography

The typographic hierarchy implements a tripartite structure tailored to generative graphics engineering:

1. **Brand & Section Headers (`Plus Jakarta Sans`):** Clean, geometric grotesque cuts that project modern software elegance without geometric distortion.
2. **Operational Copy & UI Elements (`Inter`):** Neutral, hyper-legible humanist sans engineered specifically for dense dashboards and controls.
3. **Data, Parameters & Diagnostics (`JetBrains Mono`):** Dedicated monospaced figures for quantitative tracking:
   - Noise floor decibels, PSNR (dB), SSIM indexes (0.000–1.000).
   - Compute latency (`ms`), memory usage (`GiB`), tensor dimension stamps (`1024x1024`), seeds (`uint32`).
   - Ensures vertical digit alignment during continuous inference re-renders.

## Layout & Spacing

The canvas is driven by an asymmetric workstation layout built around the visual inspection viewport:
- **Left Panel (Tooling & Model Config):** Fixed width (320px–360px), optimized for dense parameter steppers and pipeline chaining.
- **Center Canvas (Viewport Area):** Fluid layout dynamically adapting to aspect ratio and native zoom boundaries.
- **Right Panel (Metrics & Inspector):** Collapsible panel (300px) dedicated to analytical verification (split-view diagnostics, histogram, telemetry).

### Grid & Breakpoints
- **Mobile (`< 768px`):** Single-column stacked mode. The preview viewport takes top priority; controls collapse into bottom sheets using `margin-sm` and `gutter-sm`.
- **Tablet (`768px - 1199px`):** 8-column layout. Tooling stacks in an expandable off-canvas drawer. Viewport utilizes safe margins of `margin`.
- **Desktop (`≥ 1200px`):** 12-column persistent workspace with fluid center canvas and structural spacing (`gutter: 1.5rem`, `margin: 2rem`).

## Elevation & Depth

Visual hierarchy combines **crisp low-contrast boundary lines** with **diffused micro-elevation** to isolate functional instruments from the generative canvas.

### Depth Levels
- **Level 0 (Canvas Base):** Flat background color (`#F8FAFC` / `#0B0F17`). No elevation, serves as neutral zero-reference plane.
- **Level 1 (Docked Containers & Sidebars):** Surface Elevated with a 1px structural outline (`#E2E8F0` / `#232E47`). Shadow: `0 1px 2px 0 rgba(15, 23, 42, 0.04)`.
- **Level 2 (Cards, Inspector Units, Segments):** Elevated surfaces featuring `rounded-xl` geometries. Shadow: `0 4px 6px -1px rgba(15, 23, 42, 0.05), 0 2px 4px -2px rgba(15, 23, 42, 0.03)`.
- **Level 3 (Floating Canvas HUD, Splitter Handles, Overlays):** Higher-order interactive layers. Backdrop blur: `12px` (subtle glass effect at 92% surface opacity). Shadow: `0 10px 15px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -4px rgba(15, 23, 42, 0.03)`.

## Shapes

The design uses a refined curvature profile (`roundedness: 2` with standard radii scaling to 1.5rem / 24px for `rounded-xl`).

### Curvature Tokens
- **Containers & Primary Cards:** `rounded-xl` (1.5rem / 24px) for master workspace modules, preview viewports, and parameter dialogs.
- **Internal Elements & Steppers:** `rounded-lg` (0.75rem / 12px) for input groups, metric wells, dropdown triggers, and image thumbnails.
- **Interactive Controls:** `rounded-md` (0.375rem / 6px) to maintain a precise, tool-like edge for sliders, color swatches, and small segment switches.
- **Status Indicators & Micro Badges:** Fully circular (`rounded-full`) for operational pills and seed chips.

## Components

### Buttons
- **Primary Generative Action:** Deep Indigo background (`#4F46E5`), crisp white typography (`Inter 500`), subtle micro-highlight on hover (`#4338CA`), pressed state offset by 0.5px. Incorporates an inline loading spinner or progress ring during model execution.
- **Secondary / Action Tooling:** Surface Subdued background with 1px border (`#E2E8F0`), primary slate text. Hover transition to `#FFFFFF` with `shadow-sm`.
- **Ghost Utility:** Borderless for viewport actions (zoom-to-fit, rotate, reset crop) with `text-muted` shifting to `text-primary`.

### Metric Cards
- Standardized monitoring containers displaying inference analytics.
- **Header:** Uppercase `metric-sm` label (e.g., `PSNR GAIN`, `SSIM`, `LATENCY`) accompanied by an auxiliary confidence dot.
- **Body:** Large numerical display in `metric-lg` (`JetBrains Mono`). Positive performance deltas render in `#10B981` (e.g., `+4.82 dB`), baseline metrics in neutral dark.
- **Footer:** Sparkline or sub-metric status (e.g., `512x512 → 2048x2048`).

### Image Comparison Panel
- **Split View Bar:** 2px high-contrast vertical divider line (`#FFFFFF` with `#0F172A` drop-shadow) with an ergonomic, tactile dual-arrow thumb grabber centered on the axis.
- **Labels:** Floating minimal pills positioned at top-left and top-right corners (`ORIGINAL (DEGRADED)` vs `RESTORED (ESRGAN-V2)`) using `metric-sm` type on frosted glass backdrop (`rgba(15, 23, 42, 0.6)` with white text).
- **Zoom & Pan Controls:** Floating bottom-center HUD anchored with `rounded-xl` shape, containing 100% pixel lock, side-by-side toggle, and diff-matte viewer.

### Segmented Controls & Chip Badges
- **Segmented Control:** Recessed track (`#F1F5F9` light / `#1A2234` dark) with a sliding active segment in elevated pure white, styled with `shadow-sm` and `rounded-lg` radius.
- **Status Pills:** Pill-shaped (`rounded-full`) tags displaying operational state:
  - *Active Inference:* Indigo tint background (`#EEF2FF`), Indigo text (`#4F46E5`), pulsing 6px radial dot.
  - *Converged / Ready:* Emerald tint background (`#ECFDF5`), Emerald text (`#059669`).
  - *Fallback / Overload:* Amber tint background (`#FFFBEB`), Amber text (`#D97706`).

### Input Fields & Sliders
- **Numeric Steppers:** Mono-spaced numeric readout flanked by increment/decrement chevron targets.
- **Denoise / Restoration Sliders:** Flat track height of 4px in `#E2E8F0`, dynamic Indigo filled fill-bar, and a crisp 16px white circular thumb equipped with a 1px border and distinct directional focus ring.