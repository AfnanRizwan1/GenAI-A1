# Interface design (Google Stitch)

The user interface of **Restoration Studio** was designed in Google Stitch before it was implemented in React and Tailwind CSS.

Stitch project: https://stitch.withgoogle.com/projects/10012210425231381214

| File | What it is |
|---|---|
| `STITCH_BRIEF.md` | The first prompt given to Stitch |
| `stitch_v1_*.png`, `stitch_v1_export/` | **Original** Stitch generation (Universal, Hard-Routed, Soft MoE, Face-to-Sketch) with Stitch's own `DESIGN.md` and HTML |
| `DESIGN.md` | The custom design system written for the second round (palette, typography, spacing, components, motion, content rules) |
| `stitch_v2_*.png`, `stitch_v2_export/` | Refined generation from that design system: four workspaces, dark mode and the mobile layout, with the exported HTML |

What changed between v1 and v2: the first generation invented technical details that the application does not have (GPU names, network
architectures, extra sketch settings, scores). The second round removed them, so the design only shows what the application really computes.
The implemented interface follows the v2 layout and design system; where the mockup contained decoration, the application shows measured values.
