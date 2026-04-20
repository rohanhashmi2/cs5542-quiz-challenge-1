# Notebook

This directory contains the end-to-end Colab notebook (`stable_diffusion_interior.ipynb`) that produced every result in `results/` and `grids/`.

It is the **source of truth** for this project. The `src/` package is a clean, importable reorganization of the same code — see [`cell_map.md`](cell_map.md) for a function-by-function traceability map.

## How to run

1. Upload `stable_diffusion_interior.ipynb` to Google Colab.
2. `Runtime → Change runtime type → T4 GPU` (or better; an L4 or A100 works well).
3. Run cells top-to-bottom.

Expected runtime on NVIDIA L4: ~10 minutes end-to-end.

## What each cell does

- Cells 1–4: Dependency install, GPU verification, model load, test image
- Cells 5–13: Define `RoomSpec`, three prompt strategies, visual preview
- Cells 14–17: Run the full 45-image generation suite, build comparison grids
- Cells 18–30: Compute raw CLIP, semantic CLIP, LPIPS consistency & diversity, summary table
- Cells 31–37: Download reference real interior photos, compute FID, final summary
- Cells 38–41: Bundle everything into a downloadable zip

