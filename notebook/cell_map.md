# Notebook ↔ Source Module Map

This document maps each function in `src/` to the Colab notebook cell it
was extracted from. The notebook (`stable_diffusion_interior.ipynb`) is
the source of truth; the `src/` modules are a clean, importable
reorganization of the same code.

## Why have both?

The notebook is what was actually executed to produce the results in
`results/`. The `src/` modules package that same logic into a reusable
Python API so future experiments don't require copying cells.

All logic is identical up to cosmetic refactoring (e.g. wrapping
inline model loading into a `load_pipeline()` function).

## Cell map

| `src/` module & function                | Notebook cell | Status |
|------------------------------------------|:-------------:|:------:|
| `room_spec.RoomSpec` + `CATALOG`         | 9             | identical |
| `prompts.naive_prompt`                   | 11            | identical |
| `prompts.structured_prompt`              | 11            | identical |
| `prompts.structured_plus_negative_prompt`| 11            | identical |
| `prompts.semantic_description`           | 23            | identical |
| `prompts.STRATEGIES`                     | 15            | identical |
| `generate.load_pipeline`                 | 5             | refactored inline → function |
| `generate.generate_one`                  | 13            | refactored inline → function |
| `generate.run_generation_suite`          | 15            | identical |
| `evaluate.clip_score`                    | 21            | identical |
| `evaluate.compute_clip_scores`           | 21 + 24       | merged raw + semantic into one function |
| `evaluate.compute_lpips_metrics`         | 26 + 28       | merged consistency + diversity into one function |
| `evaluate.compute_fid`                   | 35            | identical |
| `evaluate.evaluate_all`                  | —             | new convenience wrapper around all of the above |

## Known divergence (intentional)

One intentional change: notebook cell 5 originally specified the model as
`runwayml/stable-diffusion-v1-5`, but that Hugging Face path was
deprecated during the project. The notebook was re-executed with the
community mirror `stable-diffusion-v1-5/stable-diffusion-v1-5` — which
is what actually produced the results in `results/`. The `src/generate.py`
module defaults to the mirror path to match the actual run.

## Reproducibility check

Running either the notebook or the `src/` modules on the same catalog,
seeds (`[42, 123, 2024]`), and reference image set should produce:

- **Byte-identical** outputs for: raw CLIP, semantic CLIP, consistency
  LPIPS, diversity LPIPS (seeded with `random.Random(0)` for the 30-pair
  diversity sample)
- **~2% variance** for: FID (Inception-v3 includes non-deterministic
  ops), but rankings across strategies will be preserved
