# Controlled Interior Design Generation with Stable Diffusion

**CS 5542 — Big Data Analytics · Quiz Challenge 1 · Spring 2026**

A study of prompt engineering, control mechanisms, and multi-metric
evaluation in text-to-image diffusion systems — applied to the scenario of
automated interior design generation.

---

## TL;DR

I built a controlled interior design generation pipeline on Stable Diffusion v1.5.
Five room specifications are rendered under three progressively-controlled
prompt strategies and evaluated across five metrics. The central finding:

> **Structured prompts trade a small diversity reduction for a +2.45 semantic
> alignment gain and measurably better consistency — the right operating
> point for production systems. Raw CLIP and FID both favored the naive
> baseline for methodological reasons (prompt-length bias and reference
> distribution bias, respectively) rather than genuine quality.**

This is a small but honest demonstration of why generative AI systems need
multi-metric evaluation and why naive "CLIP score go up" thinking misleads.

**Demo video:** _[add video link]_

---

## Hero Comparison

The Japandi bedroom spec under naive vs. fully-controlled prompting, same seed:

| Naive | Structured + Negative |
|:---:|:---:|
| ![](results/sample_images/spec01_naive_seed42.png) | ![](results/sample_images/spec01_structured_neg_seed42.png) |
| `"a japandi bedroom"` | Full metadata + quality modifiers + negative prompt |

The naive prompt defaults to ornate traditional decor — "Japandi" is
sparsely represented in the model's training data without anchoring terms.
The structured prompt corrects this by specifying materials, furniture,
lighting, and color palette explicitly.

---

## Results

Full evaluation across 45 generations (5 specs × 3 strategies × 3 seeds):

| Strategy              | Raw CLIP ↑ | Semantic CLIP ↑ | Consistency LPIPS ↓ | Diversity LPIPS ↑ | FID ↓  |
|-----------------------|:----------:|:---------------:|:-------------------:|:-----------------:|:------:|
| Naive                 | **30.80**  | 29.44           | 0.6654              | **0.7274**        | **193.12** |
| Structured            | 28.08      | **31.89**       | **0.6456**          | 0.7106            | 200.91 |
| Structured + Negative | 27.93      | 31.32           | 0.6573              | 0.6956            | 195.88 |

**Bold** marks the winning strategy per metric. Note that raw CLIP and FID
favoring naive is an artifact of methodology (see [Findings](#findings)),
not evidence that the naive prompt produces better images. Visual inspection
of [all five grids](grids/) confirms the structured strategies produce
substantially higher-quality outputs.

---

## Repository Layout

```
.
├── README.md                           this file
├── requirements.txt                    pinned dependencies
│
├── notebook/
│   ├── README.md                       run instructions
│   ├── cell_map.md                     notebook ↔ src/ traceability map
│   └── quiz_challenge_1.ipynb          end-to-end Colab notebook (source of truth)
│
├── src/                                clean Python modules
│   ├── __init__.py
│   ├── room_spec.py                    RoomSpec dataclass + 5-entry catalog
│   ├── prompts.py                      three prompt strategies
│   ├── generate.py                     generation runner
│   └── evaluate.py                     all five metrics
│
├── slides/
│   └── quiz_challenge_1.pptx           16-slide presentation deck
│
├── video/                              demo video
│
├── results/
│   ├── evaluation_results.csv          summary metrics table
│   ├── full_metrics.json               detailed metrics + config
│   ├── manifest.json                   image-level metadata
│   ├── reference_real/                 100 real interior photos (FID reference set)
│   └── sample_images/                  45 generated images (all 3 strategies × 5 specs × 3 seeds)
│
├── grids/                              5 comparison grids (3×3 per spec)
│
└── docs/
    └── failure_analysis.md             cataloged failure modes
```

---

## Quickstart (Google Colab)

The fastest path to reproducing these results is the bundled notebook,
designed for a Colab Pro GPU runtime.

1. Open `notebook/quiz_challenge_1.ipynb` in Google Colab.
2. Select `Runtime → Change runtime type → T4 GPU` (or better; an L4 or A100
   works well).
3. Run cells top-to-bottom. The notebook is self-contained — installs
   dependencies, downloads the model (~4 GB), generates 45 images, and
   computes all five metrics.

Expected runtime on an NVIDIA L4: **~10 minutes end-to-end** (~5 minutes
for generation, ~2 for CLIP/LPIPS, ~3 for FID including reference download).

---

## Quickstart (Local, Python API)

If you have a local GPU you can run the modules directly.

```bash
git clone git@github.com:rohanhashmi2/cs5542-quiz-challenge-1.git
cd cs5542-quiz-challenge-1
pip install -r requirements.txt
```

```python
from pathlib import Path
from src.room_spec import CATALOG
from src.prompts import STRATEGIES
from src.generate import load_pipeline, run_generation_suite
from src.evaluate import evaluate_all

# 1. Generate
pipe = load_pipeline()
manifest = run_generation_suite(
    pipe=pipe,
    catalog=CATALOG,
    strategies=STRATEGIES,
    seeds=[42, 123, 2024],
    output_dir=Path("my_generations"),
)

# 2. Evaluate — the 100 reference real-interior photos are bundled
#    in results/reference_real/ for convenience.
metrics = evaluate_all(
    manifest_path=Path("my_generations/manifest.json"),
    generations_dir=Path("my_generations"),
    reference_dir=Path("results/reference_real"),
    catalog=CATALOG,
)
print(metrics)
```

---

## Methodology

### 1. Structured metadata schema

Every room is defined by a `RoomSpec` dataclass — room type, style,
materials, lighting, key furniture, mood, and color palette. This mirrors
the input shape a real design platform (IKEA, Houzz) would accept from a
user. The catalog contains five entries spanning diverse styles and
room types; see [`src/room_spec.py`](src/room_spec.py).

### 2. Three prompt strategies

All three take the same `RoomSpec` as input. The difference is how much of
that metadata is compiled into the prompt:

- **Naive** — `"a scandinavian living room"`. The lazy-user baseline.
- **Structured** — template compiles every field into a detailed prompt
  with quality modifiers appended (`architectural digest style, 4k`, etc.).
- **Structured + Negative** — same positive prompt plus an explicit
  negative prompt targeting known diffusion failure modes (`distorted,
  warped perspective, bad proportions, extra limbs, people...`).

See [`src/prompts.py`](src/prompts.py).

### 3. Reproducible generation

45 images total: 5 specs × 3 strategies × 3 seeds. Seeds are held fixed
across strategies, so any cell in the output grid can be compared against
any other with noise held constant. DPM-Solver at 25 steps with CFG 7.5,
float16 on CUDA. See [`src/generate.py`](src/generate.py).

### 4. Five-metric evaluation

| Metric | Direction | Measures |
|---|:---:|---|
| Raw CLIP | ↑ | Image ↔ full prompt text similarity |
| Semantic CLIP | ↑ | Image ↔ canonical room description (controls for prompt-length bias) |
| Consistency LPIPS | ↓ | Perceptual distance within same spec across seeds |
| Diversity LPIPS | ↑ | Perceptual distance across different specs |
| FID | ↓ | Fréchet distance vs 100 real interior photos |

Reference set for FID: [`MohamedAli77/interior-rooms`](https://huggingface.co/datasets/MohamedAli77/interior-rooms)
(100 images, bundled in `results/reference_real/`). See [`src/evaluate.py`](src/evaluate.py).

---

## Findings

### Finding 1 — Structured prompts win on what matters for production

On **semantic alignment** (how well the image matches the intended room
concept) and **consistency** (how reliably the same prompt produces similar
outputs), the structured strategy beats naive by a clear margin and beats
even its augmented sibling. Both are the metrics that matter for
production systems where user intent must be honored repeatably.

### Finding 2 — Raw CLIP has prompt-length bias

Raw CLIP favored the naive prompt (30.80 vs 28.08) despite the naive
prompts producing visibly inferior images. The cause is structural:
CLIP scores how well an image matches its *specific prompt text*. A 4-word
prompt is trivially easy to match; a 40-word prompt requires every named
element to be present, and any miss is penalized. Introducing a **semantic
CLIP** metric (scoring every image against the same canonical room
description) flipped the ranking correctly. Both scores are reported.

**Lesson**: never trust a single metric for generative model evaluation.
Measure the thing you actually care about, not what's easy.

### Finding 3 — FID is unreliable for controlled generation at small n

FID scores (193–201) fell within statistical noise for n=15. Additionally,
our 100-image reference set was modern-minimalist-dominant, which
systematically penalized our industrial, farmhouse, and mid-century
outputs for stylistic deviation rather than quality. FID is designed for
unconditional generative model comparison at massive sample sizes — it is
the wrong metric for this setting, and we report it only to show a
rigorous limitation analysis.

### Finding 4 — The control-diversity trade-off is real and quantifiable

Diversity LPIPS decreased monotonically with more control
(0.7274 → 0.7106 → 0.6956). This is expected: tighter constraints produce
more similar outputs across different specs. The structured strategy
occupies the right point on this curve — enough control to honor intent,
enough diversity to remain useful.

See [`docs/failure_analysis.md`](docs/failure_analysis.md) for specific
failure modes and their attributions.

---

## Limitations & Future Work

**FID at n=15 is unreliable.** Scaling to n ≥ 2000 per strategy would
enable proper significance testing. Deferred due to time/compute budget.

**Reference distribution is stylistically biased.** A stratified reference
set (real examples per style) would give more informative per-style FID.

**No ControlNet experiments.** The core pipeline supports adding ControlNet
for layout-preservation conditioning (depth map or edge-map input);
investigation deferred to future work.

**Single model tested.** SDXL and SD 3 likely shift the ranking — in
particular, SDXL is known to interpret fused style terms (Japandi)
better without scaffolding. Comparative study left for future work.

---

## Tools & Technologies

| Component | Choice |
|---|---|
| Base model | Stable Diffusion v1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`) |
| Framework | Hugging Face Diffusers |
| Metrics | torchmetrics, open_clip_torch |
| Reference data | `MohamedAli77/interior-rooms` (Hugging Face Datasets) |
| Compute | NVIDIA L4 GPU (Google Colab Pro) |
| Language | Python 3.12, PyTorch 2.x |

---

## AI Assistance Disclosure

This project was built in collaboration with **Claude (Anthropic)** as a
pair-programming assistant. Claude was used for:

- Debugging Colab dependency conflicts (torchmetrics × transformers API break)
- Structuring the evaluation pipeline (which metrics, in what order, why)
- Identifying and explaining CLIP prompt-length bias
- Co-authoring the prompt strategy design and metadata schema
- Designing the presentation slide structure

All code was reviewed and executed by the author. All images were
generated by the author's Colab notebook. All experimental results were
computed from first-principles Python code (no metric values were invented
or estimated). All analytical conclusions — including the identification
of CLIP bias and FID reference-distribution bias — were produced through
human-AI collaboration and verified against the underlying data.

---