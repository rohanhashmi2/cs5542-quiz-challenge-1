# Failure Analysis

This document catalogs specific failure modes observed in the generations
and attributes each to a likely cause. Failure analysis is a required
component of the evaluation — and more importantly, it produces the
analytical insight that distinguishes a working system from an understood one.

## Failure 1 — Cultural misinterpretation (naive prompt, fused style term)

**Example**: `spec01_naive_seed123.png` — Japandi bedroom, naive prompt

**Observation**: Ornate traditional decor with saturated magenta bedding
and decorative wall tapestries. Not Japandi at all.

**Attribution**: "Japandi" is a fused style (Japanese + Scandinavian) that
is sparsely represented in Stable Diffusion v1.5 training data. Without
anchoring terms (`light oak flooring`, `cream walls`, `linen bedding`,
`low platform bed`), the model defaults to visually louder traditional
Asian or bohemian interpretations.

**Mitigation**: The structured prompt for the same spec produces
correctly-styled japandi bedrooms across all three seeds. **This is the
strongest single argument in the results for metadata-driven prompting.**

---

## Failure 2 — Warped furniture geometry (structured, no negatives)

**Example**: `spec00_structured_seed42.png` — Scandinavian living room

**Observation**: The coffee table's legs merge into an impossible
geometric form; proportions are off.

**Attribution**: Diffusion models are well known to produce local
anatomical or geometric errors, especially on secondary objects (furniture
that isn't the focal point). The structured prompt does not explicitly
forbid these failure modes.

**Mitigation**: Adding `distorted, bad proportions, warped perspective`
to the negative prompt fixes this cell in `spec00_structured_neg_seed42.png`
without changing overall composition. This is the clearest single-image
argument for the value of negative prompts in the results.

---

## Failure 3 — Style drift without structural anchors (naive)

**Example**: `spec02_naive_seed2024.png` — Industrial home office, naive prompt

**Observation**: A generic white-desk home office with a wicker bistro-style
chair and white shelving. None of the industrial markers — exposed brick,
polished concrete, reclaimed wood, leather chair — are present. Also note
that the naive prompt renders as `"a industrial home office"`: ungrammatical
English the template does not correct for.

**Attribution**: `"a industrial home office"` offers the model two weak
signals (room type, adjective) and no structural or material anchors. The
model falls back to the most common "home office" visual prior, which in
SD v1.5's training distribution skews toward bright, modern, Pinterest-style
rooms rather than industrial lofts.

**Mitigation**: The structured prompt (which specifies `polished concrete
floor`, `exposed brick wall`, `reclaimed wood desk`, `leather chair`)
produces correctly-styled industrial offices across all seeds. Anchor
nouns do most of the work; the quality modifiers are secondary.

---

## Failure 4 — Reference distribution bias (FID methodology)

**Observation**: FID scores (193–201) are uninformative across strategies;
all three fall within statistical noise for n=15.

**Attribution**:

1. FID requires ≥ 2000 samples for stable ranking. This project used 15
   per strategy due to compute and time constraints.
2. The 100 reference images were sourced from a luxury minimalist interior
   studio's portfolio. The generations include styles (industrial, mid-century
   modern, farmhouse) that are stylistically distant from this reference
   distribution — so FID penalizes stylistic diversity rather than realism.

**Mitigation / lesson**: FID is the wrong metric for evaluating
controlled generation where stylistic diversity is intended. Future work
should either (a) use style-stratified reference sets per spec, or
(b) substitute perceptual quality metrics that don't require a reference
distribution (e.g., learned aesthetic scorers).

---

## Failure 5 — Prompt-length bias in raw CLIP score

**Observation**: Raw CLIP favored naive prompts (30.80) over structured
prompts (28.08) — the opposite of the visual ranking.

**Attribution**: CLIP score measures how well an image matches its
specific prompt text. A 4-word prompt is trivially easy to satisfy. A
40-word prompt requires every named element to be visible; any miss
penalizes the score. This is prompt-length bias, a well-documented CLIP
limitation.

**Mitigation**: I introduced a "semantic CLIP" metric that scores every
image against the same canonical room description, stripping out
prompt-engineering filler. The ranking flipped correctly:
structured (31.89) > structured+neg (31.32) > naive (29.44). Both scores
are reported in the final evaluation for transparency.