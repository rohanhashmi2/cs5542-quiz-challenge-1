"""
Three prompt generation strategies, progressively more controlled.

The experimental design holds the input (a RoomSpec) constant and varies
only the prompt strategy. This isolates the effect of prompt engineering
from any other source of variation.

Strategies
----------
- naive                      : minimal information baseline
- structured                 : full metadata compiled via template
- structured_plus_negative   : structured + negative prompt steering

All three return a (positive_prompt, negative_prompt) tuple. The negative
prompt is an empty string for strategies that do not use one.
"""

from typing import Tuple

from .room_spec import RoomSpec


def naive_prompt(spec: RoomSpec) -> Tuple[str, str]:
    """Baseline: minimal information — what a casual user would type."""
    return f"a {spec.style} {spec.room_type}", ""


def structured_prompt(spec: RoomSpec) -> Tuple[str, str]:
    """Full metadata compiled via a template with quality modifiers appended."""
    materials = ", ".join(spec.materials)
    furniture = ", ".join(spec.key_furniture)
    prompt = (
        f"{spec.style} {spec.room_type}, "
        f"{materials}, "
        f"{furniture}, "
        f"{spec.lighting}, "
        f"{spec.mood}, "
        f"{spec.color_palette}, "
        f"interior design photography, architectural digest style, "
        f"highly detailed, 4k, professional lighting"
    )
    return prompt, ""


def structured_plus_negative_prompt(spec: RoomSpec) -> Tuple[str, str]:
    """Structured + negative prompt steering the model away from failure modes."""
    prompt, _ = structured_prompt(spec)
    negative = (
        "blurry, distorted, low quality, cartoon, illustration, painting, "
        "warped perspective, fisheye, cluttered, messy, ugly, amateur, "
        "bad proportions, oversaturated, unrealistic, watermark, text, logo, "
        "people, person, human"
    )
    return prompt, negative


# --------------------------------------------------------------------------
# Canonical semantic description — used for "fair CLIP" evaluation.
# Strips out prompt-engineering filler so every strategy is scored against
# the same room concept, controlling for CLIP's prompt-length bias.
# --------------------------------------------------------------------------
def semantic_description(spec: RoomSpec) -> str:
    """Return the canonical semantic content of the spec — no quality modifiers."""
    materials = ", ".join(spec.materials)
    furniture = ", ".join(spec.key_furniture)
    return (
        f"{spec.style} {spec.room_type} with "
        f"{materials}, {furniture}, "
        f"{spec.lighting}, {spec.color_palette}"
    )


# Strategy registry — useful for iteration in generate.py
STRATEGIES = {
    "naive": naive_prompt,
    "structured": structured_prompt,
    "structured_neg": structured_plus_negative_prompt,
}
