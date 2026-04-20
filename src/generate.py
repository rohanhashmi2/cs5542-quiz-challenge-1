"""
Generation runner: for each (spec, strategy, seed) triple, produce an image.

This is the core experimental loop. Results are saved to disk with
structured filenames and a manifest.json that ties each file to its
generation parameters — essential for reproducibility and downstream
evaluation.

Usage (from a notebook or script)
---------------------------------
    from src.generate import load_pipeline, run_generation_suite
    from src.room_spec import CATALOG
    from src.prompts import STRATEGIES

    pipe = load_pipeline()
    manifest = run_generation_suite(
        pipe=pipe,
        catalog=CATALOG,
        strategies=STRATEGIES,
        seeds=[42, 123, 2024],
        output_dir=Path("generations"),
    )
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from PIL import Image
from tqdm.auto import tqdm

from .room_spec import RoomSpec


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_SEEDS = [42, 123, 2024]


def load_pipeline(model_id: str = MODEL_ID, device: str = "cuda"):
    """Load Stable Diffusion v1.5 with DPM-Solver scheduler.

    Uses float16 for lower VRAM footprint and faster inference on
    consumer GPUs. Safety checker is disabled because interior design
    generation poses no NSFW risk and the checker consumes ~1 GB VRAM.
    """
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def generate_one(
    pipe,
    prompt: str,
    negative: str,
    seed: int,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    device: str = "cuda",
) -> Image.Image:
    """Generate a single image from a prompt and seed."""
    return pipe(
        prompt=prompt,
        negative_prompt=negative if negative else None,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device).manual_seed(seed),
    ).images[0]


def run_generation_suite(
    pipe,
    catalog: List[RoomSpec],
    strategies: Dict[str, Callable[[RoomSpec], Tuple[str, str]]],
    seeds: List[int],
    output_dir: Path,
) -> List[dict]:
    """Generate every (spec, strategy, seed) combination.

    Writes one PNG per combination plus a manifest.json keyed by filename.
    Returns the manifest as a list of dicts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[dict] = []
    total = len(catalog) * len(strategies) * len(seeds)
    pbar = tqdm(total=total, desc="Generating")

    for spec_idx, spec in enumerate(catalog):
        for strat_name, strat_fn in strategies.items():
            prompt, negative = strat_fn(spec)
            for seed in seeds:
                filename = f"spec{spec_idx:02d}_{strat_name}_seed{seed}.png"
                img = generate_one(pipe, prompt, negative, seed)
                img.save(output_dir / filename)

                manifest.append({
                    "spec_idx": spec_idx,
                    "room_type": spec.room_type,
                    "style": spec.style,
                    "strategy": strat_name,
                    "seed": seed,
                    "prompt": prompt,
                    "negative_prompt": negative,
                    "filename": filename,
                })
                pbar.update(1)

    pbar.close()

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
