"""
Evaluation pipeline computing five metrics across all generated images.

Metrics
-------
- raw_clip              : CLIP similarity against the strategy's full prompt
- semantic_clip         : CLIP similarity against a canonical description,
                          controlling for prompt-length bias
- consistency_lpips     : pairwise LPIPS within same spec, different seeds
                          (lower = more consistent)
- diversity_lpips       : pairwise LPIPS across different specs
                          (higher = more diverse)
- fid                   : Fréchet Inception Distance vs a real interior set
                          (lower = more realistic; noisy at small n)

All metrics assume images live on disk and are addressed via a manifest
produced by src.generate.run_generation_suite.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .room_spec import RoomSpec
from .prompts import semantic_description


# --------------------------------------------------------------------------
# CLIP score — implemented directly with open_clip to avoid the torchmetrics
# wrapper bug with transformers >= 4.40.
# --------------------------------------------------------------------------
def load_clip_model(device: str = "cuda"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.to(device).eval(), preprocess, tokenizer


@torch.no_grad()
def clip_score(model, preprocess, tokenizer, image: Image.Image,
               text: str, device: str = "cuda") -> float:
    """Cosine similarity (×100) between CLIP image and text embeddings."""
    img_t = preprocess(image).unsqueeze(0).to(device)
    txt_t = tokenizer([text]).to(device)
    img_f = model.encode_image(img_t)
    txt_f = model.encode_text(txt_t)
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    return (img_f @ txt_f.T).item() * 100


def compute_clip_scores(manifest: List[dict], images: Dict[str, Image.Image],
                        catalog: List[RoomSpec], device: str = "cuda") -> dict:
    """Return both raw-CLIP and semantic-CLIP scores per strategy."""
    model, preprocess, tokenizer = load_clip_model(device)
    semantic_by_spec = {i: semantic_description(s) for i, s in enumerate(catalog)}

    by_strategy: Dict[str, List[dict]] = {}
    for entry in manifest:
        by_strategy.setdefault(entry["strategy"], []).append(entry)

    results = {"raw_clip": {}, "semantic_clip": {}}
    for strat, entries in by_strategy.items():
        raw_scores = []
        sem_scores = []
        for e in entries:
            img = images[e["filename"]]
            raw_scores.append(clip_score(model, preprocess, tokenizer, img, e["prompt"]))
            sem_scores.append(
                clip_score(model, preprocess, tokenizer, img,
                           semantic_by_spec[e["spec_idx"]])
            )
        results["raw_clip"][strat] = {
            "mean": float(np.mean(raw_scores)),
            "std": float(np.std(raw_scores)),
            "n": len(raw_scores),
        }
        results["semantic_clip"][strat] = {
            "mean": float(np.mean(sem_scores)),
            "std": float(np.std(sem_scores)),
            "n": len(sem_scores),
        }
    return results


# --------------------------------------------------------------------------
# LPIPS — consistency (same spec, different seeds) and diversity (different specs).
# --------------------------------------------------------------------------
def _lpips_tensor(pil_img: Image.Image, device: str = "cuda") -> torch.Tensor:
    return transforms.ToTensor()(pil_img.resize((256, 256))).unsqueeze(0).to(device)


def compute_lpips_metrics(manifest: List[dict], images: Dict[str, Image.Image],
                          device: str = "cuda", diversity_pairs: int = 30) -> dict:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    by_strategy: Dict[str, List[dict]] = {}
    for entry in manifest:
        by_strategy.setdefault(entry["strategy"], []).append(entry)

    out = {"consistency": {}, "diversity": {}}

    # Consistency: within-spec pairs
    for strat, entries in by_strategy.items():
        by_spec: Dict[int, List[dict]] = {}
        for e in entries:
            by_spec.setdefault(e["spec_idx"], []).append(e)
        distances = []
        for spec_entries in by_spec.values():
            for i in range(len(spec_entries)):
                for j in range(i + 1, len(spec_entries)):
                    t1 = _lpips_tensor(images[spec_entries[i]["filename"]], device)
                    t2 = _lpips_tensor(images[spec_entries[j]["filename"]], device)
                    distances.append(lpips(t1, t2).item())
        out["consistency"][strat] = {
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances)),
            "n_pairs": len(distances),
        }

    # Diversity: across-spec pairs (sampled)
    rng = random.Random(0)
    for strat, entries in by_strategy.items():
        distances = []
        attempts = 0
        while len(distances) < diversity_pairs and attempts < diversity_pairs * 5:
            attempts += 1
            e1, e2 = rng.sample(entries, 2)
            if e1["spec_idx"] == e2["spec_idx"]:
                continue
            t1 = _lpips_tensor(images[e1["filename"]], device)
            t2 = _lpips_tensor(images[e2["filename"]], device)
            distances.append(lpips(t1, t2).item())
        out["diversity"][strat] = {
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances)),
            "n_pairs": len(distances),
        }

    return out


# --------------------------------------------------------------------------
# FID vs a reference set of real interior images.
# --------------------------------------------------------------------------
def compute_fid(manifest: List[dict], images: Dict[str, Image.Image],
                reference_dir: Path, device: str = "cuda") -> dict:
    from torchmetrics.image.fid import FrechetInceptionDistance

    reference_dir = Path(reference_dir)
    ref_files = sorted(list(reference_dir.glob("*.jpg")) + list(reference_dir.glob("*.png")))

    fid_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    def to_fid_tensor(pil):
        return fid_transform(pil.convert("RGB")).unsqueeze(0).to(device)

    by_strategy: Dict[str, List[dict]] = {}
    for entry in manifest:
        by_strategy.setdefault(entry["strategy"], []).append(entry)

    results = {}
    for strat, entries in by_strategy.items():
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        for f in ref_files:
            fid.update(to_fid_tensor(Image.open(f)), real=True)
        for e in entries:
            fid.update(to_fid_tensor(images[e["filename"]]), real=False)
        results[strat] = fid.compute().item()
        del fid
        torch.cuda.empty_cache()

    return results


# --------------------------------------------------------------------------
# End-to-end convenience wrapper.
# --------------------------------------------------------------------------
def evaluate_all(manifest_path: Path, generations_dir: Path,
                 reference_dir: Path, catalog: List[RoomSpec],
                 device: str = "cuda") -> dict:
    """Run every metric and return a single nested dict."""
    manifest = json.loads(Path(manifest_path).read_text())
    images = {
        e["filename"]: Image.open(Path(generations_dir) / e["filename"]).convert("RGB")
        for e in manifest
    }

    clip_r = compute_clip_scores(manifest, images, catalog, device)
    lpips_r = compute_lpips_metrics(manifest, images, device)
    fid_r = compute_fid(manifest, images, reference_dir, device)

    return {
        "raw_clip": clip_r["raw_clip"],
        "semantic_clip": clip_r["semantic_clip"],
        "consistency_lpips": lpips_r["consistency"],
        "diversity_lpips": lpips_r["diversity"],
        "fid": fid_r,
    }
