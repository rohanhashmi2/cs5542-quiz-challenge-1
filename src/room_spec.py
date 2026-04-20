"""
Room specification schema and the five-entry catalog used throughout the
CS 5542 Quiz Challenge 1 interior-design generation pipeline.

A RoomSpec captures the fields a real interior design platform (IKEA, Houzz,
Havenly, etc.) would receive from a user. The same spec feeds all three
prompt strategies, so any difference in generated output is attributable to
prompt engineering rather than input.
"""

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class RoomSpec:
    """Structured metadata describing an interior design brief."""

    room_type: str            # e.g. "living room", "bedroom", "kitchen"
    style: str                # e.g. "scandinavian", "japandi", "industrial"
    materials: List[str]      # e.g. ["oak wood floor", "linen upholstery"]
    lighting: str             # e.g. "natural window light"
    key_furniture: List[str]  # e.g. ["low sofa", "coffee table"]
    mood: str                 # e.g. "minimalist and cozy"
    color_palette: str        # e.g. "neutral tones with soft beige accents"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Catalog of 5 diverse room specifications.
# Chosen to span different rooms, styles, and style rarity — some styles are
# well-represented in SD training data (farmhouse kitchen) while others are
# fused terms that challenge the model (japandi, mid-century modern).
# --------------------------------------------------------------------------
CATALOG: List[RoomSpec] = [
    RoomSpec(
        room_type="living room",
        style="scandinavian",
        materials=["oak wood floor", "white walls", "linen upholstery"],
        lighting="natural window light",
        key_furniture=["low sofa", "wooden coffee table", "floor lamp"],
        mood="minimalist and cozy",
        color_palette="neutral tones with soft beige accents",
    ),
    RoomSpec(
        room_type="bedroom",
        style="japandi",
        materials=["light oak flooring", "cream walls", "linen bedding"],
        lighting="soft morning light",
        key_furniture=["low platform bed", "wooden nightstand", "paper lamp"],
        mood="serene and calm",
        color_palette="warm neutrals with muted sage accents",
    ),
    RoomSpec(
        room_type="home office",
        style="industrial",
        materials=["polished concrete floor", "exposed brick wall", "steel shelving"],
        lighting="pendant lighting with warm bulbs",
        key_furniture=["reclaimed wood desk", "leather chair", "metal bookshelf"],
        mood="focused and masculine",
        color_palette="dark grey and warm wood tones",
    ),
    RoomSpec(
        room_type="kitchen",
        style="modern farmhouse",
        materials=["white subway tile", "butcher block counter", "hardwood floor"],
        lighting="bright natural light from large windows",
        key_furniture=["kitchen island", "open shelving", "bar stools"],
        mood="warm and inviting",
        color_palette="white with warm wood accents",
    ),
    RoomSpec(
        room_type="dining room",
        style="mid-century modern",
        materials=["walnut wood floor", "textured plaster walls", "brass fixtures"],
        lighting="statement pendant light",
        key_furniture=["walnut dining table", "upholstered chairs", "sideboard"],
        mood="elegant and retro",
        color_palette="warm woods with mustard and teal accents",
    ),
]
