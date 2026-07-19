from __future__ import annotations

import re

from .types import QueryPlan

COLORS = {
    "black", "blue", "brown", "beige", "cream", "gold", "gray", "green", "grey",
    "khaki", "navy", "orange", "pink", "purple", "red", "silver", "tan", "teal",
    "white", "yellow",
}
GARMENTS = {
    "blazer", "blazers", "blouse", "button-down", "coat", "dress", "hoodie", "jacket",
    "jeans", "pants", "raincoat", "shirt", "shorts", "skirt", "suit", "sweater", "t-shirt",
    "tie", "top", "trousers", "vest",
}
CONTEXTS = {
    "airport", "beach", "city", "home", "indoors", "office", "outdoors", "park", "street",
    "studio", "workplace",
}
STYLES = {
    "business", "casual", "classic", "elegant", "formal", "minimalist", "professional",
    "smart-casual", "sporty", "streetwear", "weekend",
}


def _terms(text: str, vocabulary: set[str]) -> tuple[str, ...]:
    found: list[tuple[int, str]] = []
    lowered = text.lower()
    for term in vocabulary:
        match = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered)
        if match:
            canonical = {"grey": "gray", "blazers": "blazer"}.get(term, term)
            found.append((match.start(), canonical))
    unique: list[str] = []
    for _, canonical in sorted(found):
        if canonical not in unique:
            unique.append(canonical)
    return tuple(unique)


def _atomic_attributes(text: str, colors: tuple[str, ...], garments: tuple[str, ...]) -> tuple[str, ...]:
    lowered = re.sub(r"[^a-z0-9\- ]+", " ", text.lower())
    atoms: list[tuple[int, str]] = []
    for color in colors:
        for garment in garments:
            direct = re.search(
                rf"(?<!\w){re.escape(color)}(?:\s+[a-z-]+){{0,1}}\s+{re.escape(garment)}(?!\w)",
                lowered,
            )
            reverse = re.search(
                rf"(?<!\w){re.escape(garment)}\s+(?:in|colored)\s+{re.escape(color)}(?!\w)",
                lowered,
            )
            match = direct or reverse
            if match:
                atoms.append((match.start(), f"{color} {garment}"))
    if not atoms and garments:
        atoms.extend((lowered.find(g), g) for g in garments)
    unique: list[str] = []
    for _, atom in sorted(atoms):
        if atom not in unique:
            unique.append(atom)
    return tuple(unique)


def decompose_query(text: str) -> QueryPlan:
    """Parse only high-confidence visual concepts; keep the raw query as a semantic fallback."""
    colors = _terms(text, COLORS)
    garments = _terms(text, GARMENTS)
    contexts = _terms(text, CONTEXTS)
    styles = _terms(text, STYLES)
    atoms = _atomic_attributes(text, colors, garments)

    fashion_bits = [*atoms, *styles]
    fashion_text = ", ".join(dict.fromkeys(fashion_bits)) or text
    context_text = ", ".join(contexts) if contexts else text

    if contexts and (garments or colors or styles):
        fashion_weight, context_weight = 0.68, 0.32
    elif contexts:
        fashion_weight, context_weight = 0.35, 0.65
    else:
        fashion_weight, context_weight = 0.9, 0.1

    return QueryPlan(
        raw=text,
        fashion_text=fashion_text,
        context_text=context_text,
        atomic_attributes=atoms,
        colors=colors,
        garments=garments,
        contexts=contexts,
        styles=styles,
        fashion_weight=fashion_weight,
        context_weight=context_weight,
    )
