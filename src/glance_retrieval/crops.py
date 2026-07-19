from __future__ import annotations

from collections.abc import Iterable

from PIL import Image


def make_views(image: Image.Image, names: Iterable[str]) -> list[Image.Image]:
    """Deterministic body-region views; cheap, detector-free, and reproducible."""
    image = image.convert("RGB")
    width, height = image.size
    boxes = {
        "full": (0, 0, width, height),
        "upper": (0, 0, width, max(1, int(height * 0.68))),
        "lower": (0, int(height * 0.32), width, height),
        "center": (int(width * 0.15), int(height * 0.1), int(width * 0.85), int(height * 0.9)),
        "wide_center": (0, int(height * 0.18), width, int(height * 0.82)),
    }
    unknown = [name for name in names if name not in boxes]
    if unknown:
        raise ValueError(f"Unknown crop views: {unknown}")
    return [image.crop(boxes[name]) for name in names]

