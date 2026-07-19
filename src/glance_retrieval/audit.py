from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .manifest import load_manifest
from .types import resolve_path


def difference_hash(image: Image.Image, size: int = 9) -> str:
    gray = image.convert("L").resize((size, size - 1))
    pixels = list(gray.get_flattened_data())
    bits = []
    for row in range(size - 1):
        offset = row * size
        bits.extend(pixels[offset + col] > pixels[offset + col + 1] for col in range(size - 1))
    value = sum(int(bit) << idx for idx, bit in enumerate(bits))
    return f"{value:0{((size - 1) ** 2 + 3) // 4}x}"


def audit_manifest(manifest_path: Path, expected_minimum: int = 500) -> dict:
    records = load_manifest(manifest_path)
    missing: list[str] = []
    unreadable: list[dict] = []
    dimensions: list[tuple[int, int]] = []
    formats: Counter[str] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    metadata_keys: Counter[str] = Counter()

    for record in records:
        metadata_keys.update(record.metadata.keys())
        path = resolve_path(record.path, manifest_path)
        if not path.exists():
            missing.append(record.image_id)
            continue
        try:
            with Image.open(path) as image:
                image.load()
                dimensions.append(image.size)
                formats.update([image.format or "unknown"])
                hashes[difference_hash(image)].append(record.image_id)
        except (OSError, UnidentifiedImageError) as exc:
            unreadable.append({"image_id": record.image_id, "error": str(exc)})

    duplicate_groups = [ids for ids in hashes.values() if len(ids) > 1]
    widths = [width for width, _ in dimensions]
    heights = [height for _, height in dimensions]
    warnings = []
    if len(records) < expected_minimum:
        warnings.append(f"Collection has {len(records)} images; expected at least {expected_minimum}.")
    if missing:
        warnings.append(f"{len(missing)} image paths are missing.")
    if unreadable:
        warnings.append(f"{len(unreadable)} images are unreadable.")
    if duplicate_groups:
        warnings.append(f"{len(duplicate_groups)} exact dHash duplicate groups detected.")

    return {
        "manifest": str(manifest_path),
        "records": len(records),
        "valid_images": len(dimensions),
        "missing_ids": missing,
        "unreadable": unreadable,
        "duplicate_groups": duplicate_groups,
        "formats": dict(formats),
        "resolution": {
            "min_width": min(widths) if widths else None,
            "max_width": max(widths) if widths else None,
            "min_height": min(heights) if heights else None,
            "max_height": max(heights) if heights else None,
        },
        "metadata_coverage": {
            key: count / len(records) for key, count in sorted(metadata_keys.items())
        },
        "warnings": warnings,
        "passed": not (missing or unreadable or duplicate_groups)
        and len(records) >= expected_minimum,
    }


def audit_json(manifest_path: Path, expected_minimum: int = 500) -> str:
    return json.dumps(audit_manifest(manifest_path, expected_minimum), indent=2)
