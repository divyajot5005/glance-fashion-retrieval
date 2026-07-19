from __future__ import annotations

import json
from pathlib import Path

from .types import ImageRecord


def load_manifest(path: Path) -> list[ImageRecord]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if "image_id" not in raw or "path" not in raw:
            raise ValueError(f"Manifest line {line_number} needs image_id and path")
        raw.setdefault("metadata", {})
        records.append(ImageRecord(**raw))
    if not records:
        raise ValueError("Manifest is empty")
    if len({record.image_id for record in records}) != len(records):
        raise ValueError("image_id values must be unique")
    return records

