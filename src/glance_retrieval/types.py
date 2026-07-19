from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryPlan:
    raw: str
    fashion_text: str
    context_text: str
    atomic_attributes: tuple[str, ...]
    colors: tuple[str, ...]
    garments: tuple[str, ...]
    contexts: tuple[str, ...]
    styles: tuple[str, ...]
    fashion_weight: float
    context_weight: float

    @property
    def is_compositional(self) -> bool:
        return len(self.atomic_attributes) > 1 or (
            bool(self.atomic_attributes) and bool(self.contexts)
        )


@dataclass(frozen=True)
class SearchResult:
    rank: int
    image_id: str
    path: str
    score: float
    fashion_global: float
    fashion_atomic: float
    context: float
    metadata_bonus: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_path(record_path: str, manifest_path: Path) -> Path:
    path = Path(record_path).expanduser()
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()

