from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .types import ImageRecord


class VectorStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def save(
        self,
        records: list[ImageRecord],
        fashion: np.ndarray,
        context: np.ndarray,
        metadata: dict,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.save(self.root / "fashion.npy", fashion.astype(np.float32))
        np.save(self.root / "context.npy", context.astype(np.float32))
        with (self.root / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        (self.root / "index_meta.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def load(self) -> tuple[list[ImageRecord], np.ndarray, np.ndarray, dict]:
        records = [
            ImageRecord(**json.loads(line))
            for line in (self.root / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        fashion = np.load(self.root / "fashion.npy", mmap_mode="r")
        context = np.load(self.root / "context.npy", mmap_mode="r")
        metadata = json.loads((self.root / "index_meta.json").read_text(encoding="utf-8"))
        if not (len(records) == fashion.shape[0] == context.shape[0]):
            raise ValueError("Corrupt index: record and embedding counts differ")
        return records, fashion, context, metadata


def write_faiss_hnsw(root: Path, fashion_global: np.ndarray, context: np.ndarray, m: int) -> None:
    import faiss

    root = Path(root)
    for name, vectors in (("fashion", fashion_global), ("context", context)):
        index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 100
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        faiss.write_index(index, str(root / f"{name}.faiss"))

