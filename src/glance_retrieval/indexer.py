from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .crops import make_views
from .manifest import load_manifest
from .models import HuggingFaceEncoder
from .storage import VectorStore, write_faiss_hnsw
from .types import ImageRecord, resolve_path


def build_index(config: dict, manifest_path: Path, output_dir: Path) -> None:
    records = [
        ImageRecord(
            image_id=record.image_id,
            path=str(resolve_path(record.path, manifest_path)),
            metadata=record.metadata,
        )
        for record in load_manifest(manifest_path)
    ]
    fashion_encoder = HuggingFaceEncoder(
        config["models"]["fashion"],
        config["runtime"]["device"],
        config["runtime"]["use_amp"],
    )
    context_encoder = HuggingFaceEncoder(
        config["models"]["context"],
        config["runtime"]["device"],
        config["runtime"]["use_amp"],
    )
    views = config["index"]["crop_views"]
    batch_size = int(config["runtime"]["batch_size"])
    fashion_chunks: list[np.ndarray] = []
    context_chunks: list[np.ndarray] = []

    for start in tqdm(range(0, len(records), batch_size), desc="Indexing"):
        batch = records[start : start + batch_size]
        images = []
        for record in batch:
            with Image.open(record.path) as source:
                images.append(source.convert("RGB"))
        flattened = [view for image in images for view in make_views(image, views)]
        fashion = fashion_encoder.encode_images(flattened)
        fashion_chunks.append(fashion.reshape(len(batch), len(views), -1))
        context_chunks.append(context_encoder.encode_images(images))

    fashion_matrix = np.concatenate(fashion_chunks)
    context_matrix = np.concatenate(context_chunks)
    meta = {
        "fashion_model": config["models"]["fashion"],
        "context_model": config["models"]["context"],
        "crop_views": views,
        "count": len(records),
    }
    VectorStore(output_dir).save(records, fashion_matrix, context_matrix, meta)
    if config["index"].get("backend") == "faiss":
        write_faiss_hnsw(
            output_dir, fashion_matrix[:, 0], context_matrix, config["index"]["faiss_hnsw_m"]
        )
