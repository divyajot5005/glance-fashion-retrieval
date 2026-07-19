from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from .models import HuggingFaceEncoder
from .query import decompose_query
from .storage import VectorStore
from .types import SearchResult

RetrievalVariant = Literal[
    "fashion_global", "context_only", "dual_global", "multi_crop_mean", "proposed"
]
VARIANTS: tuple[RetrievalVariant, ...] = (
    "fashion_global",
    "context_only",
    "dual_global",
    "multi_crop_mean",
    "proposed",
)


def softmin(values: np.ndarray, temperature: float, axis: int = -1) -> np.ndarray:
    """Stable differentiable approximation of min; values are cosine similarities."""
    scaled = -values / temperature
    maximum = scaled.max(axis=axis, keepdims=True)
    lme = maximum + np.log(np.exp(scaled - maximum).mean(axis=axis, keepdims=True))
    return (-temperature * lme.squeeze(axis)).astype(np.float32)


def metadata_bonus(metadata: dict, plan) -> float:
    haystack = " ".join(str(value).lower() for value in metadata.values())
    concepts = (*plan.colors, *plan.garments, *plan.contexts, *plan.styles)
    if not concepts:
        return 0.0
    return sum(term in haystack for term in concepts) / len(concepts)


class Retriever:
    def __init__(self, config: dict, index_dir: Path, encoders: tuple | None = None):
        self.config = config
        self.index_dir = Path(index_dir)
        self.records, self.fashion, self.context, self.index_meta = VectorStore(index_dir).load()
        self.faiss_indices = None
        if config["index"].get("backend") == "faiss":
            try:
                import faiss

                self.faiss_indices = (
                    faiss.read_index(str(self.index_dir / "fashion.faiss")),
                    faiss.read_index(str(self.index_dir / "context.faiss")),
                )
            except (ImportError, RuntimeError) as exc:
                raise RuntimeError(
                    "FAISS backend requested but indices or faiss-cpu are unavailable"
                ) from exc
        if encoders is None:
            encoders = (
                HuggingFaceEncoder(
                    config["models"]["fashion"],
                    config["runtime"]["device"],
                    config["runtime"]["use_amp"],
                ),
                HuggingFaceEncoder(
                    config["models"]["context"],
                    config["runtime"]["device"],
                    config["runtime"]["use_amp"],
                ),
            )
        self.fashion_encoder, self.context_encoder = encoders

    def search(
        self, query: str, top_k: int = 5, variant: RetrievalVariant = "proposed"
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if variant not in VARIANTS:
            raise ValueError(f"Unknown retrieval variant: {variant}. Choose from {VARIANTS}")
        plan = decompose_query(query)
        fashion_query = self.fashion_encoder.encode_texts([plan.fashion_text])[0]
        context_query = self.context_encoder.encode_texts([plan.context_text])[0]
        fashion_global = np.asarray(self.fashion[:, 0]) @ fashion_query
        context_score = np.asarray(self.context) @ context_query

        pool_size = min(
            len(self.records), max(top_k, int(self.config["index"]["candidate_pool"]))
        )
        shortlist_score = plan.fashion_weight * fashion_global + plan.context_weight * context_score
        if self.faiss_indices is not None:
            fashion_hits = self.faiss_indices[0].search(
                np.ascontiguousarray(fashion_query[None], dtype=np.float32), pool_size
            )[1][0]
            context_hits = self.faiss_indices[1].search(
                np.ascontiguousarray(context_query[None], dtype=np.float32), pool_size
            )[1][0]
            candidates = np.unique(np.concatenate([fashion_hits, context_hits]))
            candidates = candidates[candidates >= 0]
        else:
            candidates = np.argpartition(shortlist_score, -pool_size)[-pool_size:]

        atoms = plan.atomic_attributes or (plan.fashion_text,)
        atom_embeddings = self.fashion_encoder.encode_texts(list(atoms))
        crop_scores = np.einsum(
            "nvd,ad->nva", np.asarray(self.fashion[candidates]), atom_embeddings
        )
        per_atom = crop_scores.max(axis=1)
        if variant == "multi_crop_mean":
            fashion_atomic = per_atom.mean(axis=1).astype(np.float32)
        elif variant == "proposed":
            fashion_atomic = softmin(
                per_atom, float(self.config["retrieval"]["softmin_temperature"]), axis=1
            )
        else:
            fashion_atomic = fashion_global[candidates].astype(np.float32)

        fg_weight = float(self.config["retrieval"]["fashion_global_weight"])
        fa_weight = float(self.config["retrieval"]["fashion_atomic_weight"])
        bonuses = np.asarray(
            [metadata_bonus(self.records[index].metadata, plan) for index in candidates],
            dtype=np.float32,
        )
        meta_weight = float(self.config["retrieval"].get("metadata_boost", 0.0))
        if variant == "fashion_global":
            final = fashion_global[candidates]
        elif variant == "context_only":
            final = context_score[candidates]
        elif variant == "dual_global":
            final = (
                plan.fashion_weight * fashion_global[candidates]
                + plan.context_weight * context_score[candidates]
            )
        else:
            final = (
                plan.fashion_weight
                * (fg_weight * fashion_global[candidates] + fa_weight * fashion_atomic)
                + plan.context_weight * context_score[candidates]
                + meta_weight * bonuses
            )
        order = np.argsort(-final)[:top_k]
        results = []
        for rank, local_index in enumerate(order, 1):
            index = int(candidates[local_index])
            record = self.records[index]
            results.append(
                SearchResult(
                    rank=rank,
                    image_id=record.image_id,
                    path=record.path,
                    score=float(final[local_index]),
                    fashion_global=float(fashion_global[index]),
                    fashion_atomic=float(fashion_atomic[local_index]),
                    context=float(context_score[index]),
                    metadata_bonus=float(bonuses[local_index]),
                    metadata=record.metadata,
                )
            )
        return results

    @staticmethod
    def as_json(results: list[SearchResult]) -> str:
        return json.dumps([result.to_dict() for result in results], indent=2)
