from __future__ import annotations

import json
import math
from pathlib import Path

from .retriever import VARIANTS


def evaluate(
    retriever,
    judgments_path: Path,
    ks: tuple[int, ...] = (1, 5, 10),
    variant: str = "proposed",
) -> dict:
    cases = [
        json.loads(line)
        for line in judgments_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("No evaluation cases found")
    totals = {f"recall@{k}": 0.0 for k in ks}
    totals.update({"mrr": 0.0, "ndcg@10": 0.0})
    details = []
    for case in cases:
        relevant = set(case["relevant_ids"])
        results = retriever.search(case["query"], top_k=max(max(ks), 10), variant=variant)
        ranked = [result.image_id for result in results]
        for k in ks:
            totals[f"recall@{k}"] += bool(relevant.intersection(ranked[:k]))
        reciprocal = next((1.0 / rank for rank, value in enumerate(ranked, 1) if value in relevant), 0)
        gains = [1.0 if value in relevant else 0.0 for value in ranked[:10]]
        dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
        totals["mrr"] += reciprocal
        totals["ndcg@10"] += dcg / ideal if ideal else 0.0
        details.append({"query": case["query"], "top_ids": ranked[:10]})
    metrics = {key: value / len(cases) for key, value in totals.items()}
    return {"variant": variant, "queries": len(cases), "metrics": metrics, "details": details}


def benchmark_variants(retriever, judgments_path: Path) -> dict:
    """Run the exact same judgments through every ablation variant."""
    runs = [evaluate(retriever, judgments_path, variant=variant) for variant in VARIANTS]
    return {
        "judgments": str(judgments_path),
        "variants": runs,
        "recommended_primary_metric": "ndcg@10",
    }
