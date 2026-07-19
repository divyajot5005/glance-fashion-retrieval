import json

from glance_retrieval.evaluation import benchmark_variants, evaluate
from glance_retrieval.types import SearchResult


class FakeRetriever:
    def search(self, query, top_k, variant="proposed"):
        order = ["relevant", "other"] if variant == "proposed" else ["other", "relevant"]
        return [
            SearchResult(
                rank=rank,
                image_id=image_id,
                path=f"{image_id}.jpg",
                score=1 / rank,
                fashion_global=0,
                fashion_atomic=0,
                context=0,
                metadata_bonus=0,
                metadata={},
            )
            for rank, image_id in enumerate(order, 1)
        ]


def test_metrics_and_benchmark_variants(tmp_path):
    judgments = tmp_path / "judgments.jsonl"
    judgments.write_text(
        json.dumps({"query": "q", "relevant_ids": ["relevant"]}) + "\n",
        encoding="utf-8",
    )
    proposed = evaluate(FakeRetriever(), judgments, variant="proposed")
    assert proposed["metrics"]["recall@1"] == 1.0
    assert proposed["metrics"]["mrr"] == 1.0
    report = benchmark_variants(FakeRetriever(), judgments)
    assert len(report["variants"]) == 5
