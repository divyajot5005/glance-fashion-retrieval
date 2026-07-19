import numpy as np

from glance_retrieval.retriever import Retriever, VARIANTS
from glance_retrieval.storage import VectorStore
from glance_retrieval.types import ImageRecord


class FakeEncoder:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode_texts(self, texts):
        values = np.stack([self.mapping.get(text, self.mapping["*"]) for text in texts])
        return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_bound_attributes_outweigh_single_strong_match(tmp_path):
    e0, e1, e2 = np.eye(3, dtype=np.float32)
    fashion = np.array(
        [
            [[0.8, 0.0, 0.6], [0.0, 1.0, 0.0]],  # both atoms across views
            [[0.8, 0.0, 0.6], [1.0, 0.0, 0.0]],  # red tie only
            [[0.0, 0.8, 0.6], [0.0, 1.0, 0.0]],  # white shirt only
        ],
        dtype=np.float32,
    )
    context = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    records = [ImageRecord(str(i), f"{i}.jpg") for i in range(3)]
    VectorStore(tmp_path).save(records, fashion, context, {"crop_views": ["full", "upper"]})
    config = {
        "models": {"fashion": "fake", "context": "fake"},
        "runtime": {"device": "cpu", "use_amp": False},
        "index": {"candidate_pool": 3, "backend": "numpy"},
        "retrieval": {
            "fashion_global_weight": 0.2,
            "fashion_atomic_weight": 0.8,
            "softmin_temperature": 0.05,
            "metadata_boost": 0.0,
        },
    }
    fashion_encoder = FakeEncoder({
        "red tie": e0,
        "white shirt": e1,
        "red tie, white shirt, formal": e2,
        "*": e2,
    })
    context_encoder = FakeEncoder({"office": np.array([1.0, 0.0]), "*": np.array([1.0, 0.0])})
    engine = Retriever(config, tmp_path, encoders=(fashion_encoder, context_encoder))
    results = engine.search("a red tie and a white shirt in a formal office", top_k=3)
    assert results[0].image_id == "0"
    assert results[0].fashion_atomic > results[1].fashion_atomic
    for variant in VARIANTS:
        assert len(engine.search("a red tie and a white shirt in a formal office", 2, variant)) == 2

