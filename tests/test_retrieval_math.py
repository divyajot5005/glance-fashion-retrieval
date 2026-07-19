import numpy as np

from glance_retrieval.retriever import softmin


def test_softmin_penalizes_missing_conjunct():
    balanced = np.array([[0.8, 0.8]], dtype=np.float32)
    missing_one = np.array([[0.99, 0.1]], dtype=np.float32)
    assert softmin(balanced, 0.08)[0] > softmin(missing_one, 0.08)[0]


def test_softmin_is_close_to_min_at_low_temperature():
    values = np.array([[0.2, 0.7, 0.9]], dtype=np.float32)
    assert abs(float(softmin(values, 0.01)[0]) - 0.2) < 0.02

