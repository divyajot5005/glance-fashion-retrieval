import copy

import pytest

from glance_retrieval.config import validate_config


BASE = {
    "models": {"fashion": "a", "context": "b"},
    "runtime": {"device": "cpu", "use_amp": False},
    "index": {"backend": "numpy", "crop_views": ["full"], "candidate_pool": 10},
    "retrieval": {
        "fashion_global_weight": 0.35,
        "fashion_atomic_weight": 0.65,
        "softmin_temperature": 0.08,
    },
}


def test_valid_config_passes():
    assert validate_config(copy.deepcopy(BASE))["index"]["backend"] == "numpy"


def test_invalid_weight_sum_fails():
    config = copy.deepcopy(BASE)
    config["retrieval"]["fashion_atomic_weight"] = 0.5
    with pytest.raises(ValueError, match="sum to 1"):
        validate_config(config)

