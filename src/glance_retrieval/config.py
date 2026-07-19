from __future__ import annotations


def validate_config(config: dict) -> dict:
    required = {"models", "runtime", "index", "retrieval"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Config is missing sections: {sorted(missing)}")
    if config["index"].get("backend") not in {"numpy", "faiss"}:
        raise ValueError("index.backend must be 'numpy' or 'faiss'")
    views = config["index"].get("crop_views", [])
    if not views or views[0] != "full":
        raise ValueError("index.crop_views must start with 'full'")
    if int(config["index"].get("candidate_pool", 0)) <= 0:
        raise ValueError("index.candidate_pool must be positive")
    global_weight = float(config["retrieval"].get("fashion_global_weight", -1))
    atomic_weight = float(config["retrieval"].get("fashion_atomic_weight", -1))
    if global_weight < 0 or atomic_weight < 0 or abs(global_weight + atomic_weight - 1) > 1e-6:
        raise ValueError("fashion global and atomic weights must be non-negative and sum to 1")
    if float(config["retrieval"].get("softmin_temperature", 0)) <= 0:
        raise ValueError("retrieval.softmin_temperature must be positive")
    return config

