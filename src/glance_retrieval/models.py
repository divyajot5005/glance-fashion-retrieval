from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image


def l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    denom = np.linalg.norm(values, axis=axis, keepdims=True).clip(min=1e-12)
    return (values / denom).astype(np.float32, copy=False)


class HuggingFaceEncoder:
    """Small adapter over CLIP/SigLIP-family Transformers checkpoints."""

    def __init__(self, model_name: str, device: str = "auto", use_amp: bool = True):
        import torch
        from transformers import AutoModel, AutoProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type in {"cuda", "mps"}
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def _features(self, inputs: dict, method: str) -> np.ndarray:
        import torch

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        amp_device = self.device.type if self.device.type in {"cuda", "cpu"} else "cpu"
        with torch.inference_mode(), torch.autocast(
            amp_device, enabled=self.use_amp and amp_device == "cuda"
        ):
            values = getattr(self.model, method)(**inputs)
        return l2_normalize(values.float().cpu().numpy())

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=list(images), return_tensors="pt")
        return self._features(inputs, "get_image_features")

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(
            text=list(texts), padding=True, truncation=True, return_tensors="pt"
        )
        return self._features(inputs, "get_text_features")

