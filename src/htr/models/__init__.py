"""Регистр архитектур (§4.x). Добавляйте сборщики здесь."""

from __future__ import annotations

import torch.nn as nn

from htr.models.crnn_ctc import build_crnn


def resolve_model(cfg: dict, num_classes: int) -> nn.Module:
    name = str(cfg["model"]["name"]).lower()
    m = dict(cfg["model"])
    if name in ("crnn_ctc", "cnn_bilstm_ctc"):
        return build_crnn(num_classes=num_classes, model_cfg=m)
    msg = "crnn_ctc, cnn_bilstm_ctc"
    raise KeyError(f"model.name={name!r}; доступно: {msg}")
