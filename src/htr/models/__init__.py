"""Регистр архитектур (§4.x)."""

from __future__ import annotations

import torch.nn as nn

from htr.models.attention_line import build_attention_line
from htr.models.crnn_ctc import build_crnn
from htr.models.resnet_pretrained_line_ctc import build_pretrained_resnet_line_ctc
from htr.models.transformer_line_ctc import build_transformer_line_ctc


def resolve_model(cfg: dict, num_classes: int) -> nn.Module:
    name = str(cfg["model"]["name"]).lower()
    m = dict(cfg["model"])

    if name in ("crnn_ctc", "cnn_bilstm_ctc"):
        return build_crnn(num_classes=num_classes, model_cfg=m)

    if name == "transformer_encoder_line_ctc":
        return build_transformer_line_ctc(num_classes=num_classes, model_cfg=m)

    if name == "pretrained_resnet_line_ctc":
        return build_pretrained_resnet_line_ctc(num_classes=num_classes, model_cfg=m)

    if name == "attention_line_seq2seq":
        return build_attention_line(m, num_charset_classes=num_classes)

    msg = "crnn_ctc, cnn_bilstm_ctc, transformer_encoder_line_ctc, pretrained_resnet_line_ctc, attention_line_seq2seq"
    raise KeyError(f"model.name={name!r}; допустимо: {msg}")
